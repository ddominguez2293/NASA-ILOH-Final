#!/usr/bin/env python3

# Meteo.py
# STEP 5 of the pipeline (runs AFTER Catchments.py). Replaces Meteo.Rmd (GridMET) and PET.Rmd (Hargreaves).
#
# Purpose : download ERA5-Land for the AOI (all variables per request), convert to the daily drivers the
#           model reads, and extract them to every model catchment (catchment scale) and to its whole
#           upstream basin (bsn_* = area-weighted mean over every upstream catchment). PET comes from
#           ERA5-Land potential evaporation; Ra and DOY_sin / DOY_cos use the same formulas as PET.Rmd.
# Inputs  : data/<aoi>/shapes/aoi.gpkg (Boundaries.py)
#           data/<aoi>/nhdplus/network.parquet, catchments_all.gpkg (Catchments.py)
#           ~/.cdsapirc (CDS API key)
# Outputs : data/<aoi>/meteo/era5_raw/*.nc                        raw ERA5-Land downloads (cached)
#           data/<aoi>/daily_drivers/drivers_daily/<year>.parquet  one row per (comid, date), model catchments
#
# Driver columns (same names / units as the GridMET pipeline, so DDNN_Model.py needs no changes):
#   pr (mm/day), srad (W/m^2), vs (m/s), vpd (kPa), sph (kg/kg), rmin / rmax (%), tmax_c / tmin_c / tmean_c (degC),
#   PET (mm/day), each also as bsn_*; plus lon, lat, DOY, Ra (mm/day), DOY_sin, DOY_cos
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
import argparse
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import cdsapi
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource
from scipy import sparse

from pipeline_utils import (banner, config_fingerprint, get_paths, is_up_to_date, load_config, read_aoi,
                            require_files, ROUTINGS, upstream_solver, write_stamp)

STEP = "meteo"
CONFIG_KEYS = ["aoi_name", "hucs", "start_date", "end_date", "streamorder_min", "era5"]
RES = 0.1  # ERA5-Land grid spacing (degrees)
NC_LOCK = threading.Lock()  # the netCDF / HDF5 library is not thread-safe: downloads run in parallel, file reads do not
DRIVERS = ["pr", "srad", "vs", "vpd", "sph", "rmin", "rmax", "tmax_c", "tmin_c", "tmean_c", "PET"]
INSTANT = ["tmax_c", "tmin_c", "vpd", "rmax", "rmin", "sph", "vs"]  # drivers built from instantaneous variables

### Plausible ranges for the final QC check
RANGES = {"pr": (0, 400), "srad": (0, 450), "vs": (0, 30), "vpd": (0, 8), "sph": (0, 0.035),
          "rmin": (0, 100), "rmax": (0, 100), "tmax_c": (-45, 50), "tmin_c": (-50, 40),
          "tmean_c": (-45, 45), "PET": (0, 25), "Ra": (0, 20)}
#### PET is reported but does not stop the run: ERA5-Land pev is known to be far too high over land
#### (tested 2023: mean 6.5 mm/day, max 50 mm/day); the PET source is still an open decision
WARN_ONLY = {"PET"}

## 2. Config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
parser.add_argument("--download-only", action="store_true", help="only fetch the ERA5 files")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 5 Meteo (ERA5-Land)", cfg)
e5 = cfg["era5"]

start, end = cfg["start_date"], cfg["end_date"]
years = list(range(start.year, end.year + 1))
out_dir = paths.drivers_daily
outputs = [out_dir / f"{y}.parquet" for y in years]
fingerprint = config_fingerprint(cfg, CONFIG_KEYS)
if not args.force and is_up_to_date(paths, STEP, fingerprint, outputs):
    print("Up to date, skipping (use --force to rebuild).")
    raise SystemExit(0)

require_files({"aoi_gpkg": paths.aoi_gpkg, "network": paths.network, "catchments_all": paths.catchments_all})

### Stale outputs from a different config are removed rather than reused
#### Raw downloads are kept when only the dates changed (same area / variables); per-year files are rebuilt
marker = out_dir / ".fingerprint"
if out_dir.exists() and (args.force or not marker.exists() or marker.read_text() != fingerprint):
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
marker.write_text(fingerprint)

off = int(e5["utc_offset_hours"])
if not -23 <= off <= 0:
    raise ValueError("utc_offset_hours must be between -23 and 0 (west of UTC); local-day totals assume it")


## 3. Request area
### AOI bbox + buffer, snapped outward to the 0.1 degree ERA5-Land grid
aoi = read_aoi(paths)
w, s, e, n = aoi.total_bounds
buf = float(e5["bbox_buffer_deg"])
area = [float(round(np.ceil((n + buf) / RES) * RES, 1)), float(round(np.floor((w - buf) / RES) * RES, 1)),
        float(round(np.floor((s - buf) / RES) * RES, 1)), float(round(np.ceil((e + buf) / RES) * RES, 1))]  # N, W, S, E
print(f"ERA5-Land request area [N, W, S, E]: {area}")

### Raw-download cache, keyed on what changes the requests (area, offset, variables)
#### Changing dates reuses every file already downloaded; changing workers etc. does not invalidate anything
raw_key = config_fingerprint({"area": area, "off": off, **{k: e5[k] for k in
                              ["instantaneous", "accumulated", "daily_mean", "daily_maximum", "daily_minimum"]}},
                             ["area", "off", "instantaneous", "accumulated", "daily_mean", "daily_maximum",
                              "daily_minimum"])
raw_dir = paths.era5_raw_dir / raw_key
raw_dir.mkdir(parents=True, exist_ok=True)


## 4. Download ERA5-Land
### One job per request; every variable for a request kind comes in the same request
#### accum_<y>       : accumulated variables at 00 and |offset| UTC, one year (always)
#### points/<lat>_<lon>.parquet : point time series, whole period, one grid cell (instant_source: timeseries,
####                              downloaded in section 6 once the catchment weights say which cells are needed)
#### hourly_<y>_<mm> : instantaneous variables, all 24 hours, one month   (instant_source: hourly)
#### mean_<y>_<mm>, max_<y>, min_<y> : daily-statistics product           (instant_source: daily)
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]
ACC_HOURS = ["00:00", f"{-off:02d}:00"]
source = e5["instant_source"]
if source not in ("timeseries", "hourly", "daily"):
    raise ValueError("era5.instant_source must be 'timeseries', 'hourly' or 'daily'")


def month_list(d0, d1):
    out, y, m = [], d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


#### The last local day also needs the first hours of the next UTC day
tail = end + timedelta(days=1)
months = month_list(start, tail)
hourly = {"data_format": "netcdf", "download_format": "unarchived", "area": area}

jobs = []
for y in years:
    ym = [f"{m:02d}" for yy, m in months if yy == y]
    jobs.append((f"accum_{y}", e5["hourly_dataset"],
                 {"variable": e5["accumulated"], "year": str(y), "month": ym, "day": DAYS, "time": ACC_HOURS, **hourly}))
    if source == "daily":
        for kind, stat in [("max", "daily_maximum"), ("min", "daily_minimum")]:
            jobs.append((f"{kind}_{y}", e5["daily_dataset"], {
                "variable": e5[stat], "year": str(y), "month": ym, "day": DAYS, "daily_statistic": stat,
                "time_zone": f"utc{off:+03d}:00", "frequency": "1_hourly", "area": area}))
if tail.year not in years:
    jobs.append((f"accum_{tail.year}", e5["hourly_dataset"], {
        "variable": e5["accumulated"], "year": str(tail.year), "month": [f"{tail.month:02d}"],
        "day": [f"{tail.day:02d}"], "time": ACC_HOURS, **hourly}))

for y, m in months:
    if source == "timeseries":
        break
    if source == "hourly":
        jobs.append((f"hourly_{y}_{m:02d}", e5["hourly_dataset"], {
            "variable": e5["instantaneous"], "year": str(y), "month": [f"{m:02d}"], "day": DAYS, "time": HOURS,
            **hourly}))
    elif date(y, m, 1) <= end:
        jobs.append((f"mean_{y}_{m:02d}", e5["daily_dataset"], {
            "variable": e5["daily_mean"], "year": str(y), "month": [f"{m:02d}"], "day": DAYS,
            "daily_statistic": "daily_mean", "time_zone": f"utc{off:+03d}:00", "frequency": "1_hourly", "area": area}))


def raw_file(name):
    return raw_dir / f"{name}.nc"


def run_jobs(jobs, fetch, label):
    #### Requests run in parallel; a failure is reported and the rest keep going (rerun to retry)
    print(f"{label}: {len(jobs)} to download ({int(e5['workers'])} parallel requests; CDS queue times vary)", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=int(e5["workers"])) as pool:
        futures = {pool.submit(fetch, j): j for j in jobs}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                print(f"  [{i}/{len(jobs)}] {f.result()}", flush=True)
            except Exception as exc:
                failed.append(futures[f][0])
                print(f"  [{i}/{len(jobs)}] FAILED {futures[f][0]}: {str(exc).splitlines()[-1]}", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} ERA5 requests failed (rerun to retry): {failed[:5]}")


def fetch(job):
    name, dataset, request = job
    target = raw_file(name)
    part = target.with_suffix(".part")
    client = cdsapi.Client(url=e5["cds_url"], quiet=True, progress=False)
    client.retrieve(dataset, request, str(part))

    #### The daily-statistics product returns a zip of one .nc per variable when there are several
    if zipfile.is_zipfile(part):
        unz = target.with_suffix(".unz")
        with zipfile.ZipFile(part) as z:
            z.extractall(unz)
        with NC_LOCK:
            ds = xr.merge([xr.open_dataset(f).load() for f in sorted(unz.glob("*.nc"))], compat="override")
            ds.to_netcdf(target)
        shutil.rmtree(unz)
        part.unlink()
    else:
        part.rename(target)
    return f"downloaded {target.name}"


todo = [j for j in jobs if not raw_file(j[0]).exists()]
print(f"ERA5 gridded files ({source} source): {len(jobs) - len(todo)} cached")
run_jobs(todo, fetch, "ERA5 gridded files")


## 5. Grid and land mask
### Read the grid from one file; ERA5-Land is NaN over water (e.g. Lake Michigan)
#### A cell counts as land only if every variable has data there
def open_raw(name):
    ds = xr.open_dataset(raw_file(name))
    return ds.rename({"valid_time": "time"}) if "valid_time" in ds.dims else ds


first_instant = {"timeseries": [], "hourly": [f"hourly_{months[0][0]}_{months[0][1]:02d}"],
                 "daily": [f"mean_{months[0][0]}_{months[0][1]:02d}", f"max_{years[0]}", f"min_{years[0]}"]}[source]
probe = [open_raw(nm) for nm in [f"accum_{years[0]}"] + first_instant]
lat = probe[0]["latitude"].values
lon = probe[0]["longitude"].values
if not (np.all(np.diff(lat) < 0) and np.all(np.diff(lon) > 0)):
    raise RuntimeError("Unexpected ERA5 grid orientation (expected latitude descending, longitude ascending)")
for ds in probe[1:]:
    if not (np.allclose(ds["latitude"].values, lat) and np.allclose(ds["longitude"].values, lon)):
        raise RuntimeError("ERA5 files are on different grids")

land = np.ones((len(lat), len(lon)), dtype=bool)
for ds in probe:
    for v in ds.data_vars:
        land &= np.isfinite(ds[v].isel(time=1).values)  # time=1: accum at 00 UTC on day 1 can be empty
ncell = land.size
print(f"Grid: {len(lat)} x {len(lon)} cells, {land.sum()} land")


### Hourly instantaneous series (UTC, one point) -> local-day drivers
#### Local day D = hours [D + |offset| h, D + 1 + |offset| h), the same window the gridded hourly path uses
def es(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))  # saturation vapour pressure (kPa), FAO-56


def hourly_to_daily(series):
    df = pd.DataFrame(series).sort_index()
    df.index = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=off)  # UTC -> local standard time
    df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(tail))]
    t = df["t2m"] - 273.15
    ea = es(df["d2m"] - 273.15)
    es_t = es(t)
    sp_kpa = df["sp"] / 1000.0
    h = pd.DataFrame({
        "t": t,
        "vpd": (es_t - ea).clip(lower=0),
        "rh": (100 * ea / es_t).clip(0, 100),
        "sph": 0.622 * ea / (sp_kpa - 0.378 * ea),
        "vs": np.hypot(df["u10"], df["v10"]),
    })
    g = h.groupby(h.index.normalize())
    n_hours = g.size()
    if (n_hours != 24).any():
        raise RuntimeError(f"point series has days without 24 hours: {n_hours[n_hours != 24].head(3).to_dict()}")
    out = pd.DataFrame({
        "date": n_hours.index.date,
        "tmax_c": g["t"].max().to_numpy(), "tmin_c": g["t"].min().to_numpy(),
        "vpd": g["vpd"].mean().to_numpy(), "rmax": g["rh"].max().to_numpy(), "rmin": g["rh"].min().to_numpy(),
        "sph": g["sph"].mean().to_numpy(), "vs": g["vs"].mean().to_numpy(),
    })
    return out.astype({c: np.float32 for c in INSTANT})


## 6. Catchment and basin weights
### Catchment weights: polygon x cell coverage (exactextract) x cell area (cos lat), land cells only
net = pd.read_parquet(paths.network)
cat = gpd.read_file(paths.catchments_all, engine="pyogrio").to_crs(4326)

xmin, xmax = float(lon[0] - RES / 2), float(lon[-1] + RES / 2)
ymin, ymax = float(lat[-1] - RES / 2), float(lat[0] + RES / 2)
grid = NumPyRasterSource(np.ones(land.shape), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
cov = exact_extract(grid, cat[["geometry"]].reset_index(drop=True), ["cell_id", "coverage"], output="pandas")

rows = np.repeat(np.arange(len(cat)), cov["cell_id"].map(len).to_numpy())
cols = np.concatenate(cov["cell_id"].to_numpy()).astype(np.int64)
vals = np.concatenate(cov["coverage"].to_numpy()).astype(np.float64)
cell_lat = np.repeat(lat, len(lon))
vals = vals * np.cos(np.deg2rad(cell_lat[cols]))
keep = land.ravel()[cols]
rows, cols, vals = rows[keep], cols[keep], vals[keep]

#### Catchments that only touch water cells (lake shore) fall back to the nearest land cell
covered = np.bincount(rows, weights=vals, minlength=len(cat)) > 0
if (~covered).any():
    land_idx = np.flatnonzero(land.ravel())
    land_xy = np.c_[np.tile(lon, len(lat))[land_idx], cell_lat[land_idx]]
    cxy = cat.geometry.iloc[np.flatnonzero(~covered)].representative_point()
    near = land_idx[np.argmin((cxy.x.to_numpy()[:, None] - land_xy[:, 0]) ** 2 +
                              (cxy.y.to_numpy()[:, None] - land_xy[:, 1]) ** 2, axis=1)]
    rows = np.r_[rows, np.flatnonzero(~covered)]
    cols = np.r_[cols, near]
    vals = np.r_[vals, np.ones(len(near))]
    print(f"  {int((~covered).sum())} catchments had no land cell; using the nearest land cell")

W_cat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(cat), ncell))
W_cat = sparse.diags(1.0 / np.asarray(W_cat.sum(axis=1)).ravel()) @ W_cat  # rows sum to 1

### Reorder to the network table (flowline-only reaches have no catchment -> empty row, area 0)
row_of = pd.Series(np.arange(len(cat)), index=cat["comid"].to_numpy())
net_row = row_of.reindex(net["comid"].to_numpy()).to_numpy()
has_cat = ~np.isnan(net_row)
pick = sparse.csr_matrix((np.ones(has_cat.sum()), (np.flatnonzero(has_cat), net_row[has_cat].astype(np.int64))),
                         shape=(len(net), len(cat)))
W_net = (pick @ W_cat).tocsr()
area_net = np.where(has_cat, net["areasqkm"].to_numpy(), 0.0)

### Model catchments: stream order >= min and a catchment polygon
model = np.flatnonzero(net["in_model"].to_numpy() & has_cat)
model_comid = net["comid"].to_numpy()[model]
W_model = W_net[model]
print(f"Model catchments: {len(model):,} (of {len(net):,} reaches in the network)")

### Basin weights: area-weighted catchment weights accumulated down the network
#### bsn weight row i = sum over upstream catchments u of area_u * W_u, divided by the upstream area
#### Each model reach uses the routing Catchments.py chose for it (basin_routing: split or main)
#### Solved in blocks of grid cells so the dense right-hand side stays small
AW = (sparse.diags(area_net) @ W_net).tocsc()
used = np.flatnonzero(np.asarray(AW.sum(axis=0)).ravel() > 0)
Wb = np.zeros((len(model), ncell), dtype=np.float32)
routing = net["basin_routing"].to_numpy()[model]
block = 128
for name, make_links in ROUTINGS.items():
    rows_r = np.flatnonzero(routing == name)
    if not len(rows_r):
        continue
    solve = upstream_solver(net["comid"], net["hydroseq"], make_links(net))
    for b0 in range(0, len(used), block):
        cols_b = used[b0:b0 + block]
        Wb[np.ix_(rows_r, cols_b)] = solve(AW[:, cols_b].toarray())[model[rows_r]]
up_area = Wb.sum(axis=1, dtype=np.float64)
if np.any(up_area <= 0):
    raise RuntimeError("Some model catchments have zero upstream area")
#### Same routing as Catchments.py: each row's area must equal that reach's acc_areasqkm
mismatch = np.abs(up_area / net["acc_areasqkm"].to_numpy()[model] - 1)
if mismatch.max() > 1e-3:
    raise RuntimeError(f"Basin weights disagree with network.parquet acc_areasqkm (max rel. diff {mismatch.max():.4f})")
Wb = sparse.csr_matrix(Wb / up_area[:, None].astype(np.float32))
Wb.data[Wb.data < 1e-9] = 0
Wb.eliminate_zeros()
print(f"Basin weights: {Wb.nnz:,} nonzeros "
      f"(median {np.median(np.diff(Wb.indptr)):.0f} cells per basin, max {np.diff(Wb.indptr).max()})")

### Point time series for every grid cell that carries catchment weight (instant_source: timeseries)
#### One request per cell returns the whole period, all instantaneous variables; each is reduced to local
#### days right away (date, tmax_c, tmin_c, vpd, rmax, rmin, sph, vs) and cached as a small parquet
cells_used = np.flatnonzero(np.asarray(abs(W_net).sum(axis=0)).ravel() > 0)
cell_lon = np.tile(lon, len(lat))
point_dir = raw_dir / "points"


def point_file(c):
    return point_dir / f"{cell_lat[c]:.1f}_{cell_lon[c]:.1f}.parquet"


def fetch_point(job):
    c = job[0]
    target = point_file(c)
    part = target.with_suffix(".zip")
    client = cdsapi.Client(url=e5["cds_url"], quiet=True, progress=False)
    client.retrieve(e5["timeseries_dataset"], {
        "variable": e5["instantaneous"], "location": {"latitude": float(cell_lat[c]), "longitude": float(cell_lon[c])},
        "date": [f"{start.isoformat()}/{tail.isoformat()}"], "data_format": "netcdf"}, str(part))
    unz = target.with_suffix(".unz")
    if unz.exists():
        shutil.rmtree(unz)
    with zipfile.ZipFile(part) as z:
        z.extractall(unz)
    with NC_LOCK:
        ds = xr.merge([xr.open_dataset(f).load() for f in sorted(unz.glob("*.nc"))], compat="override")
    shutil.rmtree(unz)
    part.unlink()
    if abs(float(ds["latitude"]) - cell_lat[c]) > 0.01 or abs(float(ds["longitude"]) - cell_lon[c]) > 0.01:
        raise RuntimeError(f"point request for cell {c} returned a different grid point")
    daily = hourly_to_daily({v: ds[v].to_series() for v in ["t2m", "d2m", "sp", "u10", "v10"]})
    daily.to_parquet(target.with_suffix(".tmp"), index=False)
    target.with_suffix(".tmp").rename(target)
    return f"downloaded {target.name}"


if source == "timeseries":
    point_dir.mkdir(parents=True, exist_ok=True)
    #### Leftovers from an interrupted run are discarded (only finished .parquet files count as cached)
    for leftover in list(point_dir.glob("*.zip")) + list(point_dir.glob("*.tmp")) + list(point_dir.glob("*.unz")):
        shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
    todo = [(c,) for c in cells_used if not point_file(c).exists()]
    print(f"ERA5 point series: {len(cells_used)} grid cells carry catchment weight, {len(cells_used) - len(todo)} cached")
    run_jobs(todo, fetch_point, "ERA5 point series")

if args.download_only:
    raise SystemExit(0)

#### Load every cached point series once: (all days, used cells) per driver
if source == "timeseries":
    all_days = pd.date_range(start, end, freq="D")
    POINT = {v: np.empty((len(all_days), len(cells_used)), dtype=np.float32) for v in INSTANT}
    for j, c in enumerate(cells_used):
        d = pd.read_parquet(point_file(c)).set_index("date").reindex(all_days.date)
        if d[INSTANT].isna().any().any():
            raise RuntimeError(f"{point_file(c).name} does not cover {start} to {end}; delete it and rerun")
        for v in INSTANT:
            POINT[v][:, j] = d[v].to_numpy()


### Latitude / longitude of each model catchment (for Ra and the output table)
cent = cat.geometry.iloc[net_row[model].astype(np.int64)].to_crs(5070).centroid.to_crs(4326)
m_lon, m_lat = cent.x.to_numpy(), cent.y.to_numpy()


## 7. Daily drivers from ERA5-Land fields
def flat(da):
    return da.values.reshape(da.shape[0], -1)


def concat_existing(names):
    parts = [open_raw(nm) for nm in names if raw_file(nm).exists()]
    return xr.concat(parts, dim="time").sortby("time")


def year_fields(y):
    d0, d1 = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
    days = pd.date_range(d0, d1, freq="D")
    nd = len(days)

    ### Accumulated variables: local-day totals from accumulations that reset at 00 UTC
    acc = concat_existing([f"accum_{y}", f"accum_{y + 1}"])
    t00 = days + pd.Timedelta(days=1)
    tof = days + pd.Timedelta(hours=-off)
    tof1 = t00 + pd.Timedelta(hours=-off)
    missing = [t for t in t00.append(tof).append(tof1) if t not in acc["time"].values]
    if missing:
        raise RuntimeError(f"ERA5 accumulated fields missing for {len(missing)} times, e.g. {missing[:3]}")

    def local_total(v):
        a = acc[v]
        return flat(a.sel(time=t00)) - flat(a.sel(time=tof)) + flat(a.sel(time=tof1))

    f = {
        "pr": np.clip(local_total("tp") * 1000, 0, None),  # m -> mm/day
        "srad": np.clip(local_total("ssrd") / 86400, 0, None),  # J/m^2/day -> W/m^2
        "PET": np.clip(-local_total("pev") * 1000, 0, None),  # m (negative = evaporation) -> mm/day
    }

    ### Instantaneous variables
    if source == "timeseries":
        #### Already local days (section 6); scatter the used cells back onto the full grid
        i0 = (d0 - start).days
        for v in INSTANT:
            g = np.zeros((nd, ncell), dtype=np.float32)
            g[:, cells_used] = POINT[v][i0:i0 + nd]
            f[v] = g
        f["tmean_c"] = (f["tmax_c"] + f["tmin_c"]) / 2  # same definition as the GridMET pipeline
    elif source == "hourly":
        #### Local day D = [D + |offset| h, D + 1 + |offset| h) in UTC; hours reshaped to (day, hour, cell)
        ms = [nm for nm in (f"hourly_{yy}_{m:02d}" for yy, m in months) if nm.startswith(f"hourly_{y}_")]
        ms += [f"hourly_{y + 1}_01"]
        hr = concat_existing(ms)
        t0 = pd.Timestamp(d0) + pd.Timedelta(hours=-off)
        hr = hr.sel(time=slice(t0, t0 + pd.Timedelta(hours=24 * nd - 1)))
        if hr.sizes["time"] != 24 * nd:
            raise RuntimeError(f"{y}: expected {24 * nd} hourly fields, found {hr.sizes['time']}")

        def hours(v):
            return flat(hr[v]).reshape(nd, 24, -1)

        t = hours("t2m") - 273.15
        ea = es(hours("d2m") - 273.15)
        es_t = es(t)
        rh = np.clip(100 * ea / es_t, 0, 100)
        sp_kpa = hours("sp") / 1000.0
        tmax, tmin = t.max(axis=1), t.min(axis=1)
        f.update({
            "tmax_c": tmax,
            "tmin_c": tmin,
            "tmean_c": (tmax + tmin) / 2,  # same definition as the GridMET pipeline
            "vpd": np.clip(es_t - ea, 0, None).mean(axis=1),
            "rmax": rh.max(axis=1),
            "rmin": rh.min(axis=1),
            "sph": (0.622 * ea / (sp_kpa - 0.378 * ea)).mean(axis=1),
            "vs": np.hypot(hours("u10"), hours("v10")).mean(axis=1),
        })
    else:
        #### Daily-statistics product (already in local days); RH min / max approximated from daily-mean dewpoint
        sel = dict(time=days)
        mean = concat_existing([f"mean_{yy}_{m:02d}" for yy, m in months if yy == y]).sel(**sel)
        mx, mn = open_raw(f"max_{y}").sel(**sel), open_raw(f"min_{y}").sel(**sel)
        tmax = flat(mx["t2m"]) - 273.15
        tmin = flat(mn["t2m"]) - 273.15
        ea = es(flat(mean["d2m"]) - 273.15)
        sp_kpa = flat(mean["sp"]) / 1000.0
        f.update({
            "tmax_c": tmax,
            "tmin_c": tmin,
            "tmean_c": (tmax + tmin) / 2,
            "vpd": np.clip((es(tmax) + es(tmin)) / 2 - ea, 0, None),
            "rmax": np.clip(100 * ea / es(tmin), 0, 100),  # FAO-56: humidity peaks at tmin
            "rmin": np.clip(100 * ea / es(tmax), 0, 100),  # and bottoms out at tmax
            "sph": 0.622 * ea / (sp_kpa - 0.378 * ea),
            "vs": np.hypot(flat(mean["u10"]), flat(mean["v10"])),  # speed of the mean wind vector
        })
    return days, f


### Extraterrestrial radiation Ra (mm/day), FAO-56; same formula as PET.Rmd
def ra_mm(doy, lat_deg):
    phi = np.deg2rad(lat_deg)[None, :]
    j = doy[:, None]
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * j)
    dec = 0.409 * np.sin(2 * np.pi / 365 * j - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return 0.408 * (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


## 8. Extract and write one parquet per year
### Catchment value = W_model @ field; basin value = Wb @ field (both rows sum to 1)
#### Water cells are zeroed before the product (they carry zero weight)
water = ~land.ravel()
for y in years:
    target = out_dir / f"{y}.parquet"
    if target.exists():
        print(f"  {y}: exists, skipping")
        continue

    days, fields = year_fields(y)
    nd, nm = len(days), len(model)
    cols = {
        "comid": np.tile(model_comid, nd),
        "date": np.repeat(days.values.astype("datetime64[D]"), nm),
    }
    for name in DRIVERS:
        g = np.nan_to_num(fields[name], nan=0.0)
        g[:, water] = 0.0
        cols[name] = np.asarray(W_model @ g.T, dtype=np.float32).T.ravel()
        cols[f"bsn_{name}"] = np.asarray(Wb @ g.T.astype(np.float32), dtype=np.float32).T.ravel()

    doy = days.dayofyear.to_numpy()
    cols["lon"] = np.tile(m_lon, nd).astype(np.float32)
    cols["lat"] = np.tile(m_lat, nd).astype(np.float32)
    cols["DOY"] = np.repeat(doy, nm).astype(np.int16)
    cols["Ra"] = ra_mm(doy, m_lat).astype(np.float32).ravel()
    cols["DOY_sin"] = np.repeat(np.sin(2 * np.pi * doy / 365), nm).astype(np.float32)
    cols["DOY_cos"] = np.repeat(np.cos(2 * np.pi * doy / 365), nm).astype(np.float32)

    tmp = target.with_suffix(".part")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    tmp.rename(target)
    print(f"  {y}: {nd} days x {nm:,} catchments | pr mean {cols['pr'].mean():.2f} mm/day | "
          f"tmean {cols['tmean_c'].mean():.1f} C | PET mean {cols['PET'].mean():.2f} mm/day", flush=True)


## 9. QC
### Value ranges and missingness across all years
lf = pl.scan_parquet(str(out_dir / "*.parquet"))
check_cols = DRIVERS + [f"bsn_{d}" for d in DRIVERS] + ["Ra"]
summary = lf.select(
    [pl.len().alias("n_rows"), pl.col("comid").n_unique().alias("n_comids"),
     pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")]
    + [pl.col(c).min().alias(f"{c}__min") for c in check_cols]
    + [pl.col(c).max().alias(f"{c}__max") for c in check_cols]
    + [pl.col(c).mean().alias(f"{c}__mean") for c in check_cols]
    + [pl.col(c).is_nan().sum().alias(f"{c}__nan") for c in check_cols]
).collect().row(0, named=True)

print(f"\nDrivers: {summary['n_rows']:,} rows | {summary['n_comids']:,} comids | "
      f"{summary['min_date']} to {summary['max_date']}")
problems, warnings_only = [], []
for c in check_cols:
    lo, hi = RANGES[c.removeprefix("bsn_")]
    tol = 1e-3 * max(abs(lo), abs(hi))  # float32 rounding (e.g. rmax = 100.00001)
    mn, mx, mu, nn = summary[f"{c}__min"], summary[f"{c}__max"], summary[f"{c}__mean"], summary[f"{c}__nan"]
    ok = lo - tol <= mn and mx <= hi + tol and nn == 0
    flag = "" if ok else "  <-- CHECK"
    if not ok:
        (warnings_only if c.removeprefix("bsn_") in WARN_ONLY else problems).append(c)
    print(f"  {c:12s} min {mn:9.4f} | mean {mu:9.4f} | max {mx:9.4f} | NaN {nn}{flag}")
if warnings_only:
    print(f"WARNING: {warnings_only} outside the plausible range. ERA5-Land potential evaporation is computed for a "
          f"saturated surface and runs far above land PET (DDNN_Model.py does not use PET at the moment).")
if problems:
    raise RuntimeError(f"Driver QC failed for: {problems}")

write_stamp(paths, STEP, fingerprint, outputs)
print(f"Saved {out_dir}")
