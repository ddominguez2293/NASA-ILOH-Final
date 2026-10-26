#!/usr/bin/env python3

# Meteo_gridmet.py
# STEP 5 of the pipeline (runs AFTER Catchments.py). Alternative to Meteo.py: pulls GridMET instead of
# ERA5-Land. Selected by config.yaml's met_source: gridmet (the default); set met_source: era5 and
# run_pipeline.py uses Meteo.py instead.
#
# Purpose : download GridMET for the AOI bbox via pygridmet (public THREDDS source -- no API key, no
#           request queue, unlike CDS/ERA5-Land), convert to the daily drivers the model reads, and
#           extract them to every model catchment (catchment scale) and to its whole upstream basin
#           (bsn_* = area-weighted mean over every upstream catchment). PET and Ra come straight from
#           GridMET's native "pet" variable and the shared Ra formula (pipeline_utils.ra_mm).
# Inputs  : data/<aoi>/shapes/aoi.gpkg (Boundaries.py)
#           data/<aoi>/nhdplus/network.parquet, catchments_all.gpkg (Catchments.py)
# Outputs : data/<aoi>/meteo/gridmet_raw/<y0>_<y1>.nc            raw GridMET downloads (cached, one file
#                                                                 per --chunk_years window)
#           data/<aoi>/daily_drivers/drivers_daily/<year>.parquet  one row per (comid, date), model catchments
#
# Driver columns (same names / units as Meteo.py, so DDNN_Model.py needs no changes):
#   pr (mm/day), srad (W/m^2), vs (m/s), vpd (kPa), sph (kg/kg), rmin / rmax (%), tmax_c / tmin_c / tmean_c (degC),
#   PET (mm/day), each also as bsn_*; plus lon, lat, DOY, Ra (mm/day), DOY_sin, DOY_cos
#
# Note    : GridMET's archive omits Dec 31 in leap years (~11 missing days total across 1984-2026) --
#           not worth engineering around, given everything else about the record is complete. Every
#           other driver maps directly from GridMET's native fields (pr, srad, vs, sph, rmin, rmax, vpd,
#           tmmx, tmmn, pet); unlike ERA5-Land there's no hourly-to-daily reconstruction to do, since
#           GridMET is already a daily product.
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

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pygridmet as gridmet
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from pipeline_utils import (banner, catchment_basin_weights, config_fingerprint, get_paths, is_up_to_date,
                            load_config, ra_mm, read_aoi, require_files, write_stamp)

STEP = "meteo"
CONFIG_KEYS = ["aoi_name", "hucs", "start_date", "end_date", "streamorder_min", "gridmet"]
DRIVERS = ["pr", "srad", "vs", "vpd", "sph", "rmin", "rmax", "tmax_c", "tmin_c", "tmean_c", "PET"]
GM_VARS = ["pr", "srad", "vs", "sph", "rmin", "rmax", "vpd", "tmmx", "tmmn", "pet"]

### Plausible ranges for the final QC check (same thresholds as Meteo.py)
RANGES = {"pr": (0, 400), "srad": (0, 450), "vs": (0, 30), "vpd": (0, 8), "sph": (0, 0.035),
          "rmin": (0, 100), "rmax": (0, 100), "tmax_c": (-45, 50), "tmin_c": (-50, 40),
          "tmean_c": (-45, 45), "PET": (0, 25), "Ra": (0, 20)}
#### Unlike ERA5-Land's pev, GridMET's pet (grass-reference ET) is a trusted, widely-used product --
#### no WARN_ONLY carve-out needed here.

## 2. Config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
parser.add_argument("--download-only", action="store_true", help="only fetch the GridMET files")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 5 Meteo (GridMET)", cfg)
gm = cfg.get("gridmet", {})
chunk_years = int(gm.get("chunk_years", 5))
buf = float(gm.get("bbox_buffer_deg", 0.1))

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
#### Raw downloads are kept when only the dates changed (same area); per-year files are rebuilt
marker = out_dir / ".fingerprint"
if out_dir.exists() and (args.force or not marker.exists() or marker.read_text() != fingerprint):
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
marker.write_text(fingerprint)


## 3. Request area
### AOI bbox + buffer (pygridmet snaps to the native ~1/24 deg grid internally)
aoi = read_aoi(paths)
w, s, e, n = aoi.total_bounds
bbox = (float(w - buf), float(s - buf), float(e + buf), float(n + buf))
print(f"GridMET request bbox [W, S, E, N]: {bbox}")


## 4. Download GridMET in chunks (cached as one NetCDF per chunk; public source, no API key)
raw_dir = paths.gridmet_raw_dir
raw_dir.mkdir(parents=True, exist_ok=True)
chunks = [(y0, min(y0 + chunk_years - 1, years[-1])) for y0 in range(years[0], years[-1] + 1, chunk_years)]


def chunk_file(y0, y1):
    return raw_dir / f"{y0}_{y1}.nc"


todo = [(y0, y1) for y0, y1 in chunks if not chunk_file(y0, y1).exists()]
print(f"GridMET chunks ({chunk_years} yr each): {len(chunks) - len(todo)} cached, {len(todo)} to download")
for y0, y1 in todo:
    print(f"  {y0}-{y1}: downloading...", flush=True)
    ds = gridmet.get_bygeom(bbox, list(range(y0, y1 + 1)), variables=GM_VARS)
    target = chunk_file(y0, y1)
    ds.to_netcdf(target.with_suffix(".part"))
    target.with_suffix(".part").rename(target)
    print(f"  {y0}-{y1}: done ({ds.sizes['time']} days)", flush=True)

if args.download_only:
    raise SystemExit(0)


## 5. Grid and land mask
#### GridMET is NaN outside CONUS land (e.g. over the Great Lakes); a cell counts as land only if
#### every variable has data on a sample day
probe = xr.open_dataset(chunk_file(*chunks[0]))
lat = probe["lat"].values
lon = probe["lon"].values
if not (np.all(np.diff(lat) < 0) and np.all(np.diff(lon) > 0)):
    raise RuntimeError("Unexpected GridMET grid orientation (expected latitude descending, longitude ascending)")
land = np.ones((len(lat), len(lon)), dtype=bool)
for v in probe.data_vars:
    land &= np.isfinite(probe[v].isel(time=0).values)
ncell = land.size
print(f"Grid: {len(lat)} x {len(lon)} cells, {land.sum()} land")


## 6. Catchment and basin weights (shared with Meteo.py -- see pipeline_utils.catchment_basin_weights)
net = pd.read_parquet(paths.network)
cat = gpd.read_file(paths.catchments_all, engine="pyogrio").to_crs(4326)
model, model_comid, cat_row_model, W_model, Wb = catchment_basin_weights(lat, lon, land, net, cat)

### Latitude / longitude of each model catchment (for Ra and the output table)
cent = cat.geometry.iloc[cat_row_model].to_crs(5070).centroid.to_crs(4326)
m_lon, m_lat = cent.x.to_numpy(), cent.y.to_numpy()


## 7. Daily drivers from GridMET fields
#### Everything maps directly from GridMET's native variables -- no hourly reconstruction, no
#### UTC-offset/accumulation-reset logic, since GridMET is already a daily, already-local-day product.
def flat(da):
    return da.values.reshape(da.shape[0], -1)


def year_fields(y):
    y0, y1 = next((a, b) for a, b in chunks if a <= y <= b)
    ds = xr.open_dataset(chunk_file(y0, y1))
    sel = ds.sel(time=ds["time"].dt.year == y)
    days = pd.DatetimeIndex(sel["time"].values)

    tmax_c = flat(sel["tmmx"]) - 273.15
    tmin_c = flat(sel["tmmn"]) - 273.15
    f = {
        "pr": np.clip(flat(sel["pr"]), 0, None),
        "srad": np.clip(flat(sel["srad"]), 0, None),
        "vs": np.clip(flat(sel["vs"]), 0, None),
        "sph": np.clip(flat(sel["sph"]), 0, None),
        "rmin": np.clip(flat(sel["rmin"]), 0, 100),
        "rmax": np.clip(flat(sel["rmax"]), 0, 100),
        "vpd": np.clip(flat(sel["vpd"]), 0, None),
        "tmax_c": tmax_c,
        "tmin_c": tmin_c,
        "tmean_c": (tmax_c + tmin_c) / 2,  # same definition as Meteo.py / the original GridMET pipeline
        "PET": np.clip(flat(sel["pet"]), 0, None),
    }
    return days, f


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
problems = []
for c in check_cols:
    lo, hi = RANGES[c.removeprefix("bsn_")]
    tol = 1e-3 * max(abs(lo), abs(hi))  # float32 rounding (e.g. rmax = 100.00001)
    mn, mx, mu, nn = summary[f"{c}__min"], summary[f"{c}__max"], summary[f"{c}__mean"], summary[f"{c}__nan"]
    ok = lo - tol <= mn and mx <= hi + tol and nn == 0
    flag = "" if ok else "  <-- CHECK"
    if not ok:
        problems.append(c)
    print(f"  {c:12s} min {mn:9.4f} | mean {mu:9.4f} | max {mx:9.4f} | NaN {nn}{flag}")
if problems:
    raise RuntimeError(f"Driver QC failed for: {problems}")

write_stamp(paths, STEP, fingerprint, outputs)
print(f"Saved {out_dir}")
