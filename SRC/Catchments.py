#!/usr/bin/env python3

# Catchments.py
# STEP 4 of the pipeline (runs AFTER Temp.py).
#
# Purpose : read NHDPlus HR for every HU4 in hu4s.json, clip to the AOI, build the routed
#           network (all stream orders), snap every USGS Q / T site to a flowline, and write
#           the stream-order >= streamorder_min model domain.
# Inputs  : data/<aoi>/shapes/aoi.gpkg, hu4s.json (Boundaries.py)
#           data/<aoi>/observations/q_t.parquet, sites.parquet (Temp.py)
#           data/_nhdplushr/NHDPLUS_H_<HU4>_HU4_GDB.zip (downloaded here on first use)
# Outputs : data/<aoi>/nhdplus/network.parquet       every reach: routing (tocomid), areas, stream order
#           data/<aoi>/nhdplus/catchments_all.gpkg   every catchment polygon (Meteo.py basin drivers)
#           data/<aoi>/nhdplus/flowlines.gpkg        stream order >= min (model domain)
#           data/<aoi>/nhdplus/catchments.gpkg       stream order >= min (model domain)
#           data/<aoi>/observations/sites_with_comid.csv
#           data/<aoi>/observations/q_t_comid.parquet  Q + T with comid and streamorde (order >= min)
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
import argparse
import json
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyogrio
import requests
import shapely

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline_utils import (banner, config_fingerprint, get_paths, is_up_to_date, load_config, read_aoi,
                            require_files, ROUTINGS, routing_links, upstream_solver, write_stamp)

warnings.filterwarnings("ignore", message="Measured \\(M\\) geometry types are not supported")

STEP = "catchments"
CONFIG_KEYS = ["aoi_name", "hucs", "start_date", "end_date", "streamorder_min", "snap_crs", "snap_max_distance_m"]
HR_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_{hu4}_HU4_GDB.zip"

## 2. Config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 4 Catchments", cfg)

so_min = int(cfg["streamorder_min"])
sites_path = paths.obs_dir / "sites.parquet"
outputs = [paths.network, paths.catchments_all, paths.flowlines, paths.catchments,
           paths.sites_with_comid, paths.q_t_comid]
fingerprint = config_fingerprint(cfg, CONFIG_KEYS)
if not args.force and is_up_to_date(paths, STEP, fingerprint, outputs):
    print("Up to date, skipping (use --force to rebuild).")
    raise SystemExit(0)

require_files({"aoi_gpkg": paths.aoi_gpkg, "hu4s": paths.hu4_list, "q_t": paths.q_t, "sites": sites_path})
hu4s = json.loads(paths.hu4_list.read_text())["hu4s"]


## 3. Download NHDPlus HR (cached per HU4)
### Raw zips live in data/_nhdplushr/ so every AOI that shares an HU4 reuses them
def hr_gdb(hu4: str) -> str:
    paths.nhd_cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = paths.nhd_cache_dir / f"NHDPLUS_H_{hu4}_HU4_GDB.zip"
    if not zpath.exists():
        print(f"  downloading NHDPlus HR {hu4} ...")
        tmp = zpath.with_suffix(".zip.part")
        with requests.get(HR_URL.format(hu4=hu4), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)
        tmp.rename(zpath)
    return f"/vsizip/{zpath}/NHDPLUS_H_{hu4}_HU4_GDB.gdb"


## 4. Read flowlines, VAA and catchments for every HU4
### VAA carries the routing and areas (replaces the NHDPlus V2 get_vaa() lookup, whose IDs never matched HR)
VAA_COLS = ["NHDPlusID", "StreamOrde", "StreamCalc", "FromNode", "ToNode", "HydroSeq", "DnHydroSeq",
            "LevelPathI", "Divergence", "StartFlag", "TerminalFl", "AreaSqKm", "TotDASqKm", "Slope"]
FL_COLS = ["NHDPlusID", "GNIS_Name", "LengthKM", "ReachCode", "FType"]

vaa_parts, fl_parts, cat_parts = [], [], []
for hu4 in hu4s:
    gdb = hr_gdb(hu4)
    vaa_parts.append(pyogrio.read_dataframe(gdb, layer="NHDPlusFlowlineVAA", columns=VAA_COLS, read_geometry=False))
    fl_parts.append(pyogrio.read_dataframe(gdb, layer="NHDFlowline", columns=FL_COLS))
    cat_parts.append(pyogrio.read_dataframe(gdb, layer="NHDPlusCatchment", columns=["NHDPlusID", "AreaSqKm"]))
    print(f"  {hu4}: {len(fl_parts[-1]):,} flowlines | {len(cat_parts[-1]):,} catchments")

vaa = pd.concat(vaa_parts, ignore_index=True)
fl = gpd.GeoDataFrame(pd.concat(fl_parts, ignore_index=True), crs=fl_parts[0].crs)
cat = gpd.GeoDataFrame(pd.concat(cat_parts, ignore_index=True), crs=cat_parts[0].crs)

### Integer COMIDs (HR stores NHDPlusID as an exact-integer double)
for df in (vaa, fl, cat):
    df["comid"] = df.pop("NHDPlusID").astype(np.int64)
for c in ["FromNode", "ToNode", "HydroSeq", "DnHydroSeq", "LevelPathI"]:
    vaa[c] = vaa[c].astype(np.int64)

vaa = vaa.drop_duplicates("comid")
fl = fl.drop_duplicates("comid")
cat = cat.drop_duplicates("comid")

### Drop Z / M and move to lon / lat
fl["geometry"] = shapely.force_2d(fl.geometry.values)
fl = fl.to_crs(4326)
cat = cat.to_crs(4326)


## 5. Clip to the AOI
### Keep a catchment if its representative point is inside the AOI (100 m buffer)
#### For HU4-only configs this keeps (almost) everything; it matters when HUC8s clip an HU4
#### A point test (not a geometric intersection) keeps catchment polygons whole
aoi = read_aoi(paths)
aoi_5070 = aoi.to_crs(5070).buffer(100).union_all()
shapely.prepare(aoi_5070)


def inside_aoi(gdf):
    pts = gdf.geometry.representative_point().to_crs(5070)
    return shapely.contains(aoi_5070, pts.values)


cat = cat[inside_aoi(cat)]
fl = fl[fl["comid"].isin(cat["comid"]) | inside_aoi(fl)]
fl = fl[fl["comid"].isin(vaa["comid"])]  # network flowlines only (VAA = routed)
print(f"  inside AOI: {len(fl):,} routed flowlines | {len(cat):,} catchments")


## 6. Build the routed network table (all stream orders)
### tocomid = main-path downstream reach (DnHydroSeq -> HydroSeq); -1 = leaves the AOI / terminal
net = (
    fl[["comid", "GNIS_Name", "LengthKM", "ReachCode", "FType"]]
    .merge(vaa, on="comid", how="left")
    .merge(cat[["comid", "AreaSqKm"]].rename(columns={"AreaSqKm": "cat_areasqkm"}), on="comid", how="left")
)
hs_to_comid = pd.Series(net["comid"].values, index=net["HydroSeq"].values)
net["tocomid"] = net["DnHydroSeq"].map(hs_to_comid).fillna(-1).astype(np.int64)

#### Local area: catchment polygon area where present (flowline-only reaches contribute 0)
net["areasqkm"] = net["cat_areasqkm"].fillna(0.0)

net = net.rename(columns={
    "GNIS_Name": "gnis_name", "LengthKM": "lengthkm", "ReachCode": "reachcode", "FType": "ftype",
    "StreamOrde": "streamorde", "StreamCalc": "streamcalc", "FromNode": "fromnode", "ToNode": "tonode",
    "HydroSeq": "hydroseq", "DnHydroSeq": "dnhydroseq", "LevelPathI": "levelpathi", "Divergence": "divergence",
    "StartFlag": "startflag", "TerminalFl": "terminalfl", "TotDASqKm": "totdasqkm", "Slope": "slope",
})

### Accumulate local area downstream under both routings (pipeline_utils.ROUTINGS)
#### split: node topology, flow split equally at divergences (side channels inherit the upstream basin)
#### main : NHDPlus main path only (no area lost where a braid dead-ends)
#### Each reach keeps the routing whose accumulated area is closer to NHDPlus TotDASqKm;
#### Meteo.py builds that reach's bsn_* drivers with the same routing
links = routing_links(net)
td = net["totdasqkm"].to_numpy()
acc = {}
for name, make_links in ROUTINGS.items():
    acc[name] = upstream_solver(net["comid"], net["hydroseq"], make_links(net))(net["areasqkm"].values)
use_main = np.abs(acc["main"] / td - 1) <= np.abs(acc["split"] / td - 1)
net["basin_routing"] = np.where(use_main, "main", "split")
net["acc_areasqkm"] = np.where(use_main, acc["main"], acc["split"])
net["in_model"] = net["streamorde"] >= so_min

print(f"\nRouting links: {len(links):,} ({int((links['frac'] < 1).sum()):,} split at divergences)")
print("Accumulated area vs NHDPlus TotDASqKm, model reaches (share within 5% | below 50%):")
m = net["in_model"].to_numpy() & (td > 0)
for name, a in [("split", acc["split"]), ("main", acc["main"]), ("chosen", net["acc_areasqkm"].to_numpy())]:
    r = a[m] / td[m]
    print(f"  {name:6s} {np.mean(np.abs(r - 1) < 0.05):.4f} | {np.mean(r < 0.5):.4f}")
print(f"  routing used: {net.loc[m, 'basin_routing'].value_counts().to_dict()}")
outlet = net.loc[net["totdasqkm"].idxmax()]
print(f"  largest reach ({outlet['gnis_name']}): accumulated {outlet['acc_areasqkm']:,.0f} of "
      f"{outlet['totdasqkm']:,.0f} km^2 TotDA")

### Missing upstream HUCs
#### A reach with no upstream link inside the AOI but a large TotDASqKm is fed from outside the AOI;
#### its bsn_* drivers will only cover the part of its basin that is inside the AOI
inflow = net[~net["comid"].isin(links["dn_comid"]) & (net["totdasqkm"] - net["areasqkm"] > 50)]
if len(inflow):
    print(f"WARNING: {len(inflow)} reaches receive water from outside the AOI "
          f"({inflow['totdasqkm'].sub(inflow['areasqkm']).sum():,.0f} km^2 of upstream area not modeled);")
    print("         add the upstream HUC(s) to config.yaml if that matters. Largest:")
    print(inflow.sort_values("totdasqkm", ascending=False).head(5)
          [["comid", "gnis_name", "streamorde", "totdasqkm"]].to_string(index=False))
else:
    print("  no inflow from outside the AOI: every upstream area is modeled")

n_orphan = int(((net["fromnode"] == 0) | (net["tonode"] == 0)).sum())
print(f"  orphan reaches (missing from / to node): {n_orphan}")


## 7. Snap sites to COMID (#8: in meters, EPSG:5070)
### Nearest flowline of ANY stream order within snap_max_distance_m
#### Using the full network keeps small-tributary gauges from being pulled onto a mainstem
sites = pd.read_parquet(sites_path)
sites_gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites["lon"], sites["lat"]), crs=4326)
snap_crs = int(cfg["snap_crs"])
max_d = float(cfg["snap_max_distance_m"])

snapped = gpd.sjoin_nearest(
    sites_gdf.to_crs(snap_crs),
    fl[["comid", "geometry"]].to_crs(snap_crs),
    how="left", max_distance=max_d, distance_col="snap_dist_m",
).drop_duplicates("site_no")

sites_with_comid = (
    snapped.dropna(subset=["comid"])
    .assign(comid=lambda d: d["comid"].astype(np.int64))
    .merge(net[["comid", "streamorde", "totdasqkm"]], on="comid", how="left")
    [["site_no", "comid", "snap_dist_m", "streamorde", "totdasqkm"]]
)
n_dropped = len(sites) - len(sites_with_comid)
print(f"\nSites snapped: {len(sites_with_comid)} of {len(sites)} "
      f"({n_dropped} farther than {max_d:.0f} m from any AOI flowline)")
print(f"  snap distance (m): median {sites_with_comid['snap_dist_m'].median():.1f} | "
      f"max {sites_with_comid['snap_dist_m'].max():.1f}")


## 8. Attach stream order to Q + T and filter to the model domain
q_t = pd.read_parquet(paths.q_t)
q_t_comid = (
    q_t.merge(sites_with_comid[["site_no", "comid", "streamorde"]], on="site_no", how="inner")
    .query("streamorde >= @so_min")
)
has_data = q_t_comid.groupby("site_no")[["q_m3s", "water_temp_C"]].transform(lambda s: s.notna().any())
q_t_comid = q_t_comid[has_data.any(axis=1)].reset_index(drop=True)

### Unique Q and T sensors by stream order (all snapped sites, then model domain)
counts = (
    q_t.merge(sites_with_comid[["site_no", "streamorde"]], on="site_no")
    .groupby("streamorde")
    .agg(n_q_sensors=("site_no", lambda s: s[q_t.loc[s.index, "q_m3s"].notna()].nunique()),
         n_t_sensors=("site_no", lambda s: s[q_t.loc[s.index, "water_temp_C"].notna()].nunique()))
)
print("\nUnique sensors by stream order (all snapped sites):")
print(counts.to_string())


## 9. Write outputs
### Network table: every reach in the AOI (routing + areas), used by Meteo.py and Q_Processing_python.py
net_cols = ["comid", "tocomid", "fromnode", "tonode", "hydroseq", "dnhydroseq", "levelpathi", "divergence",
            "startflag", "terminalfl", "streamorde", "streamcalc", "ftype", "gnis_name", "reachcode",
            "lengthkm", "slope", "areasqkm", "totdasqkm", "acc_areasqkm", "basin_routing", "in_model"]
net[net_cols].to_parquet(paths.network, index=False)


def write_gpkg(gdf, path):
    if path.exists():
        path.unlink()
    gdf.to_file(path, engine="pyogrio")


### All catchments (for upstream-basin drivers) and the model-domain layers
write_gpkg(cat[["comid", "AreaSqKm", "geometry"]].rename(columns={"AreaSqKm": "areasqkm"}), paths.catchments_all)

model_comids = set(net.loc[net["in_model"], "comid"])
flow_cols = ["comid", "reachcode", "gnis_name", "lengthkm", "areasqkm", "totdasqkm", "slope", "streamorde",
             "streamcalc", "hydroseq", "levelpathi", "fromnode", "tonode", "terminalfl", "startflag", "divergence"]
flowlines_model = fl[["comid", "geometry"]].merge(net[flow_cols], on="comid")
flowlines_model = flowlines_model[flowlines_model["comid"].isin(model_comids)]
catchments_model = cat[cat["comid"].isin(model_comids)][["comid", "AreaSqKm", "geometry"]].rename(
    columns={"AreaSqKm": "areasqkm"})

write_gpkg(flowlines_model, paths.flowlines)
write_gpkg(catchments_model, paths.catchments)

sites_with_comid.to_csv(paths.sites_with_comid, index=False)
q_t_comid.to_parquet(paths.q_t_comid, index=False)

### Final summary
print(f"\nModel domain (stream order >= {so_min}): {len(flowlines_model):,} flowlines | "
      f"{len(catchments_model):,} catchments")
print("Final Q+T+COMID:")
print(f"  rows {len(q_t_comid):,} | sites {q_t_comid['site_no'].nunique()} | comids {q_t_comid['comid'].nunique()} | "
      f"{q_t_comid['Date'].min()} to {q_t_comid['Date'].max()} | "
      f"pct_Q {q_t_comid['q_m3s'].notna().mean():.3f} | pct_T {q_t_comid['water_temp_C'].notna().mean():.3f}")


## 10. Site map
### AOI, model-domain flowlines (grey) and snapped sites (red)
fig, ax = plt.subplots(figsize=(8, 8))
aoi.plot(ax=ax, color="0.95", edgecolor="black")
flowlines_model.plot(ax=ax, color="0.6", linewidth=0.3)
kept = sites_gdf[sites_gdf["site_no"].isin(q_t_comid["site_no"])]
kept.plot(ax=ax, color="red", markersize=4)
ax.set_title(f"{cfg['aoi_name']}: stream order >= {so_min}, {len(kept)} Q/T sites")
ax.set_axis_off()
fig.savefig(paths.nhd_dir / "site_map.png", dpi=130, bbox_inches="tight")
plt.close(fig)

write_stamp(paths, STEP, fingerprint, outputs)
print(f"Saved {paths.network}, {paths.flowlines}, {paths.catchments}, {paths.q_t_comid}")
