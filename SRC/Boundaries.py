#!/usr/bin/env python3

# Boundaries.py
# STEP 1 of the pipeline (first script; nothing runs before it).
#
# Purpose : download the WBD boundary for every HUC listed in config.yaml (any mix of
#           HUC2 / HUC4 / HUC8), dissolve them into one AOI, and record which HU4s
#           NHDPlus HR has to be downloaded for (Catchments.py).
# Inputs  : config.yaml (hucs, aoi_name)
# Outputs : data/<aoi>/shapes/aoi.gpkg    layer "aoi"  (dissolved AOI, EPSG:4326) <- read by every later step
#                                         layer "hucs" (one polygon per configured HUC)
#           data/<aoi>/shapes/hu4s.json   (HU4 codes to download + the HUC8 clip list)
#           data/<aoi>/shapes/aoi_map.png (quick-look map)
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

import geopandas as gpd
import matplotlib
import pandas as pd
from pynhd import WaterData

matplotlib.use("Agg")  # no GUI window, so the step runs unattended
import matplotlib.pyplot as plt

from pipeline_utils import banner, config_fingerprint, get_paths, is_up_to_date, load_config, write_stamp

STEP = "boundaries"
CONFIG_KEYS = ["aoi_name", "hucs"]

## 2. Config
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 1 Boundaries", cfg)

outputs = [paths.aoi_gpkg, paths.hu4_list]
fingerprint = config_fingerprint(cfg, CONFIG_KEYS)
if not args.force and is_up_to_date(paths, STEP, fingerprint, outputs):
    print("Up to date, skipping (use --force to rebuild).")
    raise SystemExit(0)

## 3. Drop nested HUCs
### A HUC that sits inside another configured HUC adds nothing
#### e.g. ["0712", "07120001"] -> keep "0712" only
hucs = cfg["hucs"]
nested = [h for h in hucs if any(h != o and h.startswith(o) for o in hucs)]
if nested:
    print(f"WARNING: dropping HUCs already covered by a broader HUC in the list: {nested}")
hucs = [h for h in hucs if h not in nested]

## 4. Download WBD polygons
### One WaterData query per HUC level
#### wbd02 / wbd04 / wbd08 are the WaterData layers that hold each level's boundaries
LAYER = {2: ("wbd02", "huc2"), 4: ("wbd04", "huc4"), 8: ("wbd08", "huc8")}

parts = []
for level, (layer, col) in LAYER.items():
    codes = [h for h in hucs if len(h) == level]
    if not codes:
        continue
    g = WaterData(layer).byid(col, codes)
    found = set(g[col].astype(str))
    absent = sorted(set(codes) - found)
    if absent:
        raise ValueError(f"WBD has no {col.upper()} for: {absent}")
    g = g.rename(columns={col: "huc"})[["huc", "geometry"]]
    g["huc_level"] = level
    parts.append(g)

hucs_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs(4326)

## 5. Resolve HU4s for the NHDPlus HR download
### HUC4 -> itself, HUC8 -> its parent HU4, HUC2 -> every child HU4
#### HUC8 entries are recorded separately so Catchments.py clips the HU4 network back to them
hu4s = set()
for h in hucs:
    if len(h) in (4, 8):
        hu4s.add(h[:4])
    else:
        children = WaterData("wbd04").byfilter(f"huc4 LIKE '{h}%'")
        hu4s.update(children["huc4"].astype(str))

hu4s = sorted(hu4s)
clip_huc8s = sorted(h for h in hucs if len(h) == 8)

print(f"Configured HUCs : {hucs}")
print(f"HU4s to download: {hu4s}")
if clip_huc8s:
    print(f"HUC8 clip list  : {clip_huc8s}")

## 6. Dissolve into one AOI and save
### aoi.gpkg layer "aoi" is the AOI every later step reads
#### Area is reported in EPSG:5070 (equal-area, meters)
aoi = hucs_gdf.dissolve()[["geometry"]].reset_index(drop=True)
aoi["aoi_name"] = cfg["aoi_name"]
aoi["hucs"] = ",".join(hucs)
aoi["area_km2"] = aoi.to_crs(5070).area / 1e6

if paths.aoi_gpkg.exists():
    paths.aoi_gpkg.unlink()  # gpkg layers append otherwise
aoi.to_file(paths.aoi_gpkg, layer="aoi", engine="pyogrio")
hucs_gdf.to_file(paths.aoi_gpkg, layer="hucs", engine="pyogrio")

paths.hu4_list.write_text(json.dumps({"hucs": hucs, "hu4s": hu4s, "clip_huc8s": clip_huc8s}, indent=2))

### Sanity check: AOI should be a single (multi)polygon
n_parts = len(aoi.geometry.iloc[0].geoms) if aoi.geom_type.iloc[0] == "MultiPolygon" else 1
print(f"AOI area: {aoi['area_km2'].iloc[0]:,.0f} km^2 | polygon parts: {n_parts}")
if n_parts > 1:
    print("WARNING: AOI is not contiguous; the river network will be split into disconnected pieces.")
print(f"AOI bounds (lon/lat): {[round(v, 3) for v in aoi.total_bounds]}")

## 7. Quick-look map
### Saved to PNG instead of shown, so the step never blocks
fig, ax = plt.subplots(figsize=(7, 7))
hucs_gdf.plot(ax=ax, column="huc", cmap="Pastel1", edgecolor="grey", legend=True)
aoi.boundary.plot(ax=ax, color="black", linewidth=1)
ax.set_title(f"AOI: {cfg['aoi_name']} ({', '.join(hucs)})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.savefig(paths.shape_dir / "aoi_map.png", dpi=120, bbox_inches="tight")
plt.close(fig)

write_stamp(paths, STEP, fingerprint, outputs)
print(f"Saved {paths.aoi_gpkg} and {paths.hu4_list}")
