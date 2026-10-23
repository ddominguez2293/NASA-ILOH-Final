#!/usr/bin/env python3

# Temp.py
# STEP 3 of the pipeline (runs AFTER Streamflow.py).
#
# Purpose : pull daily-mean USGS water temperature (parameter 00010) for every site inside the
#           AOI, average multiple series / sensors to one value per (site, Date), then FULL JOIN
#           the streamflow table from Streamflow.py.
# Inputs  : data/<aoi>/shapes/aoi.gpkg (Boundaries.py)
#           data/<aoi>/observations/q_obs.parquet, q_sites.parquet (Streamflow.py)
# Outputs : data/<aoi>/observations/q_t.parquet      one row per (site_no, Date): q_m3s, water_temp_C
#           data/<aoi>/observations/sites.parquet    one row per Q and / or T site: site_no, lon, lat
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
import argparse

import pandas as pd

from pipeline_utils import (banner, config_fingerprint, get_paths, is_up_to_date, load_config,
                            read_aoi, require_files, usgs_daily_mean, write_stamp)

STEP = "temp"
CONFIG_KEYS = ["aoi_name", "hucs", "start_date", "end_date"]

## 2. Config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 3 Temp", cfg)

q_sites_path = paths.obs_dir / "q_sites.parquet"
sites_path = paths.obs_dir / "sites.parquet"
outputs = [paths.q_t, sites_path]
fingerprint = config_fingerprint(cfg, CONFIG_KEYS)
if not args.force and is_up_to_date(paths, STEP, fingerprint, outputs):
    print("Up to date, skipping (use --force to rebuild).")
    raise SystemExit(0)

require_files({"aoi_gpkg": paths.aoi_gpkg, "q_obs": paths.q_obs, "q_sites": q_sites_path})

## 3. Pull daily-mean water temperature (00010)
### Query uses the AOI itself (#8: the old hard-coded Illinois bbox missed northern / eastern sites)
aoi = read_aoi(paths)
t_raw, t_sites = usgs_daily_mean("00010", aoi, cfg["start_date"], cfg["end_date"])

## 4. Clean and aggregate sensors
### degC series only, averaged to one value per (site_no, Date)
#### Sites with several daily-mean series (sublocations / multiple sensors) are averaged, as in Temp.Rmd
print(f"  units: {t_raw['unit_of_measure'].value_counts().to_dict()}")
t_raw = t_raw[t_raw["unit_of_measure"] == "degC"]

temp_daily = (
    t_raw.dropna(subset=["value"])
    .groupby(["site_no", "Date"], as_index=False)
    .agg(water_temp_C=("value", "mean"))
)

### QC: flag implausible water temperatures
bad = temp_daily[(temp_daily["water_temp_C"] < -1) | (temp_daily["water_temp_C"] > 40)]
if len(bad):
    print(f"WARNING: {len(bad)} water temperatures outside [-1, 40] degC (kept)")
    print(bad["site_no"].value_counts().head(10))

## 5. Join Q and T
### Full join on (site_no, Date): Q-only and T-only days are both kept
q_obs = pd.read_parquet(paths.q_obs)[["site_no", "Date", "q_m3s"]]
q_t = q_obs.merge(temp_daily, on=["site_no", "Date"], how="outer")

if q_t.duplicated(["site_no", "Date"]).any():
    raise RuntimeError("q_t has duplicate (site_no, Date) rows")

### Combined site table (Q sites + T sites) for snapping in Catchments.py
q_sites = pd.read_parquet(q_sites_path)
sites = (
    pd.concat([q_sites, t_sites[t_sites["site_no"].isin(temp_daily["site_no"])]], ignore_index=True)
    .drop_duplicates("site_no")
    .reset_index(drop=True)
)

## 6. Save q_t
q_t.to_parquet(paths.q_t, index=False)
sites.to_parquet(sites_path, index=False)

print("\nQ+T summary:")
print(f"  rows            : {len(q_t):,}")
print(f"  sites           : {q_t['site_no'].nunique()} "
      f"(Q {q_t.loc[q_t['q_m3s'].notna(), 'site_no'].nunique()}, "
      f"T {q_t.loc[q_t['water_temp_C'].notna(), 'site_no'].nunique()}, "
      f"both {len(set(q_t.loc[q_t['q_m3s'].notna(), 'site_no']) & set(q_t.loc[q_t['water_temp_C'].notna(), 'site_no']))})")
print(f"  dates           : {q_t['Date'].min()} to {q_t['Date'].max()}")
print(f"  pct_rows_with_Q : {q_t['q_m3s'].notna().mean():.3f}")
print(f"  pct_rows_with_T : {q_t['water_temp_C'].notna().mean():.3f}")
print(f"  water_temp_C    : min {q_t['water_temp_C'].min():.1f} | max {q_t['water_temp_C'].max():.1f}")

write_stamp(paths, STEP, fingerprint, outputs)
print(f"Saved {paths.q_t}")
