#!/usr/bin/env python3

# Streamflow.py
# STEP 2 of the pipeline (runs AFTER Boundaries.py).
#
# Purpose : pull daily-mean USGS streamflow (parameter 00060) for every site inside the AOI
#           and convert it to m^3/s. No COMID mapping here; COMID is assigned in Catchments.py.
# Inputs  : data/<aoi>/shapes/aoi.gpkg (Boundaries.py)
# Outputs : data/<aoi>/observations/q_obs.parquet   one row per (site_no, Date): Flow (cfs), q_m3s
#           data/<aoi>/observations/q_sites.parquet one row per site: site_no, lon, lat
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
import argparse

from pipeline_utils import (banner, config_fingerprint, get_paths, is_up_to_date, load_config,
                            read_aoi, require_files, usgs_daily_mean, write_stamp)

STEP = "streamflow"
CONFIG_KEYS = ["aoi_name", "hucs", "start_date", "end_date"]
CFS_TO_M3S = 0.0283168  # 1 cfs = 0.0283168 m^3/s

## 2. Config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 2 Streamflow", cfg)

q_sites_path = paths.obs_dir / "q_sites.parquet"
outputs = [paths.q_obs, q_sites_path]
fingerprint = config_fingerprint(cfg, CONFIG_KEYS)
if not args.force and is_up_to_date(paths, STEP, fingerprint, outputs):
    print("Up to date, skipping (use --force to rebuild).")
    raise SystemExit(0)

require_files({"aoi_gpkg": paths.aoi_gpkg})

## 3. Pull daily-mean streamflow (00060)
### Daily-mean series inside the AOI, downloaded in batches
aoi = read_aoi(paths)
q_raw, q_sites = usgs_daily_mean("00060", aoi, cfg["start_date"], cfg["end_date"])

## 4. Clean and convert units
### Keep cfs series only, convert to m^3/s, one row per (site_no, Date)
#### A site can carry more than one daily-mean series (e.g. sublocations); those are averaged
units = q_raw["unit_of_measure"].value_counts()
print(f"  units: {units.to_dict()}")
q_raw = q_raw[q_raw["unit_of_measure"] == "ft^3/s"]

q_obs = (
    q_raw.dropna(subset=["value"])
    .groupby(["site_no", "Date"], as_index=False)
    .agg(Flow=("value", "mean"), approval_status=("approval_status", "first"))
)
q_obs["q_m3s"] = q_obs["Flow"] * CFS_TO_M3S
q_obs = q_obs[["site_no", "Date", "Flow", "q_m3s", "approval_status"]]

q_sites = q_sites[q_sites["site_no"].isin(q_obs["site_no"])]

## 5. QC
### Flag physically impossible values (negative flow)
n_neg = int((q_obs["q_m3s"] < 0).sum())
if n_neg:
    print(f"WARNING: {n_neg} negative streamflow values (kept; check the sites below)")
    print(q_obs.loc[q_obs["q_m3s"] < 0, "site_no"].value_counts().head(10))

if q_obs.duplicated(["site_no", "Date"]).any():
    raise RuntimeError("q_obs has duplicate (site_no, Date) rows")

## 6. Save q_obs
q_obs.to_parquet(paths.q_obs, index=False)
q_sites.to_parquet(q_sites_path, index=False)

print("\nQ summary:")
print(f"  rows     : {len(q_obs):,}")
print(f"  sites    : {q_obs['site_no'].nunique()}")
print(f"  dates    : {q_obs['Date'].min()} to {q_obs['Date'].max()}")
print(f"  q_m3s    : min {q_obs['q_m3s'].min():.3f} | median {q_obs['q_m3s'].median():.2f} "
      f"| max {q_obs['q_m3s'].max():,.1f}")
print(f"  approval : {q_obs['approval_status'].value_counts().to_dict()}")

write_stamp(paths, STEP, fingerprint, outputs)
print(f"Saved {paths.q_obs}")
