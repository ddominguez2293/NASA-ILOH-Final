#!/usr/bin/env python3

# Q_Processing_python.py
# STEP 6 of the pipeline (runs AFTER Meteo.py; the model script runs after this).
#
# Purpose : join Q / T observations to the ERA5 daily drivers, add static catchment / flowline
#           attributes, and write the model input tables.
# Inputs  : data/<aoi>/daily_drivers/drivers_daily/<year>.parquet  (Meteo.py)
#           data/<aoi>/observations/q_t_comid.parquet               (Catchments.py, Q/T already carry COMID)
#           data/<aoi>/nhdplus/network.parquet                      (Catchments.py)
# Outputs : data/<aoi>/daily_drivers/hydro_nodes_allcomids_parquet/nodes.parquet
#           data/<aoi>/daily_drivers/edge_list.parquet
#           data/<aoi>/daily_drivers/hydro_temp_data_analysis_pre2013.parquet   (date <  analysis_cut_date)
#           data/<aoi>/daily_drivers/hydro_temp_data_trainval_2013plus.parquet  (date >= analysis_cut_date,
#                                                                                stream order >= trainval_streamorder_min)
#           edge_list.parquet + hydro_temp_data_trainval_2013plus.parquet are also copied to data/daily_drivers/,
#           the fixed folder DDNN_Model.py reads from
#
# Note    : the split filenames keep "2013" because DDNN_Model.py reads them by name; the actual
#           cut date is analysis_cut_date in config.yaml.
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

import polars as pl

from pipeline_utils import (banner, config_fingerprint, get_paths, is_up_to_date, load_config,
                            require_files, write_stamp)

STEP = "q_processing"
CONFIG_KEYS = ["aoi_name", "hucs", "start_date", "end_date", "streamorder_min", "era5",
               "analysis_cut_date", "trainval_streamorder_min", "model_split_date"]

## 2. Config and paths
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
parser.add_argument("--force", action="store_true", help="rebuild even if outputs are up to date")
args = parser.parse_args()

cfg = load_config(args.config)
paths = get_paths(cfg)
banner("STEP 6 Q_Processing", cfg)

drivers_glob = str(paths.drivers_daily / "*.parquet")

out_nodes_parquet_dir = paths.drivers_dir / "hydro_nodes_allcomids_parquet"
out_nodes_parquet = out_nodes_parquet_dir / "nodes.parquet"
out_edges_parquet = paths.drivers_dir / "edge_list.parquet"
out_analysis_parquet = paths.drivers_dir / "hydro_temp_data_analysis_pre2013.parquet"
out_trainval_parquet = paths.drivers_dir / "hydro_temp_data_trainval_2013plus.parquet"

outputs = [out_nodes_parquet, out_edges_parquet, out_analysis_parquet, out_trainval_parquet,
           paths.model_input_dir / out_edges_parquet.name, paths.model_input_dir / out_trainval_parquet.name]
fingerprint = config_fingerprint(cfg, CONFIG_KEYS)
if not args.force and is_up_to_date(paths, STEP, fingerprint, outputs):
    print("Up to date, skipping (use --force to rebuild).")
    raise SystemExit(0)


## 3. Required file checks
### Stop early if any input is missing
require_files({"network": paths.network, "q_t_comid": paths.q_t_comid, "drivers_daily": paths.drivers_daily})
if not list(paths.drivers_daily.glob("*.parquet")):
    raise FileNotFoundError(f"No driver files in {paths.drivers_daily}; run Meteo.py first")


## 4. Build STATICS table
### Local catchment area, total upstream drainage area (#2), slope, length and stream order
#### tot_area_m2 tells the model how much water enters the top of a reach whose upstream
#### lower-order tributaries were cut by the stream-order filter (the GAT cannot see them)
#### Only reaches with a catchment polygon (areasqkm > 0) are model nodes
statics = (
    pl.read_parquet(paths.network)
    .filter(pl.col("in_model") & (pl.col("areasqkm") > 0))
    .select([
        pl.col("comid").cast(pl.Int64),
        pl.col("slope").cast(pl.Float64).alias("Slope"),
        pl.col("lengthkm").cast(pl.Float64).alias("LengthKM"),
        pl.col("streamorde").cast(pl.Int32),
        (pl.col("areasqkm").cast(pl.Float64) * 1e6).alias("area_m2"),
        (pl.col("totdasqkm").cast(pl.Float64) * 1e6).alias("tot_area_m2"),
        pl.col("fromnode").cast(pl.Int64),
        pl.col("tonode").cast(pl.Int64),
    ])
    .unique(subset=["comid"], keep="first")
)

print("\nSTATICS summary:")
print(statics.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("area_m2").is_not_null().mean().alias("pct_with_area_m2"),
    pl.col("tot_area_m2").is_not_null().mean().alias("pct_with_tot_area_m2"),
    pl.col("Slope").is_not_null().mean().alias("pct_with_Slope"),
    pl.col("LengthKM").is_not_null().mean().alias("pct_with_LengthKM"),
    pl.col("streamorde").min().alias("min_streamorde"),
]))


## 5. Read ERA5 drivers
### Lazy scan of the per-year driver files
drivers = (
    pl.scan_parquet(drivers_glob)
    .with_columns([pl.col("comid").cast(pl.Int64), pl.col("date").cast(pl.Date)])
)

driver_check = drivers.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
    pl.col("PET").is_not_null().mean().alias("pct_PET"),
    pl.col("bsn_PET").is_not_null().mean().alias("pct_bsn_PET"),
]).collect()

print("\nDRIVERS input summary:")
print(driver_check)


## 6. Read Q/T observations
### q_t_comid.parquet has: site_no, Date, q_m3s, water_temp_C, comid, streamorde
qtobs_raw = (
    pl.scan_parquet(str(paths.q_t_comid))
    .with_columns([
        pl.col("site_no").cast(pl.Utf8),
        pl.col("Date").cast(pl.Date),
        pl.col("comid").cast(pl.Int64),
        pl.col("q_m3s").cast(pl.Float64),
        pl.col("water_temp_C").cast(pl.Float64),
    ])
)

print("\nQ/T COMID input by stream order:")
print(
    qtobs_raw.group_by("streamorde")
    .agg([
        pl.col("site_no").n_unique().alias("n_sites"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("q_m3s").is_not_null().sum().alias("n_q_obs"),
        pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
    ])
    .sort("streamorde")
    .collect()
)

### Aggregate to one row per COMID-date
#### Protects against multiple sites / sensors mapping to the same COMID and date
qtobs = (
    qtobs_raw
    .select([
        pl.col("comid"),
        pl.col("Date").alias("date"),
        pl.col("q_m3s").alias("q_m3s_obs"),
        pl.col("water_temp_C"),
    ])
    .group_by(["comid", "date"])
    .agg([
        pl.col("q_m3s_obs").mean(),
        pl.col("water_temp_C").mean(),
    ])
)


## 7. Build NODES
### Left-join Q/T obs and statics onto the drivers (one row per comid-date)
#### Inner join to statics keeps only model nodes (drivers already cover exactly these)
nodes = (
    drivers
    .join(statics.lazy().drop(["fromnode", "tonode"]), on="comid", how="inner")
    .join(qtobs, on=["comid", "date"], how="left")
)

### Observations whose COMID has no drivers would be silently lost; report them
lost = qtobs.join(statics.lazy().select("comid"), on="comid", how="anti").select(pl.len()).collect().item()
if lost:
    print(f"WARNING: {lost} Q/T COMID-days have no model node (no catchment polygon) and are dropped")


## 8. Write nodes parquet
### Stream the lazy frame to zstd parquet, then summarize from disk
out_nodes_parquet_dir.mkdir(parents=True, exist_ok=True)
nodes.sink_parquet(str(out_nodes_parquet), compression="zstd")
print(f"\nWROTE nodes parquet to: {out_nodes_parquet}")

nodes_lf = pl.scan_parquet(str(out_nodes_parquet))
stats = nodes_lf.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
    pl.col("q_m3s_obs").is_not_null().sum().alias("n_rows_with_q"),
    pl.col("water_temp_C").is_not_null().sum().alias("n_rows_with_temp"),
    pl.col("comid").filter(pl.col("q_m3s_obs").is_not_null()).n_unique().alias("n_comids_with_q"),
    pl.col("comid").filter(pl.col("water_temp_C").is_not_null()).n_unique().alias("n_comids_with_temp"),
    pl.col("tot_area_m2").is_not_null().mean().alias("pct_rows_with_tot_area_m2"),
    pl.col("Ra").is_not_null().mean().alias("pct_rows_with_Ra"),
]).collect()

print("\nNODES summary:")
print(stats)


## 9. Build EDGES
### edge_list.parquet (comid, fromnode, tonode) for model nodes
#### Keeps the format DDNN_Model.py expects; it converts fromnode / tonode into COMID-to-COMID edges
edge_df = statics.select(["comid", "fromnode", "tonode"]).drop_nulls().unique()
edge_df.write_parquet(str(out_edges_parquet), compression="zstd")
print(f"WROTE edges parquet to: {out_edges_parquet} ({edge_df.height:,} rows)")


## 10. Write modeling split files
### Analysis (before analysis_cut_date) and train / validation (on / after, stream order >= trainval min)
cut_date = cfg["analysis_cut_date"]
streamorder_min = int(cfg["trainval_streamorder_min"])

analysis_nodes = nodes_lf.filter(pl.col("date") < cut_date)
trainval_nodes = nodes_lf.filter((pl.col("date") >= cut_date) & (pl.col("streamorde") >= streamorder_min))

analysis_nodes.sink_parquet(str(out_analysis_parquet), compression="zstd")
trainval_nodes.sink_parquet(str(out_trainval_parquet), compression="zstd")
print(f"WROTE analysis split to: {out_analysis_parquet}")
print(f"WROTE train/validation split to: {out_trainval_parquet}")


## 11. Split diagnostics
### Train / test availability by stream order using the model split date
model_split_date = cfg["model_split_date"]
model_period_check = (
    pl.scan_parquet(str(out_trainval_parquet))
    .with_columns(
        pl.when(pl.col("date") < model_split_date).then(pl.lit("train")).otherwise(pl.lit("test")).alias("model_split")
    )
    .group_by(["model_split", "streamorde"])
    .agg([
        pl.len().alias("n_rows"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("comid").filter(pl.col("q_m3s_obs").is_not_null()).n_unique().alias("n_comids_with_q"),
        pl.col("comid").filter(pl.col("water_temp_C").is_not_null()).n_unique().alias("n_comids_with_temp"),
    ])
    .sort(["model_split", "streamorde"])
    .collect()
)
print(f"\nTRAIN/VAL availability by model split date {model_split_date}:")
print(model_period_check)


## 12. Hand off to the model
### Copy the two files DDNN_Model.py reads into data/daily_drivers/
for src in [out_edges_parquet, out_trainval_parquet]:
    dst = paths.model_input_dir / src.name
    tmp = dst.with_suffix(".part")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    print(f"COPIED {src.name} -> {paths.model_input_dir}")

write_stamp(paths, STEP, fingerprint, outputs)
print("\nDONE.")
