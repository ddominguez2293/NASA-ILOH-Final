#!/usr/bin/env python3

# Q_Processing_python.py
# STEP 7 of the pipeline (runs AFTER PET.Rmd; the model script runs after this).
#
# Purpose : join Q / T observations to the PET-enhanced daily drivers, add static
#           catchment / flowline attributes, and write the model input tables.
# Inputs  : drivers_daily.parquet        (PET.Rmd)
#           q_t_comid.parquet            (Catchments.Rmd, Q/T already carry COMID)
#           flowlines.gpkg, catchments.gpkg (Catchments.Rmd)
# Outputs : hydro_nodes_allcomids_parquet/nodes.parquet
#           edge_list.parquet
#           hydro_temp_data_analysis_pre2013.parquet      (date <  2013-01-01)
#           hydro_temp_data_trainval_2013plus.parquet     (date >= 2013-01-01, stream order >= 5)
#
# Note    : uses q_t_comid.parquet directly; it does NOT rejoin q_t.parquet to sites_with_comid.csv.
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
## 1. Imports
from pathlib import Path
import polars as pl
import geopandas as gpd



## 2. Paths
### Input and output locations

base_dir = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH")  # NOTE: hard-coded absolute path; change to your project root

daily_drivers_dir = base_dir / "data/daily_drivers"

drivers_parquet = daily_drivers_dir / "drivers_daily.parquet"

### Q/T observations with COMID already attached
#### q_t_comid.parquet contains: site_no, Date, q_m3s, water_temp_C, comid, streamorde
qt_comid_parquet = daily_drivers_dir / "q_t_comid.parquet"

flowlines_gpkg = base_dir / "data/nhdplus/flowlines.gpkg"
catchments_gpkg = base_dir / "data/nhdplus/catchments.gpkg"

out_nodes_parquet_dir = daily_drivers_dir / "hydro_nodes_allcomids_parquet"
out_nodes_parquet = out_nodes_parquet_dir / "nodes.parquet"

out_edges_parquet = daily_drivers_dir / "edge_list.parquet"

out_analysis_parquet = daily_drivers_dir / "hydro_temp_data_analysis_pre2013.parquet"
out_trainval_parquet = daily_drivers_dir / "hydro_temp_data_trainval_2013plus.parquet"


## 3. Required file checks
### Stop early if any input file is missing

required_files = {
    "drivers_parquet": drivers_parquet,
    "qt_comid_parquet": qt_comid_parquet,
    "flowlines_gpkg": flowlines_gpkg,
    "catchments_gpkg": catchments_gpkg,
}

missing = [f"{label}: {path}" for label, path in required_files.items() if not path.exists()]

if missing:
    raise FileNotFoundError(
        "Missing required input files:\n" + "\n".join(missing)
    )

print("\nUsing input files:")
for label, path in required_files.items():
    print(f"  {label}: {path}")


## 4. Build STATICS table
### Catchment area (m^2) from catchments.gpkg

catch = gpd.read_file(
    catchments_gpkg,
    engine="pyogrio"
)[["comid", "areasqkm"]]

catch_pl = (
    pl.from_pandas(catch)
    .with_columns([
        pl.col("comid").cast(pl.Int64, strict=False),
        (pl.col("areasqkm").cast(pl.Float64, strict=False) * 1e6).alias("area_m2"),
    ])
    .select(["comid", "area_m2"])
    .drop_nulls(["comid"])
    .unique(subset=["comid"], keep="first")
)

valid_catchment_comids = catch_pl.select("comid").unique()

### Slope, length and stream order from flowlines.gpkg
flow = gpd.read_file(
    flowlines_gpkg,
    engine="pyogrio"
)[["comid", "fromnode", "tonode", "slope", "lengthkm", "streamorde"]]

flow_pl = (
    pl.from_pandas(flow)
    .with_columns([
        pl.col("comid").cast(pl.Int64, strict=False),
        pl.col("fromnode").cast(pl.Int64, strict=False),
        pl.col("tonode").cast(pl.Int64, strict=False),
        pl.col("slope").cast(pl.Float64, strict=False).alias("Slope"),
        pl.col("lengthkm").cast(pl.Float64, strict=False).alias("LengthKM"),
        pl.col("streamorde").cast(pl.Int32, strict=False),
    ])
    .drop_nulls(["comid"])
)

### Join into one statics table, one row per COMID
#### Keep statics only for COMIDs that have catchments; this avoids carrying the ~507 flowline-only COMIDs downstream
statics = (
    flow_pl
    .join(valid_catchment_comids, on="comid", how="inner")
    .select(["comid", "Slope", "LengthKM", "streamorde"])
    .unique(subset=["comid"], keep="first")
    .join(catch_pl, on="comid", how="left")
)

### Coverage check
print("\nSTATICS summary:")
print(
    statics.select([
        pl.len().alias("n_rows"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("area_m2").is_not_null().mean().alias("pct_with_area_m2"),
        pl.col("Slope").is_not_null().mean().alias("pct_with_Slope"),
        pl.col("LengthKM").is_not_null().mean().alias("pct_with_LengthKM"),
        pl.col("streamorde").is_not_null().mean().alias("pct_with_streamorde"),
    ])
)


## 5. Read PET-enhanced drivers
### Lazy scan of drivers_daily.parquet

drivers = (
    pl.scan_parquet(str(drivers_parquet))
    .with_columns([
        pl.col("comid").cast(pl.Int64, strict=False),
        pl.col("date").cast(pl.Date),
    ])
    .filter(pl.col("comid").is_not_null())
)

### Coverage check
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
### Read Q/T observations with COMID already attached
#### q_t_comid.parquet has: site_no, Date, q_m3s, water_temp_C, comid, streamorde

qtobs_raw = (
    pl.scan_parquet(str(qt_comid_parquet))
    .with_columns([
        pl.col("site_no").cast(pl.Utf8),
        pl.col("Date").cast(pl.Date),
        pl.col("comid").cast(pl.Int64, strict=False),
        pl.col("streamorde").cast(pl.Int32, strict=False),
        pl.col("q_m3s").cast(pl.Float64, strict=False),
        pl.col("water_temp_C").cast(pl.Float64, strict=False),
    ])
    .filter(pl.col("comid").is_not_null())
)

### Coverage check, overall and by stream order
qtobs_raw_check = qtobs_raw.select([
    pl.len().alias("n_rows"),
    pl.col("site_no").n_unique().alias("n_sites"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("Date").min().alias("min_date"),
    pl.col("Date").max().alias("max_date"),
    pl.col("q_m3s").is_not_null().sum().alias("n_q_obs"),
    pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
]).collect()

print("\nQ/T COMID input summary:")
print(qtobs_raw_check)

print("\nQ/T COMID input by stream order:")
print(
    qtobs_raw
    .group_by("streamorde")
    .agg([
        pl.len().alias("n_rows"),
        pl.col("site_no").n_unique().alias("n_sites"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("q_m3s").is_not_null().sum().alias("n_q_obs"),
        pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
        pl.col("Date").min().alias("min_date"),
        pl.col("Date").max().alias("max_date"),
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
        pl.col("q_m3s_obs").mean().alias("q_m3s_obs"),
        pl.col("water_temp_C").mean().alias("water_temp_C"),
    ])
)

### Coverage check
qtobs_check = qtobs.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
    pl.col("q_m3s_obs").is_not_null().sum().alias("n_q_obs"),
    pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
]).collect()

print("\nQ/T aggregated to COMID-date summary:")
print(qtobs_check)


## 7. Build NODES
### Left-join Q/T obs and statics onto the drivers (one row per comid-date)

nodes = (
    drivers
    .join(qtobs, on=["comid", "date"], how="left")
    .join(statics.lazy(), on="comid", how="left")
)


## 8. Summarize before writing
### Row counts, coverage and value ranges of the nodes table

stats = nodes.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),

    pl.col("q_m3s_obs").is_not_null().sum().alias("n_rows_with_q"),
    pl.col("water_temp_C").is_not_null().sum().alias("n_rows_with_temp"),

    pl.col("q_m3s_obs").is_not_null().mean().alias("pct_rows_with_q"),
    pl.col("water_temp_C").is_not_null().mean().alias("pct_rows_with_temp"),

    pl.col("comid")
    .filter(pl.col("q_m3s_obs").is_not_null())
    .n_unique()
    .alias("n_comids_with_any_q"),

    pl.col("comid")
    .filter(pl.col("water_temp_C").is_not_null())
    .n_unique()
    .alias("n_comids_with_any_temp"),

    pl.col("area_m2").is_not_null().mean().alias("pct_rows_with_area_m2"),
    pl.col("Slope").is_not_null().mean().alias("pct_rows_with_Slope"),
    pl.col("LengthKM").is_not_null().mean().alias("pct_rows_with_LengthKM"),
    pl.col("streamorde").is_not_null().mean().alias("pct_rows_with_streamorde"),

    pl.col("PET").is_not_null().mean().alias("pct_rows_with_PET"),
    pl.col("bsn_PET").is_not_null().mean().alias("pct_rows_with_bsn_PET"),
    pl.col("Ra").is_not_null().mean().alias("pct_rows_with_Ra"),
    pl.col("DOY_sin").is_not_null().mean().alias("pct_rows_with_DOY_sin"),
    pl.col("DOY_cos").is_not_null().mean().alias("pct_rows_with_DOY_cos"),

    pl.col("q_m3s_obs").min().alias("min_q_m3s_obs"),
    pl.col("q_m3s_obs").max().alias("max_q_m3s_obs"),
    pl.col("water_temp_C").min().alias("min_water_temp_C"),
    pl.col("water_temp_C").max().alias("max_water_temp_C"),
]).collect()

print("\nNODES summary before writing:")
print(stats)


## 9. Q/T availability by stream order
### Check before writing

qt_by_stream = (
    nodes
    .group_by("streamorde")
    .agg([
        pl.len().alias("n_rows"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("q_m3s_obs").is_not_null().sum().alias("n_q_obs"),
        pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
        pl.col("comid")
        .filter(pl.col("q_m3s_obs").is_not_null())
        .n_unique()
        .alias("n_comids_with_q"),
        pl.col("comid")
        .filter(pl.col("water_temp_C").is_not_null())
        .n_unique()
        .alias("n_comids_with_temp"),
    ])
    .sort("streamorde")
    .collect()
)

print("\nNODES Q/T availability by stream order before writing:")
print(qt_by_stream)


## 10. Write nodes parquet
### Stream the lazy frame to zstd parquet

out_nodes_parquet_dir.mkdir(parents=True, exist_ok=True)

nodes.sink_parquet(
    str(out_nodes_parquet),
    compression="zstd"
)

print(f"\nWROTE nodes parquet to: {out_nodes_parquet}")


## 11. Build EDGES
### Write edge_list.parquet (comid, fromnode, tonode)
#### Keeps the original format expected by DDNN_Model.py
#### The model script internally converts fromnode / tonode into COMID-to-COMID edges

edge_df = (
    flow_pl
    .join(valid_catchment_comids, on="comid", how="inner")
    .select([
        pl.col("comid").cast(pl.Int64),
        pl.col("fromnode").cast(pl.Int64),
        pl.col("tonode").cast(pl.Int64),
    ])
    .drop_nulls(["fromnode", "tonode"])
    .unique()
)

edge_df.write_parquet(str(out_edges_parquet), compression="zstd")

print(f"WROTE edges parquet to: {out_edges_parquet}")
print(
    "Edge rows:",
    edge_df.height,
    "unique COMIDs:",
    edge_df.select(pl.col("comid").n_unique()).item()
)


## 12. Re-read nodes and verify
### Confirm Q/T joined properly in the written file

nodes_check_lf = pl.scan_parquet(str(out_nodes_parquet))

qt_check = nodes_check_lf.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),

    pl.col("q_m3s_obs").is_not_null().sum().alias("n_rows_with_q"),
    pl.col("water_temp_C").is_not_null().sum().alias("n_rows_with_temp"),

    pl.col("q_m3s_obs").is_not_null().mean().alias("pct_rows_with_q"),
    pl.col("water_temp_C").is_not_null().mean().alias("pct_rows_with_temp"),

    pl.col("comid")
    .filter(pl.col("q_m3s_obs").is_not_null())
    .n_unique()
    .alias("n_comids_with_q"),

    pl.col("comid")
    .filter(pl.col("water_temp_C").is_not_null())
    .n_unique()
    .alias("n_comids_with_temp"),

    pl.col("q_m3s_obs").min().alias("min_q_m3s_obs"),
    pl.col("q_m3s_obs").max().alias("max_q_m3s_obs"),
    pl.col("water_temp_C").min().alias("min_water_temp_C"),
    pl.col("water_temp_C").max().alias("max_water_temp_C"),
]).collect()

print("\nRE-READ NODES Q/T CHECK:")
print(qt_check)


## 13. Write modeling split files
### Analysis (pre-2013) and train / validation (2013+) tables
#### These replace the old R Watertemp split script; Q/T are already joined in nodes
#### The train / validation table is restricted to stream order >= 5

cut_date = pl.date(2013, 1, 1)  # split date: before = analysis set, on / after = train / validation set
streamorder_min = 5  # train / validation set keeps stream order >= 5

analysis_nodes = (
    nodes_check_lf
    .filter(pl.col("date") < cut_date)
)

trainval_nodes = (
    nodes_check_lf
    .filter(pl.col("date") >= cut_date)
    .filter(pl.col("streamorde") >= streamorder_min)
)

analysis_nodes.sink_parquet(
    str(out_analysis_parquet),
    compression="zstd"
)

trainval_nodes.sink_parquet(
    str(out_trainval_parquet),
    compression="zstd"
)

print(f"WROTE analysis split to: {out_analysis_parquet}")
print(f"WROTE train/validation split to: {out_trainval_parquet}")


## 14. Split diagnostics
### Coverage of the train / validation table

split_stats = trainval_nodes.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),

    pl.col("streamorde").min().alias("min_streamorde"),
    pl.col("streamorde").max().alias("max_streamorde"),

    pl.col("PET").is_not_null().mean().alias("pct_PET"),
    pl.col("bsn_PET").is_not_null().mean().alias("pct_bsn_PET"),

    pl.col("q_m3s_obs").is_not_null().sum().alias("n_q_obs"),
    pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),

    pl.col("comid")
    .filter(pl.col("q_m3s_obs").is_not_null())
    .n_unique()
    .alias("n_comids_with_q"),

    pl.col("comid")
    .filter(pl.col("water_temp_C").is_not_null())
    .n_unique()
    .alias("n_comids_with_temp"),

    pl.col("q_m3s_obs").min().alias("min_q_m3s_obs"),
    pl.col("q_m3s_obs").max().alias("max_q_m3s_obs"),
    pl.col("water_temp_C").min().alias("min_water_temp_C"),
    pl.col("water_temp_C").max().alias("max_water_temp_C"),
]).collect()

print("\nTRAIN/VAL split summary:")
print(split_stats)

### Q/T availability by stream order
split_by_stream = (
    trainval_nodes
    .group_by("streamorde")
    .agg([
        pl.len().alias("n_rows"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("q_m3s_obs").is_not_null().sum().alias("n_q_obs"),
        pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
        pl.col("comid")
        .filter(pl.col("q_m3s_obs").is_not_null())
        .n_unique()
        .alias("n_comids_with_q"),
        pl.col("comid")
        .filter(pl.col("water_temp_C").is_not_null())
        .n_unique()
        .alias("n_comids_with_temp"),
    ])
    .sort("streamorde")
    .collect()
)

print("\nTRAIN/VAL Q/T availability by stream order:")
print(split_by_stream)


## 15. Diagnostics by model test period
### Train / test availability using the model split date

model_split_date = pl.date(2025, 1, 1)  # same split date used in DDNN_Model.py

model_period_check = (
    trainval_nodes
    .with_columns([
        pl.when(pl.col("date") < model_split_date)
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("model_split")
    ])
    .group_by(["model_split", "streamorde"])
    .agg([
        pl.len().alias("n_rows"),
        pl.col("comid").n_unique().alias("n_comids"),
        pl.col("q_m3s_obs").is_not_null().sum().alias("n_q_obs"),
        pl.col("water_temp_C").is_not_null().sum().alias("n_temp_obs"),
        pl.col("comid")
        .filter(pl.col("q_m3s_obs").is_not_null())
        .n_unique()
        .alias("n_comids_with_q"),
        pl.col("comid")
        .filter(pl.col("water_temp_C").is_not_null())
        .n_unique()
        .alias("n_comids_with_temp"),
    ])
    .sort(["model_split", "streamorde"])
    .collect()
)

print("\nTRAIN/VAL availability by model split date 2025-01-01:")
print(model_period_check)


print("\nDONE.")