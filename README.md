# NASA-ILOH Final

Physics-guided deep learning for river discharge and stream temperature in the Illinois–Ohio River system.

This repository holds the data pipeline and model for the NASA Illinois–Ohio River Basin project: jointly modeling daily discharge (Q) and water temperature (T) on the routed river network, in order to test the T/Q hypothesis for drivers of algae.

> **Notice:** This repository and its contents are the proprietary intellectual property of Daniel Dominguez. This is **not** an open-source project — access is granted to specific collaborators by invitation only, and no rights to use, copy, modify, or distribute any part of it are granted beyond what's stated in [`LICENSE`](LICENSE). Please read the license terms before using or building on anything here. See also the [Abbreviations](#abbreviations) glossary below if you're new to the project.

## Pipeline

The whole pipeline is Python. Everything before the model is driven by [`config.yaml`](config.yaml): list one or more HUCs (any mix of HUC2 / HUC4 / HUC8), set the time window, and run

```
python run_pipeline.py                         # steps 1-6, skipping any step that is already up to date
python run_pipeline.py --config other.yaml     # a different HUC set / time window
python run_pipeline.py --from-step catchments  # restart part-way through
python run_pipeline.py --only meteo --force    # rebuild one step
```

Each AOI gets its own folder, `data/<aoi_name>/` (not tracked by git). Every step records the config it was built with, so changing the HUCs or dates rebuilds exactly the steps that depend on them instead of reusing stale files. The last step copies the two model inputs to `data/daily_drivers/`, where `DDNN_Model.py` reads them.

| Step | Script | What it does | Main outputs (under `data/<aoi_name>/`) |
|------|--------|--------------|--------------|
| 1 | `Boundaries.py` | Downloads the WBD boundary for every configured HUC with `pynhd`, dissolves them into one AOI, and resolves the NHDPlus HR HU4s to download (HUC2 → child HU4s, HUC8 → parent HU4). Drops HUCs nested in another configured HUC and warns if the AOI is not contiguous. | `shapes/aoi.gpkg`, `shapes/hu4s.json` |
| 2 | `Streamflow.py` | Pulls daily-mean USGS streamflow (`00060`) for every site inside the AOI (Water Data API via `dataretrieval`) and converts it to m³/s. | `observations/q_obs.parquet` |
| 3 | `Temp.py` | Pulls daily-mean USGS water temperature (`00010`) inside the AOI, averages sensors to one value per site and date, and full-joins it to streamflow. | `observations/q_t.parquet` |
| 4 | `Catchments.py` | Reads NHDPlus HR for each HU4 (zips cached in `data/_nhdplushr/`), builds the routed network for **all** stream orders, snaps every Q/T site to the nearest flowline within 250 m (EPSG:5070), and writes the stream order ≥ `streamorder_min` model domain. Checks accumulated area against NHDPlus `TotDASqKm` and warns when a reach is fed from outside the AOI (an upstream HUC is missing from the config). | `nhdplus/network.parquet`, `nhdplus/catchments_all.gpkg`, `nhdplus/flowlines.gpkg`, `nhdplus/catchments.gpkg`, `observations/q_t_comid.parquet` |
| 5 | `Meteo.py` | Downloads ERA5-Land for the AOI (see below) and builds the daily drivers `pr`, `srad`, `vs`, `vpd`, `sph`, `rmin`, `rmax`, `tmax_c`, `tmin_c`, `tmean_c` and `PET` for every model catchment, plus `bsn_*` versions (area-weighted mean over the whole upstream basin) and `Ra`, `DOY_sin`, `DOY_cos`. | `daily_drivers/drivers_daily/<year>.parquet` |
| 6 | `Q_Processing_python.py` | Joins the Q/T observations to the drivers by `comid` and `date`, attaches static attributes (local area `area_m2`, total upstream drainage area `tot_area_m2`, slope, length, stream order), builds the edge list, and splits by date (before `analysis_cut_date` = analysis set; after = train/validation set with stream order ≥ `trainval_streamorder_min`). | `daily_drivers/edge_list.parquet`, `daily_drivers/hydro_temp_data_*.parquet`, copied to `data/daily_drivers/` |
| 7 | `DDNN_Model.py` | **The model.** Trains and evaluates the multitask graph model (see below). Run from the terminal with CLI flags so repeat experiments don't require editing the file. | per-run folder under `outputs/`: `results.txt` (test variables + final metrics), `test_metrics_by_streamorder.csv`, diagnostic plots; model weights/scalers/features are only written to `SAVE_DIR` if `--save-model` is passed |
| 8 | `GraphSHAP.py` | Explains a trained model with Graph-SHAP (feature, temporal, node×feature, and structural levels). Reloads the weights/scalers/features saved by `DDNN_Model.py --save-model` rather than training anything itself — run it once you've settled on a config and want to interpret that model. | Graph-SHAP figures and `.npy` arrays in `OUTPUTS_DIR` / `SAVE_DIR` |

`pipeline_utils.py` is not a pipeline step. It loads and validates `config.yaml`, defines the `data/<aoi_name>/` layout, the up-to-date stamps, the USGS download helper, and the network routing shared by `Catchments.py` and `Meteo.py`.

### Meteorology (ERA5-Land)

- **Source:** hourly ERA5-Land (0.1°, about 9 km), aggregated to US Central (UTC−6) local days.
  - Temperature, dewpoint, pressure and wind come from the ERA5-Land **point time-series** product: one request per grid cell that touches a catchment returns the whole period with all variables together, and is reduced to daily values as it arrives (`meteo/era5_raw/<key>/points/`). The Illinois basin needs 926 cells, about 2–3 hours.
  - Precipitation, solar radiation and potential evaporation come from gridded hourly ERA5-Land, one request per year at 00 and 06 UTC only; ERA5-Land accumulations reset at 00 UTC, so those two fields give the exact local-day total.
- **Why not the gridded or daily products for temperature / wind:** tested on 2026-10-22/23. The CDS runs only about one request per user at a time and refuses gridded hourly requests longer than a month, so 1984–2026 would take about a week; the daily-statistics product refused a year of daily means, queued each accepted request 2–3.5 h, and drops accumulated variables altogether. Both remain available (`era5.instant_source: hourly` or `daily`), and the point and gridded hourly routes give identical daily values (checked to float precision on 2023).
- **Derived drivers:** `tmax_c` / `tmin_c` are the hourly max / min (`tmean_c` = their average, as before); `rmin` / `rmax` are the hourly min / max relative humidity; `vpd`, `sph` and `vs` are daily means of the hourly values; `PET` is ERA5-Land potential evaporation (mm/day, ≥ 0); `Ra` uses the FAO-56 formula from the former `PET.Rmd`.
- **Catchment values** use each catchment's exact grid-cell coverage (`exactextract`), restricted to land cells. **Basin values (`bsn_*`)** are the area-weighted mean over every upstream catchment of every stream order. Two routings are computed and each reach keeps the one whose accumulated area is closer to NHDPlus `TotDASqKm` (`basin_routing` in `network.parquet`): *main* follows the NHDPlus main path (no area is lost where an HR braid dead-ends, which drops ~11,000 km² from the Illinois mainstem under the other rule), and *split* follows the node topology and divides flow equally at divergences (so side channels inherit the upstream basin). Neither routing ever counts a catchment twice.
- **CDS access:** needs a key in `~/.cdsapirc` and the ERA5-Land licence accepted on the CDS website. The CDS address comes from `era5.cds_url` in `config.yaml`. Downloads are cached in `data/<aoi_name>/meteo/era5_raw/`, so an interrupted run resumes where it stopped.

## Comment style

Every script (and `config.yaml`) is commented the same way, so the code reads like a markdown outline:

| Marker | Meaning |
|--------|---------|
| `## N. Section` | A major section of the script (numbered in order) |
| `### Step` | What the next block of code does |
| `#### Note` | Why, assumptions, or caveats for that step |
| `# inline` | A short single-line bit next to the code |

Each script opens with a header comment giving its step number, purpose, inputs and outputs. Only comments follow this convention; the code is not affected.

## Model (`DDNN_Model.py`)

A multitask Graph LSTM–GAT model that predicts discharge (`q_m3s_obs`) and water temperature (`water_temp_C`) together for every reach in the network.

- **Inputs:** `hydro_temp_data_trainval_2013plus.parquet` (daily COMID × date records with meteorological drivers, PET, static attributes, and observed Q/T) and `edge_list.parquet` (NHDPlus `fromnode`/`tonode` routing).
- **Architecture:** a single-layer LSTM reads a lookback window for each reach; two stacked GAT layers then pass information along the routed river graph; a sigmoid-bounded head outputs Q and T.
- **Domain:** reaches with `streamorde` inside an inclusive `[min, max]` band (a range filter, not just a floor — lets specific order bands be isolated for comparison).
- **Data split:** windows with a target date before 2025-01-01 are used for training, and windows on or after that date are held out for testing. Scalers are fit on the training period only.
- **Loss:** a weighted NSE-based term for Q (weight 0.7) plus a masked MSE term for T (weight 0.3), so missing temperature observations do not contribute to the loss. A graph-smoothness penalty and high-flow upweighting are also available in the configuration.
- **Log-scaling (experimental):** an off-by-default toggle that log1p-transforms discharge before scaling (and inverts it back at evaluation time) — a tail-compression option to test as its own experiment.

### Running experiments from the terminal

The variables most worth sweeping are exposed as CLI flags, so repeat experiments don't require editing the file:

```
python SRC/DDNN_Model.py \
  --streamorder-min 8 --streamorder-max 9 \
  --lookback 30 \
  --epochs 200 \
  [--log-scale-q] [--save-model] [--target-comid COMID] [--run-name NAME]
```

- `--streamorder-min` / `--streamorder-max` — the stream-order band (default `8`/`9`).
- `--lookback` — antecedent days per window (default `30`).
- `--log-scale-q` — enable the log1p experiment above (default off).
- `--save-model` — persist trained weights/scalers/`features.pkl` to `SAVE_DIR` (default off, since repeat experiment runs would otherwise just keep overwriting the same files). Pass this once a config is settled on and you want `GraphSHAP.py` to be able to explain that model.
- `--target-comid` — COMID for the single-reach diagnostic plots; if omitted (or not present in the current stream-order band) the script falls back to an arbitrary in-domain COMID and records a warning.
- `--run-name` — names the output folder; default is built from the test variables plus a timestamp.

Every run writes to its own `outputs/<run-name>/` folder: `results.txt` (every test variable, written at the start of the run, with final NSE/RMSE/MAE metrics appended once training finishes), `test_metrics_by_streamorder.csv`, and the diagnostic plots. Nothing is overwritten between runs unless you reuse `--run-name`.

All other hyperparameters (hidden sizes, learning rate, loss weights, graph coarsening, and so on) are set in section `2. Configuration` at the top of the script.

## Graph-SHAP (`GraphSHAP.py`)

Explains a trained model with a three-level Monte Carlo Shapley framework (feature, temporal, node×feature) plus structural Graph-SHAP over incoming edges — kept in its own script so training runs aren't slowed down by SHAP's permutation sampling.

- **Inputs:** the same node/edge parquet files as `DDNN_Model.py`, plus the trained weights, `scaler_X.pkl`, and `features.pkl` written to `SAVE_DIR` by a `DDNN_Model.py --save-model` run.
- **Important:** the stream-order band, `LOOKBACK`, graph-coarsening settings, and model hyperparameters at the top of `GraphSHAP.py` are duplicated from `DDNN_Model.py` rather than imported, and must match whichever training run produced the artifacts in `SAVE_DIR` — update them by hand before running.
- **Outputs:** ranked Shapley bar charts, a node×feature heatmap, and `.npy` arrays, written to `OUTPUTS_DIR` / `SAVE_DIR`.

## Setup notes

- **Paths:** every script resolves its paths relative to the repository root, so they can be run from anywhere; nothing needs hand-editing on a different machine.
- **Python:** `pip install -r requirements.txt` (pinned versions in `requirements-lock.txt`) installs the model stack (`numpy`, `pandas`, `torch`, `torch-geometric`, `shap`, ...) and the pipeline stack (`geopandas`, `pynhd`, `polars`, `exactextract`, `dataretrieval`, `cdsapi`, `xarray`, ...). R is no longer needed.
- **Disk:** NHDPlus HR zips are about 0.25–0.4 GB per HU4; the driver tables for the full Illinois basin and 1984–2026 run to tens of GB.
- **Data:** everything under `data/` is generated by the pipeline and is excluded from version control.

## Abbreviations

Glossary for anyone new to the project — hydrology, modeling, and pipeline terms used throughout this README and the scripts.

| Term | Meaning |
|------|---------|
| COMID | Common Identifier — the unique ID NHDPlus assigns to each flowline/catchment reach |
| NHD / NHDPlus | National Hydrography Dataset (Plus) — USGS's digital river network dataset |
| HUC2 / HUC4 | Hydrologic Unit Code — nested watershed boundary codes (HUC2 = region, HUC4 = sub-region) |
| USGS / NWIS | U.S. Geological Survey / National Water Information System — source of streamflow and water temperature observations |
| Q | Discharge (streamflow), in m³/s |
| T | Water temperature, in °C |
| ERA5-Land | ECMWF's land reanalysis (0.1°, hourly), the source of the meteorological drivers (precipitation, temperature, radiation, wind, humidity, PET) |
| CDS | Copernicus Climate Data Store — where ERA5-Land is downloaded from |
| AOI | Area of interest — the dissolved union of the HUCs in `config.yaml` |
| PET | Potential evapotranspiration (ERA5-Land potential evaporation) |
| Ra | Extraterrestrial radiation (top-of-atmosphere solar radiation, FAO-56) |
| DOY | Day of year (used for the sine/cosine seasonal features) |
| LSTM | Long Short-Term Memory — the recurrent layer that encodes each reach's time-series input |
| GAT / GNN | Graph Attention (Network) layer / Graph Neural Network — propagates information along the routed river graph |
| MTL | Multitask learning — the model predicts Q and T jointly (see the `GraphLSTM_MTL` class) |
| NSE | Nash–Sutcliffe Efficiency — the primary goodness-of-fit metric for discharge |
| RMSE / MAE / MSE | Root Mean Squared Error / Mean Absolute Error / Mean Squared Error |
| SHAP / MC Shapley | SHapley Additive exPlanations — the feature-attribution framework `GraphSHAP.py` adapts with Monte Carlo (MC) sampling for graph models |
| QA / QC | Quality assurance / quality control checks applied to the data |
| CLI | Command-line interface — see the `DDNN_Model.py` flags above |

## License

Proprietary — All Rights Reserved. See `LICENSE`.
