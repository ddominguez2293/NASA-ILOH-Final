# NASA-ILOH Final

Physics-guided deep learning for river discharge and stream temperature in the Illinois–Ohio River system.

This repository holds the data pipeline and model for the NASA Illinois–Ohio River Basin project: jointly modeling daily discharge (Q) and water temperature (T) on the routed river network, in order to test the T/Q hypothesis for drivers of algae.

> **Notice:** This repository and its contents are the proprietary intellectual property of Daniel Dominguez. This is **not** an open-source project — access is granted to specific collaborators by invitation only, and no rights to use, copy, modify, or distribute any part of it are granted beyond what's stated in [`LICENSE`](LICENSE). Please read the license terms before using or building on anything here. See also the [Abbreviations](#abbreviations) glossary below if you're new to the project.

## Pipeline

The scripts in `SRC/` form a single pipeline. Run them in the order below, from the repository root. Each step reads what the previous steps wrote to `data/` (which is not tracked by git).

| Step | Script | What it does | Main outputs |
|------|--------|--------------|--------------|
| 1 | `Boundaries.py` | Downloads the Ohio River Basin (HUC2 `05`) and Illinois River Basin (HUC4 `0712` + `0713`, dissolved into one polygon) boundaries with `pynhd`. The Illinois basin is the area of interest used by every R step. | `data/shapes/*.shp` |
| 2 | `Streamflow.Rmd` | Pulls daily USGS streamflow (parameter `00060`) for every site in the area of interest and converts it to m³/s. | `q_obs.parquet` |
| 3 | `Temp.Rmd` | Pulls daily USGS water temperature (parameter `00010`), aggregates to one value per site and date, and joins it to the streamflow table. | `q_t.parquet` |
| 4 | `Catchments.Rmd` | Downloads NHDPlus HR (HU4 `0712` + `0713`), snaps every Q/T site to a flowline COMID, builds one upstream basin per catchment locally, and keeps stream order ≥ 4. | `flowlines.gpkg`, `catchments.gpkg`, `site_upstream_basins.gpkg`, `sites_with_comid.csv`, `q_t_comid.parquet` |
| 5 | `Meteo.Rmd` | Downloads daily GridMET drivers (`pr`, `tmmx`, `tmmn`, `srad`, `vs`, `vpd`, `sph`, `rmin`, `rmax`) and extracts them to each catchment with `exactextractr`. Basin-scale drivers are the area-weighted mean of the upstream catchment values. Temperatures are converted from K to °C. | `drivers_daily_joined.parquet` |
| 6 | `PET.Rmd` | Computes extraterrestrial radiation and Hargreaves potential evapotranspiration at catchment and basin scale (`PET`, `bsn_PET`), and adds day-of-year sine/cosine terms. | `drivers_daily.parquet` |
| 7 | `Q_Processing_python.py` | Joins the Q/T observations to the drivers by `comid` and `date`, attaches static catchment and flowline attributes (area, slope, length, stream order), builds the routing edge list, and splits the data by date (pre-2013 analysis set; 2013+ train/validation set restricted to stream order ≥ 5). Uses `polars` so the large tables can be handled. | `nodes.parquet`, `edge_list.parquet`, `hydro_temp_data_analysis_pre2013.parquet`, `hydro_temp_data_trainval_2013plus.parquet` |
| 8 | `DDNN_Model.py` | **The model.** Trains and evaluates the multitask graph model (see below). Run from the terminal with CLI flags so repeat experiments don't require editing the file. | per-run folder under `outputs/`: `results.txt` (test variables + final metrics), `test_metrics_by_streamorder.csv`, diagnostic plots; model weights/scalers/features are only written to `SAVE_DIR` if `--save-model` is passed |
| 9 | `GraphSHAP.py` | Explains a trained model with Graph-SHAP (feature, temporal, node×feature, and structural levels). Reloads the weights/scalers/features saved by `DDNN_Model.py --save-model` rather than training anything itself — run it once you've settled on a config and want to interpret that model. | Graph-SHAP figures and `.npy` arrays in `OUTPUTS_DIR` / `SAVE_DIR` |

`Functions_Paths.R` is not a pipeline step. It centralizes the directory layout, loads the R package stack, and defines the helper functions shared by the R scripts (each one starts with `source("SRC/Functions_Paths.R")`).

## Comment style

Every script in `SRC/` is commented the same way, so the code reads like a markdown outline:

| Marker | Meaning |
|--------|---------|
| `## N. Section` | A major section of the script (numbered in order) |
| `### Step` | What the next block of code does |
| `#### Note` | Why, assumptions, or caveats for that step |
| `# inline` | A short single-line bit next to the code |

Inside the R Markdown files these markers sit inside the code chunks, and each file opens with a short prose header (step number, purpose, inputs, end-state). `Python` and `R` scripts open with a header comment listing inputs and outputs. Only comments follow this convention; the code is not affected.

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

- **Working directory:** the R scripts treat the current working directory as the project root, so run them from the repository root.
- **Paths:** `Q_Processing_python.py`, `DDNN_Model.py`, and `GraphSHAP.py` all resolve their input/output paths (`NODES_PATH`, `EDGES_PATH`, `SAVE_DIR`, `OUTPUTS_DIR`) relative to the repository root automatically — no hand-editing needed on a different machine.
- **Python:** `pip install -r requirements.txt` (pinned versions in `requirements-lock.txt`) installs `numpy`, `pandas`, `matplotlib`, `joblib`, `scikit-learn`, `torch`, `torch-geometric`, `shap`, `pyarrow`, and the rest of the pipeline's dependencies, plus `geopandas`/`pynhd`/`polars` for the earlier steps.
- **R:** the package stack loaded in `Functions_Paths.R`, plus `climateR`, `terra`, `exactextractr`, and `data.table` for the meteorological step.
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
| GridMET | The gridded daily meteorological dataset used for weather drivers (precipitation, radiation, wind, humidity, etc.) |
| PET | Potential evapotranspiration |
| Ra | Extraterrestrial radiation (used in the Hargreaves PET calculation) |
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
