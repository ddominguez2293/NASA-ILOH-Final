# NASA-ILOH Final

Physics-guided deep learning for river discharge and stream temperature in the Illinois–Ohio River system.

This repository holds the data pipeline and model for the NASA Illinois–Ohio River Basin project: jointly modeling daily discharge (Q) and water temperature (T) on the routed river network, in order to test the T/Q hypothesis for drivers of algae.

## Pipeline

The scripts in `SRC/` form a single pipeline. Run them in the order below, from the repository root. Each step reads what the previous steps wrote to `data/` (which is not tracked by git).

| Step | Script | What it does | Main outputs |
|------|--------|--------------|--------------|
| 1 | `Boundaries.py` | Downloads the Ohio River Basin (HUC2 `05`) and Illinois River Basin (HUC4 `0712`) boundaries with `pynhd` and merges them. The Illinois basin is the area of interest used by the R steps. | `data/shapes/*.shp` |
| 2 | `Streamflow.Rmd` | Pulls daily USGS streamflow (parameter `00060`) for every site in the area of interest and converts it to m³/s. | `q_obs.parquet` |
| 3 | `Temp.Rmd` | Pulls daily USGS water temperature (parameter `00010`), aggregates to one value per site and date, and joins it to the streamflow table. | `q_t.parquet` |
| 4 | `Catchments.Rmd` | Builds the hydrologic geospatial layer: NHDPlus flowlines and catchments, upstream basins, and the Q/T site records matched to NHDPlus COMIDs. | `flowlines.gpkg`, `catchments.gpkg`, `site_upstream_basins.gpkg`, `q_t_comid.parquet` |
| 5 | `Meteo.Rmd` | Downloads daily GridMET drivers (`pr`, `tmmx`, `tmmn`, `srad`, `vs`, `vpd`, `sph`, `rmin`, `rmax`) and aggregates them to each catchment and to its upstream basin with polygon-weighted extraction (`exactextractr`). | `drivers_daily_joined.parquet` |
| 6 | `PET.Rmd` | Computes extraterrestrial radiation and Hargreaves potential evapotranspiration at catchment and basin scale (`PET`, `bsn_PET`), and adds day-of-year sine/cosine terms. | `drivers_daily.parquet` |
| 7 | `Q_Processing_python.py` | Joins the Q/T observations to the drivers by `comid` and `date`, attaches static catchment and flowline attributes (area, slope, length, stream order), builds the routing edge list, and splits the data by date. Uses `polars` so the large tables can be handled. | `nodes.parquet`, `edge_list.parquet`, `hydro_temp_data_analysis_pre2013.parquet`, `hydro_temp_data_trainval_2013plus.parquet` |
| 8 | `DDNN_Model.py` | **The model.** Trains and evaluates the multitask graph model (see below). | model weights, feature/target scalers, metrics, plots, Graph-SHAP explanations |

`Functions_Paths.R` is not a pipeline step. It centralizes the directory layout, loads the R package stack, and defines the helper functions shared by the R scripts (each one starts with `source("SRC/Functions_Paths.R")`).

## Model (`DDNN_Model.py`)

A multitask Graph LSTM–GAT model that predicts discharge (`q_m3s_obs`) and water temperature (`water_temp_C`) together for every reach in the network.

- **Inputs:** `hydro_temp_data_trainval_2013plus.parquet` (daily COMID × date records with meteorological drivers, PET, static attributes, and observed Q/T) and `edge_list.parquet` (NHDPlus `fromnode`/`tonode` routing).
- **Architecture:** a single-layer LSTM reads a 7-day lookback window for each reach; two stacked GAT layers then pass information along the routed river graph; a sigmoid-bounded head outputs Q and T.
- **Domain:** reaches with stream order ≥ 5.
- **Data split:** windows with a target date before 2025-01-01 are used for training, and windows on or after that date are held out for testing. Scalers are fit on the training period only.
- **Loss:** a weighted NSE-based term for Q (weight 0.7) plus a masked MSE term for T (weight 0.3), so missing temperature observations do not contribute to the loss. A graph-smoothness penalty and high-flow upweighting are also available in the configuration.
- **Outputs:** the trained weights and fitted scalers are written to `SAVE_DIR`; metrics (NSE, RMSE, MAE) and diagnostic plots to `OUTPUTS_DIR`. The script also runs a Graph-SHAP analysis at the feature, temporal, node × feature, and structural levels.

All hyperparameters (lookback, hidden sizes, epochs, learning rate, loss weights, and so on) are set in the `CONFIGURATION` block at the top of the script.

## Setup notes

- **Working directory:** the R scripts treat the current working directory as the project root, so run them from the repository root.
- **Paths:** `Q_Processing_python.py` and `DDNN_Model.py` define their input and output locations near the top of the file (`base_dir`, `NODES_PATH`, `EDGES_PATH`, `SAVE_DIR`, `OUTPUTS_DIR`). Update these to match your machine before running.
- **Python:** `geopandas`, `pynhd`, `polars`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `shap`, `torch`, and `torch_geometric`.
- **R:** the package stack loaded in `Functions_Paths.R`, plus `climateR`, `terra`, `exactextractr`, and `data.table` for the meteorological step.
- **Data:** everything under `data/` is generated by the pipeline and is excluded from version control.

## License

MIT. See `LICENSE`.
