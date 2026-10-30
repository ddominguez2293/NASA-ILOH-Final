#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DDNN_Model.py  (the model file)
# STEP 8 of the pipeline (runs AFTER Q_Processing_python.py; GraphSHAP.py runs after this).
#
# Purpose : train and evaluate a multitask Graph LSTM-GAT that predicts daily discharge (Q)
#           and water temperature (T) on the routed NHDPlus network. Run from the terminal
#           with CLI flags (see Section 2) so repeat experiments don't require editing this
#           file. Graph-SHAP interpretation lives in GraphSHAP.py, run separately afterward.
# Inputs  : hydro_temp_data_trainval_2013plus.parquet   (NODES_PATH)
#           edge_list.parquet                           (EDGES_PATH)
# Outputs : outputs/<RUN_NAME>/results.txt (test variables + final metrics),
#           test_metrics_by_streamorder.csv, and diagnostic figures — one
#           folder per run. Trained weights/scalers/features.pkl are only
#           written to SAVE_DIR when --save-model is passed.
#
# Note    : the study area is the Illinois River Basin, HUC4 0712 + 0713 (the module docstring
#           below mentions 0712 only).
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
"""
DDNN_Model — Graph LSTM-GAT Multitask Model for Discharge & Water Temperature
Illinois River Basin (HUC 0712 ) | Daily | 2013–present

Architecture : Single-layer LSTM → 2× GAT layers → sigmoid-bounded dual output head
Targets      : Discharge Q (m³/s) and water temperature T (°C)
Loss         : Weighted NSE-like (Q, 0.7) + MSE (Temp, 0.3) with masked supervision
Explanation  : Full Graph-SHAP — feature, temporal, node×feature, and structural levels
"""

## 1. Imports
### Standard library
import os
import argparse
import random
import time
from pathlib import Path
from datetime import datetime

### Scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

### PyTorch + graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv

### Sklearn
from sklearn.preprocessing import MinMaxScaler


## 2. Configuration

### Command-line experiment overrides
#### Lets repeat experiments be launched from the terminal without editing
#### this file each run, e.g.:
####   python SRC/DDNN_Model.py --streamorder-min 8 --streamorder-max 9 --lookback 30
#### Defaults below match the current first verification run.
parser = argparse.ArgumentParser(description="Train the Graph LSTM-GAT multitask Q/T model.")
parser.add_argument("--streamorder-min", type=int, default=8)
parser.add_argument("--streamorder-max", type=int, default=9)
parser.add_argument("--lookback", type=int, default=30, help="days of antecedent input per window")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--log-scale-q", action="store_true", default=False,
                     help="log1p-transform discharge before scaling (off by default)")
parser.add_argument("--save-model", action="store_true", default=False,
                     help="persist trained weights/scalers/features to SAVE_DIR "
                          "(off by default while running experiments)")
parser.add_argument("--run-name", type=str, default=None,
                     help="outputs/<run-name> folder name; default is built from "
                          "the test variables + a timestamp")
parser.add_argument("--target-comid", type=int, default=None,
                     help="COMID for single-reach diagnostic plots; defaults to 14787569, "
                          "falling back to an in-domain COMID if that one isn't in this "
                          "run's stream-order band")

### Multitask weighting, seasonal ablation, high-flow weighting (Experiments B/C/D)
parser.add_argument("--q-weight", type=float, default=0.70,
                     help="task weight for discharge (Q) in the multitask loss")
parser.add_argument("--t-weight", type=float, default=0.30,
                     help="task weight for water temperature (T) in the multitask loss")
parser.add_argument("--include-doy", action=argparse.BooleanOptionalAction, default=True,
                     help="include DOY sine/cosine seasonal features; pass "
                          "--no-include-doy to ablate them (Experiment C)")
parser.add_argument("--doy-component", choices=["both", "sin", "cos"], default="both",
                     help="when --include-doy is set, which DOY feature(s) to keep: "
                          "'both' (default), or only 'sin' / only 'cos' to test whether "
                          "a single seasonal component carries the same signal (ex- series)")
parser.add_argument("--q-high-max-mult", type=float, default=5.0,
                     help="max multiplier for high-flow Q residual upweighting; "
                          "set to 1.0 for uniform (unweighted) loss (Experiment D)")
parser.add_argument("--q-high-thresh", type=float, default=100.0,
                     help="discharge threshold (m3/s) above which high-flow "
                          "upweighting ramps up")

### Validation/test independence and reproducibility
parser.add_argument("--val-days", type=int, default=180,
                     help="days immediately before --split-date held out as an "
                          "independent validation window used for early stopping "
                          "and model selection; the test window (on/after the split "
                          "date) is held out from training/validation and is only "
                          "touched once, for final reporting")
parser.add_argument("--seed", type=int, default=0,
                     help="random seed for numpy/torch reproducibility")

### Common stream-order subset for cross-domain comparability
parser.add_argument("--common-so-min", type=int, default=8,
                     help="lower bound of the stream-order band reported as the "
                          "'common subset' shared across all stream-order experiments")
parser.add_argument("--common-so-max", type=int, default=9,
                     help="upper bound of the stream-order band reported as the "
                          "'common subset' shared across all stream-order experiments")

### Graph architecture ablations (Phase III / G-series: is the GAT helping,
#### hurting, or irrelevant to the per-reach Q skill seen in Phase I?)
parser.add_argument("--use-direct-edges", action="store_true", default=False,
                     help="use direct (uncoarsened) upstream-downstream edges only, "
                          "instead of the default coarsened graph (G02)")
parser.add_argument("--lambda-smooth", type=float, default=1e-3,
                     help="graph smoothness penalty weight on predicted Q differences "
                          "along edges; set to 0 to disable it entirely (G01)")
parser.add_argument("--gat-layers", type=int, default=2,
                     help="number of stacked GAT layers (default 2; try 1 or 3 for G04/G05)")
parser.add_argument("--no-gat", action="store_true", default=False,
                     help="skip graph attention entirely and feed the per-node LSTM "
                          "embedding straight to the output head (LSTM-only ablation, G03)")
parser.add_argument("--bidirectional-edges", action="store_true", default=False,
                     help="add the reverse (downstream->upstream) edge for every "
                          "upstream->downstream edge, so attention can also propagate "
                          "downstream context toward smaller tributaries (G06)")
args = parser.parse_args()

### Input data paths
#### Node table: daily COMID×date records with met drivers, discharge, and temperature.
#### Edge table: NHDPlus fromnode/tonode routing used to build the directed graph.
#### Paths are resolved relative to the repository root for portability.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NODES_PATH = PROJECT_ROOT / "data" / "daily_drivers" / "hydro_temp_data_trainval_2013plus.parquet"
EDGES_PATH = PROJECT_ROOT / "data" / "daily_drivers" / "edge_list.parquet"

### Temporal split & sequence
#### Temporal split: windows whose target date falls on/after SPLIT_DATE are the
#### held-out test set and are touched exactly once, for final reporting in
#### Section 12. VAL_DAYS carves an independent validation window out of the
#### days immediately before SPLIT_DATE; that window (not the test set) drives
#### early stopping, LR scheduling, and best-epoch model selection in Section 11.
#### Everything strictly before VAL_START_DATE is training data. LOOKBACK sets how
#### many antecedent days feed each window; both LOOKBACK and the stream order
#### band below come from --lookback / --streamorder-min / --streamorder-max.
SPLIT_DATE     = pd.Timestamp("2025-01-01")
VAL_DAYS       = args.val_days
VAL_START_DATE = SPLIT_DATE - pd.Timedelta(days=VAL_DAYS)
LOOKBACK       = args.lookback               # days of antecedent input per window

### Model hyperparameters
#### Compact single-layer LSTM captures short-term temporal dynamics at each reach;
#### two stacked GAT layers then propagate those embeddings downstream through the
#### routed river graph.
H_LSTM       = 64
LSTM_LAYERS  = 1
LSTM_DROPOUT = 0.2

H_GNN        = 64
GNN_DROPOUT  = 0.2
GAT_HEADS    = 4                            # multi-head attention in each GAT layer

### Training schedule
EPOCHS       = args.epochs
LR           = 1e-4
BATCH_SIZE   = 1                            # one full graph window per step
PATIENCE     = 10                           # early-stopping patience (epochs)

### Modeling domain
#### Stream order range filter (inclusive on both ends), replacing the old
#### single min-only threshold so specific order bands can be tested in
#### isolation. First verification run targets orders 8-9 only.
STREAMORDER_MIN = args.streamorder_min
STREAMORDER_MAX = args.streamorder_max

### Feature groups
#### Meteorological drivers from GridMET: precipitation, radiation, wind, vapor
#### pressure, humidity, and basin-averaged precip. Temperature variables were
#### converted from Kelvin to °C; mean daily temp is derived from tmax/tmin.
#### PET (represented via Ra) captures atmospheric water demand beyond precip alone.
BASE_FEATURES = [
    "pr", "srad", "vs", "vpd", "sph",
    "tmax_c", "tmin_c", "tmean_c",
    "rmin", "rmax", "Ra", "bsn_pr"
]

### Static catchment attributes
#### Fixed per COMID, constant through time.
STATIC_FEATURES = ["area_m2", "Slope", "LengthKM", "streamorde"]

### Seasonal features
#### Sine/cosine encoding of day-of-year: preserves the cyclical annual structure
#### so Dec 31 and Jan 1 remain adjacent rather than artificially far apart.
#### Excluded entirely via --no-include-doy (Experiment C); --doy-component sin/cos
#### keeps only one of the pair, testing whether a single seasonal component
#### carries the same signal as both together (ex- series).
if not args.include_doy:
    SEASONAL_FEATURES = []
elif args.doy_component == "sin":
    SEASONAL_FEATURES = ["DOY_sin"]
elif args.doy_component == "cos":
    SEASONAL_FEATURES = ["DOY_cos"]
else:
    SEASONAL_FEATURES = ["DOY_sin", "DOY_cos"]

### Targets
Q_COL    = "q_m3s_obs"
T_COL    = "water_temp_C"
TARGETS  = [Q_COL, T_COL]

### Task weights
#### Discharge is the primary target (0.7 default); temperature is secondary
#### (0.3 default) but still requires the shared representation to learn it.
#### Overridable via --q-weight / --t-weight for Experiment B's sensitivity sweep.
Q_WEIGHT     = args.q_weight
T_WEIGHT     = args.t_weight
TASK_WEIGHTS = torch.tensor([Q_WEIGHT, T_WEIGHT], dtype=torch.float32)

### High-flow upweighting
#### Ramps residual weight up to Q_HIGH_MAX_MULT for observations above the
#### threshold, helping the model better represent hydrologically important peaks.
#### Set --q-high-max-mult 1.0 for Experiment D's uniform (unweighted) loss.
Q_HIGH_THRESH_M3S = args.q_high_thresh
Q_HIGH_MAX_MULT   = args.q_high_max_mult
Q_HIGH_POWER      = 1.0

### Log scaling training option (experimental)
#### Jake's suggestion: discharge is heavy right-tailed, so log1p-transforming
#### Q before MinMax scaling compresses high-flow outliers and may make the
#### NSE-like loss easier to optimize. Applied to the Q column only, before
#### scaler_y is fit, and inverted (expm1) after inverse_transform at eval
#### time — see Section 6 and Section 12. Off by default; this is its own
#### experiment to run, not part of the first stream-order/lookback test.
USE_LOG_SCALE_Q = args.log_scale_q

### Graph smoothness regularization
#### Optional L2 penalty on predicted Q differences along directed edges.
#### Discourages physically implausible jumps in discharge across connected reaches,
#### but may also discourage legitimate differences between a reach and its
#### tributary (G01 tests --lambda-smooth 0 to check this directly).
LAMBDA_SMOOTH        = args.lambda_smooth   # tune between 1e-4 and 1e-2; 0 disables it
USE_GRAPH_SMOOTHNESS = LAMBDA_SMOOTH > 0

### Output paths
TARGET_COMID_FOR_PLOTS = args.target_comid or 14787569  # COMID used for single-reach diagnostics

### Model saving
#### Off by default while running experiments — repeat runs would otherwise
#### keep overwriting the same weights/scaler files in SAVE_DIR for no benefit
#### until a config is actually settled on. Pass --save-model to persist them.
SAVE_MODEL = args.save_model

### Run tag and per-run output folder
#### Every run gets its own outputs/<RUN_NAME>/ folder (figures + a written
#### record of the test variables and final metrics) so repeat experiments
#### with different settings don't overwrite each other's results.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_is_default_qt = abs(Q_WEIGHT - 0.70) < 1e-9 and abs(T_WEIGHT - 0.30) < 1e-9
_is_default_hfm = abs(Q_HIGH_MAX_MULT - 5.0) < 1e-9
_is_default_smooth = abs(LAMBDA_SMOOTH - 1e-3) < 1e-12
_doy_suffix = (
    "_noDOY" if not args.include_doy else
    "_doySinOnly" if args.doy_component == "sin" else
    "_doyCosOnly" if args.doy_component == "cos" else
    ""
)
RUN_NAME = args.run_name or (
    f"so{STREAMORDER_MIN}-{STREAMORDER_MAX}_lb{LOOKBACK}"
    f"{'_logQ' if USE_LOG_SCALE_Q else ''}"
    f"{_doy_suffix}"
    f"{'' if _is_default_qt else f'_qt{Q_WEIGHT:.2f}-{T_WEIGHT:.2f}'}"
    f"{'' if _is_default_hfm else f'_hfm{Q_HIGH_MAX_MULT:.1f}'}"
    f"{'' if _is_default_smooth else f'_lambdaSmooth{LAMBDA_SMOOTH:g}'}"
    f"{'_directEdges' if args.use_direct_edges else ''}"
    f"{'_bidir' if args.bidirectional_edges else ''}"
    f"{'_lstmOnly' if args.no_gat else (f'_gat{args.gat_layers}' if args.gat_layers != 2 else '')}"
    f"_seed{args.seed}_{RUN_TIMESTAMP}"
)

SAVE_DIR    = PROJECT_ROOT / "data" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / RUN_NAME
FEATURES_PATH       = os.path.join(SAVE_DIR, "features.pkl")
EDGE_SUSPICIOUS_OUT = os.path.join(SAVE_DIR, "edge_suspicious_checks.csv")

### Graph coarsening
#### When enabled, the full NHDPlus network is collapsed so edges connect only
#### modeled COMIDs; intermediate unobserved reaches are skipped by walking
#### downstream until the nearest observed connection is found. --use-direct-edges
#### tests sensitivity to this topology construction choice (G02).
USE_COARSENED_GRAPH = not args.use_direct_edges
MAX_HOPS_COARSEN    = 5000
COARSE_MODE         = "nearest"

### Graph message-passing depth / ablation (G03-G05)
#### USE_GAT=False skips graph attention entirely (LSTM-only baseline, isolates
#### how much the GAT is contributing vs. the per-reach temporal signal alone).
#### N_GAT_LAYERS controls stacked-layer depth when GAT is used.
USE_GAT      = not args.no_gat
N_GAT_LAYERS = args.gat_layers

MODEL_FILENAME = "graph_lstm_gat_multitask_q_t_maskedloss_sigmoid_head.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Reproducibility
#### Fixed seed so repeat runs of the same config are comparable; promising
#### configs should additionally be rerun with --seed 1 / --seed 2 etc.
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

### Record the test variables before anything runs
#### Written first so a run's config is on disk even if training is
#### interrupted; final metrics are appended to the same file in Section 12.
os.makedirs(OUTPUTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUTPUTS_DIR, "results.txt")

with open(RESULTS_PATH, "w") as f:
    f.write(f"Run: {RUN_NAME}\n")
    f.write(f"Timestamp: {RUN_TIMESTAMP}\n\n")
    f.write("=== Test variables ===\n")
    for key, val in {
        "NODES_PATH": NODES_PATH,
        "EDGES_PATH": EDGES_PATH,
        "SPLIT_DATE": SPLIT_DATE.date(),
        "VAL_DAYS": VAL_DAYS,
        "VAL_START_DATE": VAL_START_DATE.date(),
        "LOOKBACK": LOOKBACK,
        "SEED": SEED,
        "STREAMORDER_MIN": STREAMORDER_MIN,
        "STREAMORDER_MAX": STREAMORDER_MAX,
        "COMMON_SO_MIN": args.common_so_min,
        "COMMON_SO_MAX": args.common_so_max,
        "USE_LOG_SCALE_Q": USE_LOG_SCALE_Q,
        "INCLUDE_DOY": args.include_doy,
        "DOY_COMPONENT": args.doy_component,
        "Q_WEIGHT": Q_WEIGHT,
        "T_WEIGHT": T_WEIGHT,
        "H_LSTM": H_LSTM,
        "LSTM_LAYERS": LSTM_LAYERS,
        "LSTM_DROPOUT": LSTM_DROPOUT,
        "H_GNN": H_GNN,
        "GNN_DROPOUT": GNN_DROPOUT,
        "GAT_HEADS": GAT_HEADS,
        "USE_GAT": USE_GAT,
        "N_GAT_LAYERS": N_GAT_LAYERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "PATIENCE": PATIENCE,
        "TASK_WEIGHTS": TASK_WEIGHTS.tolist(),
        "Q_HIGH_THRESH_M3S": Q_HIGH_THRESH_M3S,
        "Q_HIGH_MAX_MULT": Q_HIGH_MAX_MULT,
        "Q_HIGH_POWER": Q_HIGH_POWER,
        "USE_GRAPH_SMOOTHNESS": USE_GRAPH_SMOOTHNESS,
        "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
        "USE_COARSENED_GRAPH": USE_COARSENED_GRAPH,
        "MAX_HOPS_COARSEN": MAX_HOPS_COARSEN,
        "COARSE_MODE": COARSE_MODE,
        "BIDIRECTIONAL_EDGES": args.bidirectional_edges,
        "TARGET_COMID_FOR_PLOTS": TARGET_COMID_FOR_PLOTS,
        "SAVE_MODEL": SAVE_MODEL,
        "BASE_FEATURES": BASE_FEATURES,
        "STATIC_FEATURES": STATIC_FEATURES,
        "SEASONAL_FEATURES": SEASONAL_FEATURES,
    }.items():
        f.write(f"{key}: {val}\n")

print(f"\nRun '{RUN_NAME}'")
print(f"Test variables written to: {RESULTS_PATH}")
print(f"Outputs for this run: {OUTPUTS_DIR}")


## 3. Utility functions

SENTINEL_MISS = -9998.0

### Replace -9998 sentinels with NaN
def replace_sentinel_with_nan(df: pd.DataFrame, cols):
    """Replace -9998 sentinels with NaN in selected numeric columns."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            df.loc[x == SENTINEL_MISS, c] = np.nan
    return df

### Repair negative discharge by per-COMID interpolation
def interpolate_negative_q_by_comid(
    df: pd.DataFrame,
    q_col: str,
    date_col: str = "date",
    comid_col: str = "comid",
    q_floor: float = 0.0,
    flag_col: str = "q_was_interpolated",
):
    """
    Repair negative discharge values by COMID time interpolation.

    Steps:
      1) Coerce q to numeric
      2) Mark q < q_floor as NaN
      3) Interpolate linearly within each COMID over time (inside only)
      4) Clip Q to >= 0 for numerical safety

    Also stamps `flag_col` (bool) marking rows whose Q value was filled by
    interpolation rather than a genuine measurement, so downstream sample-count
    reporting can distinguish measured vs. repaired Q observations.

    Returns:
      repaired_df, summary_dict
    """
    out = df.copy()
    out = out.sort_values([comid_col, date_col]).reset_index(drop=True)

    out[q_col] = pd.to_numeric(out[q_col], errors="coerce")

    neg_mask_before = out[q_col].notna() & (out[q_col] < q_floor)
    n_neg_before = int(neg_mask_before.sum())

    # mark bad negatives as missing
    out.loc[neg_mask_before, q_col] = np.nan
    missing_before_interp = out[q_col].isna()

    # interpolate per COMID in time order; no endpoint extrapolation
    out[q_col] = (
        out.groupby(comid_col, group_keys=False)[q_col]
           .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
    )

    # rows that were missing (incl. negative-flagged) but got filled by interpolation
    out[flag_col] = missing_before_interp & out[q_col].notna()

    # safety clip
    out[q_col] = out[q_col].clip(lower=0.0)

    summary = {
        "q_negatives_marked_nan_before_interp": n_neg_before,
        "q_nan_after_interp": int(out[q_col].isna().sum()),
        "q_interpolated_count": int(out[flag_col].sum()),
        "q_min_after_repair": float(np.nanmin(out[q_col].to_numpy())) if out[q_col].notna().any() else np.nan,
    }
    return out, summary

### Sanity-print feature ranges
def print_feature_sanity_ranges(nodes_df_raw: pd.DataFrame, FEATURES: list):
    """Quick sanity print for a few statics to catch order/magnitude issues early."""
    check_cols = [c for c in ["area_m2", "areasqkm", "Slope", "LengthKM", "streamorde"] if c in FEATURES]
    if not check_cols:
        return
    print("\n=== Raw feature sanity (min/max) for key statics ===")
    for c in check_cols:
        v = pd.to_numeric(nodes_df_raw[c], errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            print(f"{c:>10s}: all NA")
        else:
            print(f"{c:>10s}: min={np.min(v):.6g}  max={np.max(v):.6g}")

### Scaler min / max diagnostic
def scaler_feature_diagnostic(scaler: MinMaxScaler, FEATURES: list, max_lines=40):
    """Print scaler min/max pairs to verify they align with expected magnitudes."""
    if not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
        print("\nWARNING: scaler missing data_min_/data_max_; cannot run scaler diagnostics.")
        return

    mins = scaler.data_min_.astype(float)
    maxs = scaler.data_max_.astype(float)

    priority = [f for f in ["area_m2", "areasqkm", "Slope", "LengthKM", "streamorde",
                            "pr", "srad", "tmean_c", "vs", "vpd", "sph",
                            "tmax_c", "tmin_c", "rmin", "rmax", "Ra", "bsn_pr"]
                if f in FEATURES]
    fallback = [f for f in FEATURES[:min(len(FEATURES), 12)] if f not in priority]
    show = (priority + fallback)[:max_lines]

    print("\n=== Scaler_X training min/max (selected FEATURES) ===")
    for f in show:
        j = FEATURES.index(f)
        print(f"{f:>10s} | train_min={mins[j]:.6g}  train_max={maxs[j]:.6g}")

### Build upstream -> downstream COMID edge pairs from fromnode / tonode
def build_edge_pairs_from_nodes(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct upstream->downstream COMID edges by joining on node ids:
      (comid_up, tonode) matches (comid_dn, fromnode)
    Returns edge_pairs_all with columns [comid_up, comid_dn] (deduped).
    """
    edges_df = edges_df.dropna(subset=["fromnode", "tonode", "comid"]).copy()
    up = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_up"})
    dn = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_dn"})

    edge_pairs_all = (
        up.merge(dn, left_on="tonode", right_on="fromnode", how="inner")[["comid_up", "comid_dn"]]
          .drop_duplicates()
          .reset_index(drop=True)
    )
    return edge_pairs_all

### Coarsen the network to edges between observed COMIDs
def build_coarsened_edges(edge_pairs_all: pd.DataFrame, observed_comids: set,
                          max_hops=5000, mode="nearest") -> pd.DataFrame:
    """
    Collapse the full COMID network down to edges between observed COMIDs only.

    For each observed COMID, walk downstream through the full graph until you hit another
    observed COMID. Add an edge (src_observed -> downstream_observed).
      mode="nearest": stop at first encountered observed downstream node.
    """
    dn_map = {}
    for u, v in edge_pairs_all[["comid_up", "comid_dn"]].itertuples(index=False):
        try:
            u_i = int(u); v_i = int(v)
        except Exception:
            continue
        dn_map.setdefault(u_i, []).append(v_i)

    coarse_edges = set()

    for src in observed_comids:
        src = int(src)
        seen = {src}
        frontier = [src]
        hops = 0

        while frontier and hops < max_hops:
            curr = frontier.pop()

            for dn in dn_map.get(curr, []):
                if dn in seen:
                    continue
                seen.add(dn)

                if dn in observed_comids and dn != src:
                    coarse_edges.add((src, dn))
                    if mode == "nearest":
                        frontier = []
                        break
                else:
                    frontier.append(dn)

            hops += 1

    return pd.DataFrame(list(coarse_edges), columns=["comid_up", "comid_dn"])

### Edge QA checks
def edge_qa_checks(edge_pairs: pd.DataFrame, all_comids: np.ndarray) -> None:
    """Lightweight QA checks for edge health."""
    print("\n====================")
    print("EDGE QA CHECKS")
    print("====================")

    n_nodes = len(all_comids)
    n_edges = len(edge_pairs)

    print(f"Nodes in modeling set: {n_nodes}")
    print(f"Edges after filtering/coarsening: {n_edges}")
    print(f"Edge density: {n_edges / max(n_nodes,1):.4f} edges/node")

    if n_edges == 0:
        print("WARNING: No edges in graph. Model will fail because it requires edge_index.")
        return

    self_loops = int((edge_pairs["comid_up"].astype(int) == edge_pairs["comid_dn"].astype(int)).sum())
    dup_edges = int(edge_pairs.duplicated(subset=["comid_up", "comid_dn"]).sum())

    nodes_up = set(edge_pairs["comid_up"].astype(int).unique())
    nodes_dn = set(edge_pairs["comid_dn"].astype(int).unique())
    nodes_any = nodes_up | nodes_dn

    print(f"Nodes as upstream:   {len(nodes_up)} ({100*len(nodes_up)/n_nodes:.2f}%)")
    print(f"Nodes as downstream: {len(nodes_dn)} ({100*len(nodes_dn)/n_nodes:.2f}%)")
    print(f"Nodes in any edge:   {len(nodes_any)} ({100*len(nodes_any)/n_nodes:.2f}%)")
    print(f"Self-loops: {self_loops}")
    print(f"Duplicate edges: {dup_edges}")
    print("EDGE QA DONE.")

### High-flow residual weights (computed in scaled Q space)
def _q_highflow_weights_from_scaled(q_true_scaled: torch.Tensor,
                                   q_thresh_scaled: float,
                                   max_mult: float,
                                   power: float,
                                   eps: float = 1e-8) -> torch.Tensor:
    """
    Returns per-node weights w_i >= 1.
    - For q <= thresh: w=1
    - For q >  thresh: w ramps up to max_mult as q approaches 1.
    """
    if max_mult <= 1.0:
        return torch.ones_like(q_true_scaled)

    thr = torch.tensor(q_thresh_scaled, device=q_true_scaled.device, dtype=q_true_scaled.dtype)
    one = torch.tensor(1.0, device=q_true_scaled.device, dtype=q_true_scaled.dtype)

    denom = torch.clamp(one - thr, min=eps)
    frac = torch.clamp((q_true_scaled - thr) / denom, min=0.0, max=1.0)
    if power != 1.0:
        frac = torch.pow(frac, power)

    w = 1.0 + (max_mult - 1.0) * frac
    return w

### Graph smoothness loss on Q along directed edges
def graph_smoothness_loss(yhat_q: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """L2 smoothness along directed edges: mean (q_u - q_v)^2."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.tensor(0.0, device=yhat_q.device, dtype=yhat_q.dtype)
    src, dst = edge_index[0], edge_index[1]
    return torch.mean((yhat_q[src] - yhat_q[dst]) ** 2)


## 4. Study area and data loading
#### Current domain: HUC 0712 (Upper Illinois, ~10,900 mi²) and HUC 0713
#### (Lower Illinois, ~17,700 mi²), together covering the full Illinois River
#### basin across Illinois, Indiana, Michigan, and Wisconsin.
#
#### The workflow is configured to model this basin as a single hydrologic region,
#### restricting preprocessing, training, and evaluation to COMIDs within the
#### combined HUC 0712–0713 extent.  This provides a controlled setting for
#### debugging graph construction, validating data alignment, and assessing model
#### behavior before extending to additional sub-regions.
#
#### Node table: each row is a unique COMID × date combination containing daily
#### met drivers, observed discharge, and observed water temperature.
#### Edge table: NHDPlus fromnode/tonode routing used to derive the directed graph.

### Stream-order predicate pushed into the parquet read itself: NODES_PATH is
#### ~12GB / 94M rows across the full basin, but any one experiment only needs
#### a narrow stream-order band (e.g. SO8-9 is ~7M rows). Filtering at read time
#### (instead of loading everything then filtering a dataframe) skips reading
#### whole row groups and cuts load time from minutes to seconds per run —
#### this script is run many times across the experiment matrix.
nodes_df_raw = pd.read_parquet(
    NODES_PATH,
    filters=[("streamorde", ">=", STREAMORDER_MIN), ("streamorde", "<=", STREAMORDER_MAX)],
)
print("NODES_PATH:", NODES_PATH)
print("nodes_df_raw rows:", len(nodes_df_raw))
print("unique COMIDs in nodes_df_raw:", nodes_df_raw["comid"].nunique())
print("min date:", nodes_df_raw["date"].min(), "max date:", nodes_df_raw["date"].max())

edges_df = pd.read_parquet(EDGES_PATH)

assert {"comid", "date", Q_COL}.issubset(nodes_df_raw.columns), f"{Q_COL} missing"
assert T_COL in nodes_df_raw.columns, f"{T_COL} missing"

nodes_df_raw["date"] = pd.to_datetime(nodes_df_raw["date"])
nodes_df_raw = nodes_df_raw.sort_values(["date", "comid"]).reset_index(drop=True)

### Pre-QC range check
#### Inspect raw target distributions before any repair or scaling
q_min_raw = pd.to_numeric(nodes_df_raw[Q_COL], errors="coerce").min()
t_min_raw = pd.to_numeric(nodes_df_raw[T_COL], errors="coerce").min()

print(f"Raw minimum {Q_COL}: {q_min_raw}")
print(f"Raw minimum {T_COL}: {t_min_raw}")

### Observed streamflow QC
#### Daily discharge from USGS NWIS was standardized and converted from cfs to
#### m³/s before joining to COMIDs.  Negative values are physically impossible;
#### they are treated as data errors, masked as NaN, and filled by linear
#### interpolation within each COMID's time series (interior gaps only).
nodes_df_raw, q_repair_summary = interpolate_negative_q_by_comid(
    nodes_df_raw,
    q_col=Q_COL,
    date_col="date",
    comid_col="comid",
    q_floor=0.0
)

print("\n=== Q repair by COMID interpolation ===")
for k, v in q_repair_summary.items():
    print(f"{k}: {v}")

print(f"Post-repair minimum {Q_COL}: {pd.to_numeric(nodes_df_raw[Q_COL], errors='coerce').min()}")
print(f"Post-repair minimum {T_COL}: {pd.to_numeric(nodes_df_raw[T_COL], errors='coerce').min()}")

### Water temperature QC
#### Physically implausible temperature readings (below -5 °C or above 60 °C)
#### are nulled before masks and scalers are computed, preventing outliers from
#### corrupting the training signal or the scaled-space bounds.
TEMP_QC_MIN_C = -5.0
TEMP_QC_MAX_C = 60.0

t_raw = pd.to_numeric(nodes_df_raw[T_COL], errors="coerce")
bad_t = t_raw.notna() & ((t_raw < TEMP_QC_MIN_C) | (t_raw > TEMP_QC_MAX_C))
if bad_t.any():
    print(f"[TempQC] Setting {int(bad_t.sum()):,} implausible temps to NaN "
          f"(<{TEMP_QC_MIN_C}C or >{TEMP_QC_MAX_C}C) before masks/scaling.")
    nodes_df_raw.loc[bad_t, T_COL] = np.nan

### Snapshot observation availability (before any imputation or scaling)
#### These boolean masks are the ground truth for which nodes/dates have real
#### observations; they gate the loss function so missing values are never
#### treated as true supervision signal. q_avail_raw includes Q values that were
#### repaired by interpolation (see interpolate_negative_q_by_comid above);
#### q_avail_measured excludes those, isolating genuine USGS measurements for
#### sample-count reporting (prescription note: distinguish measured vs.
#### interpolated Q in sample counts). Temperature has no interpolation step,
#### so temp_avail_raw is already measured-only.
nodes_df_raw["temp_avail_raw"]  = ~nodes_df_raw[T_COL].isna()
nodes_df_raw["q_avail_raw"]     = ~nodes_df_raw[Q_COL].isna()
nodes_df_raw["q_avail_measured"] = nodes_df_raw["q_avail_raw"] & ~nodes_df_raw["q_was_interpolated"]

### Modeling domain: stream order filter
#### Applied to both train and test to keep the domain consistent.
#### Restricting to STREAMORDER_MIN <= order <= STREAMORDER_MAX isolates a
#### specific order band for comparison across experiments (e.g. orders 8-9).
so_filter = pd.to_numeric(nodes_df_raw["streamorde"], errors="coerce")

keep_rows = (so_filter >= STREAMORDER_MIN) & (so_filter <= STREAMORDER_MAX)

n_before = len(nodes_df_raw)
nodes_df_raw = nodes_df_raw.loc[keep_rows].copy()
nodes_df_raw = nodes_df_raw.sort_values(["date", "comid"]).reset_index(drop=True)
n_after = len(nodes_df_raw)

print("\n=== Applied stream order filter ===")
print(f"Rule: keep {STREAMORDER_MIN} <= streamorde <= {STREAMORDER_MAX}")
print(f"Rows before: {n_before:,}")
print(f"Rows after : {n_after:,}")
print(f"Dropped    : {n_before - n_after:,}")
print(f"Unique COMIDs after filter: {nodes_df_raw['comid'].nunique():,}")
print("Remaining stream orders:", sorted(nodes_df_raw["streamorde"].dropna().unique()))

### Feature assembly
#### Final feature vector = met drivers + DOY seasonality + static catchment attrs.
#### Order is fixed here and persisted to disk so inference can reproduce the
#### exact scaler alignment used during training.
FEATURES = BASE_FEATURES + SEASONAL_FEATURES + STATIC_FEATURES

required_cols = {"comid", "date", Q_COL, T_COL, "temp_avail_raw", "q_avail_raw"} | set(FEATURES)
missing_cols = [c for c in sorted(required_cols) if c not in nodes_df_raw.columns]
if missing_cols:
    raise ValueError("Missing required columns in nodes_df_raw:\n" + "\n".join(missing_cols))

### Ensure numeric coercion for FEATURES + TARGETS before dtype checks / scaling
for c in FEATURES + TARGETS:
    nodes_df_raw[c] = pd.to_numeric(nodes_df_raw[c], errors="coerce")

non_numeric = [f for f in FEATURES if not pd.api.types.is_numeric_dtype(nodes_df_raw[f])]
if non_numeric:
    raise TypeError("These FEATURES are not numeric dtypes:\n" + "\n".join(non_numeric))

print(f"Using explicit FEATURES (n={len(FEATURES)}):")
print(FEATURES)

if SAVE_MODEL:
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(FEATURES, FEATURES_PATH)
    print(f"Saved FEATURES to: {FEATURES_PATH}")

keep_cols = ["comid", "date"] + FEATURES + TARGETS + ["temp_avail_raw", "q_avail_raw", "q_avail_measured"]
nodes_df = nodes_df_raw[keep_cols].copy()

### Replace -9998 sentinels with NaN
#### Recommended before scaling
nodes_df = replace_sentinel_with_nan(nodes_df, FEATURES + TARGETS)
print_feature_sanity_ranges(nodes_df, FEATURES)

## 5. Graph construction
#### The directed graph encodes the routed river network: nodes are COMIDs,
#### edges follow upstream → downstream flow direction derived from NHDPlus
#### fromnode/tonode identifiers.  When coarsening is enabled, intermediate
#### unobserved reaches are collapsed so every edge connects two modeled COMIDs.

### COMID → integer index mapping
all_comids = np.sort(nodes_df["comid"].unique())
comid_to_idx = {c: i for i, c in enumerate(all_comids)}
n_nodes = len(all_comids)

### Edge construction
#### Match each reach's tonode to the downstream reach's fromnode to get directed
#### upstream→downstream COMID pairs, then optionally coarsen to observed nodes.
edge_pairs_all = build_edge_pairs_from_nodes(edges_df)

edge_pairs_direct = edge_pairs_all[
    edge_pairs_all["comid_up"].isin(all_comids) & edge_pairs_all["comid_dn"].isin(all_comids)
].drop_duplicates().reset_index(drop=True)

if USE_COARSENED_GRAPH:
    observed_comids = set(all_comids.astype(int))
    edge_pairs = build_coarsened_edges(
        edge_pairs_all=edge_pairs_all,
        observed_comids=observed_comids,
        max_hops=MAX_HOPS_COARSEN,
        mode=COARSE_MODE
    )
    edge_pairs["comid_up"] = edge_pairs["comid_up"].astype(int)
    edge_pairs["comid_dn"] = edge_pairs["comid_dn"].astype(int)
    edge_pairs = edge_pairs[
        edge_pairs["comid_up"].isin(observed_comids) & edge_pairs["comid_dn"].isin(observed_comids)
    ].drop_duplicates().reset_index(drop=True)

    print("\n=== Edge diagnostics (COARSENED) ===")
    print(f"edge_pairs_direct edges (direct-only): {len(edge_pairs_direct)}")
    print(f"edge_pairs coarsened edges: {len(edge_pairs)}")
else:
    edge_pairs = edge_pairs_direct
    print("\n=== Edge diagnostics (DIRECT ONLY) ===")
    print(f"edge_pairs filtered edges: {len(edge_pairs)}")

### Optional bidirectional edges (G06)
#### Routing is naturally one-directional (upstream->downstream), but GAT
#### attention only lets a reach hear from its upstream neighbors under that
#### convention. Adding the reverse edge lets downstream context (e.g. a
#### large mainstem reach) also propagate back toward its small tributaries,
#### testing whether that helps the smaller, harder-to-predict stream orders.
if args.bidirectional_edges and len(edge_pairs) > 0:
    reversed_pairs = edge_pairs.rename(columns={"comid_up": "comid_dn", "comid_dn": "comid_up"})
    edge_pairs = pd.concat([edge_pairs, reversed_pairs], ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"\n[BidirectionalEdges] Added reverse edges: {len(edge_pairs)} total directed edges "
          f"(both directions of each connection).")

if len(edge_pairs) > 0:
    src = edge_pairs["comid_up"].map(comid_to_idx).to_numpy()
    dst = edge_pairs["comid_dn"].map(comid_to_idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).to(device)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} directed edges")
edge_qa_checks(edge_pairs, all_comids)

if edge_index.numel() == 0:
    raise RuntimeError("edge_index is empty. Ensure edges exist and USE_COARSENED_GRAPH settings are correct.")

## 6. Feature scaling
#### MinMaxScaler is fit on training rows only to prevent data leakage.
#### NaN values are temporarily filled with 0 for scaler mechanics; actual
#### supervision is gated by the pre-computed availability masks so those
#### filled values never act as true training signal.

### Optional log1p transform of discharge (training option)
#### Applied to nodes_df in physical units, before scaler_y is fit, so the
#### scaler operates on the log-compressed distribution. Inverted (expm1)
#### after scaler_y.inverse_transform in Section 12 so metrics/plots stay
#### in physical units regardless of this flag.
if USE_LOG_SCALE_Q:
    nodes_df[Q_COL] = np.log1p(nodes_df[Q_COL])
    print(f"\n[LogScale] Applied log1p to {Q_COL} before scaling.")

### Scalers are fit on true training rows only (strictly before the validation
#### window), so neither the validation nor the test period leaks into scaling.
train_mask_rows = nodes_df["date"] < VAL_START_DATE
if train_mask_rows.sum() == 0:
    raise ValueError("No training rows found before VAL_START_DATE. Check SPLIT_DATE/--val-days or data date range.")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
fit_X = np.nan_to_num(nodes_df.loc[train_mask_rows, FEATURES].values, nan=0.0)
fit_y = np.nan_to_num(nodes_df.loc[train_mask_rows, TARGETS].values, nan=0.0)

scaler_X.fit(fit_X)
scaler_y.fit(fit_y)

if hasattr(scaler_X, "n_features_in_") and scaler_X.n_features_in_ != len(FEATURES):
    raise RuntimeError(f"scaler_X expects {scaler_X.n_features_in_} features but FEATURES has {len(FEATURES)}.")

scaler_feature_diagnostic(scaler_X, FEATURES, max_lines=40)

### Transform all rows using train-fitted scalers
X_all = np.nan_to_num(nodes_df[FEATURES].values, nan=0.0)
Y_all = np.nan_to_num(nodes_df[TARGETS].values, nan=0.0)
nodes_df[FEATURES] = scaler_X.transform(X_all)
nodes_df[TARGETS]  = scaler_y.transform(Y_all)

### Precompute scaled threshold for high-flow Q weighting
q_idx = TARGETS.index(Q_COL)
q_min = float(scaler_y.data_min_[q_idx])
q_max = float(scaler_y.data_max_[q_idx])
q_range = max(q_max - q_min, 1e-12)
q_thresh_phys = np.log1p(Q_HIGH_THRESH_M3S) if USE_LOG_SCALE_Q else Q_HIGH_THRESH_M3S
q_thresh_scaled = float((q_thresh_phys - q_min) / q_range)
q_thresh_scaled = float(np.clip(q_thresh_scaled, 0.0, 1.0))

print(f"\nQ weighting setup:")
print(f"  Q_HIGH_THRESH_M3S = {Q_HIGH_THRESH_M3S}")
print(f"  scaler_y q_min/q_max = {q_min:.6g} / {q_max:.6g}")
print(f"  q_thresh_scaled = {q_thresh_scaled:.6f}")
print(f"  Q_HIGH_MAX_MULT = {Q_HIGH_MAX_MULT}, Q_HIGH_POWER = {Q_HIGH_POWER}")

## 7. Tensor assembly
#### Reshape the scaled node table into dense arrays for graph-sequence modeling:
####   X_TNF  [T, N, F] — predictors at every time step and node
####   Y_TN2  [T, N, 2] — discharge and temperature targets
####   Qmask  [T, N]    — True where discharge observations exist
####   Tmask  [T, N]    — True where temperature observations exist
dates = np.sort(nodes_df["date"].unique())
Tsteps = len(dates)
Fdim = len(FEATURES)

X_TNF     = np.zeros((Tsteps, n_nodes, Fdim), dtype=np.float32)
Y_TN2     = np.zeros((Tsteps, n_nodes, 2),   dtype=np.float32)
Tmask_TN  = np.zeros((Tsteps, n_nodes),      dtype=bool)
Qmask_TN  = np.zeros((Tsteps, n_nodes),      dtype=bool)

date_to_idx = {d: i for i, d in enumerate(dates)}

### Vectorized scatter into the dense [T, N, F] arrays via fancy indexing,
#### instead of looping over every date and re-filtering nodes_df each time
#### (that per-date loop rescans the whole table O(n_dates) times; at ~4800
#### dates this dominated runtime, especially for wider stream-order bands).
t_idx_arr = nodes_df["date"].map(date_to_idx).to_numpy()
n_idx_arr = nodes_df["comid"].map(comid_to_idx).to_numpy()

X_TNF[t_idx_arr, n_idx_arr, :] = nodes_df[FEATURES].to_numpy(dtype=np.float32)
Y_TN2[t_idx_arr, n_idx_arr, 0] = nodes_df[Q_COL].to_numpy(dtype=np.float32)
Y_TN2[t_idx_arr, n_idx_arr, 1] = nodes_df[T_COL].to_numpy(dtype=np.float32)

# masks come from raw availability columns (not scaled)
Tmask_TN[t_idx_arr, n_idx_arr] = nodes_df["temp_avail_raw"].to_numpy(dtype=bool)
### Fix #4: Qmask must come from q_avail_measured (genuine USGS measurements only),
#### not q_avail_raw, which also marks interpolated/repaired Q values as "available"
#### and lets the model train/be scored against values it never actually observed.
Qmask_TN[t_idx_arr, n_idx_arr] = nodes_df["q_avail_measured"].to_numpy(dtype=bool)

## 8. Train–validation–test datasets
#### Each sample is a LOOKBACK-day input window across all nodes ending at time t,
#### paired with targets for that final day.  The split is strictly temporal and
#### three-way: windows before VAL_START_DATE are training; windows in
#### [VAL_START_DATE, SPLIT_DATE) are an independent validation set used for
#### early stopping / LR scheduling / best-epoch selection (Section 11); windows
#### on/after SPLIT_DATE are the held-out test set, touched only once, in
#### Section 12, for final reporting. Because each batch spans the full graph,
#### BATCH_SIZE = 1 (one window = one graph time step).
class GraphSeqDataset(Dataset):
    def __init__(self, X_TNF, Y_TN2, Tmask_TN, Qmask_TN, lookback, dates):
        self.X = X_TNF
        self.Y = Y_TN2
        self.Mt = Tmask_TN
        self.Mq = Qmask_TN
        self.lb = lookback
        self.dates = dates

    def __len__(self):
        return self.X.shape[0] - self.lb

    def __getitem__(self, idx):
        t = idx + self.lb
        t0 = t - self.lb + 1
        t1 = t + 1

        x_win = self.X[t0:t1, :, :]  # [lb, N, F]
        x_win = np.transpose(x_win, (1, 0, 2)).astype(np.float32)  # [N, lb, F]

        y_t = self.Y[t, :, :].astype(np.float32)   # [N, 2]
        mt  = self.Mt[t, :].astype(bool)           # [N]
        mq  = self.Mq[t, :].astype(bool)           # [N]

        return (
            torch.from_numpy(x_win),
            torch.from_numpy(y_t),
            torch.from_numpy(mt),
            torch.from_numpy(mq),
        )

dataset = GraphSeqDataset(X_TNF, Y_TN2, Tmask_TN, Qmask_TN, LOOKBACK, dates)

t_dates = dates[LOOKBACK:]  # dates aligned to dataset indices
train_indices = [i for i, d in enumerate(t_dates) if d < VAL_START_DATE]
val_indices   = [i for i, d in enumerate(t_dates) if VAL_START_DATE <= d < SPLIT_DATE]
test_indices  = [i for i, d in enumerate(t_dates) if d >= SPLIT_DATE]

assert len(train_indices) > 0, "No training samples before the validation window."
assert len(val_indices)   > 0, "No validation samples in [VAL_START_DATE, SPLIT_DATE); check --val-days."
assert len(test_indices)  > 0, "No testing samples on/after split date."

train_ds = Subset(dataset, train_indices)
val_ds   = Subset(dataset, val_indices)
test_ds  = Subset(dataset,  test_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining on {len(train_ds)} time steps (< {VAL_START_DATE.date()}); "
      f"validating (early stopping/model selection) on {len(val_ds)} time steps "
      f"[{VAL_START_DATE.date()} .. {SPLIT_DATE.date()}); "
      f"held-out testing on {len(test_ds)} time steps (≥ {SPLIT_DATE.date()}), "
      f"touched only once for final metrics.")

## 9. Model architecture (Graph LSTM + GAT multitask)
#### For each node, the 7-day feature sequence is passed through an LSTM to
#### encode short-term temporal dynamics (Kratzert et al., 2018).  The final
#### hidden state becomes that node's embedding for the current day.  N_GAT_LAYERS
#### stacked GAT layers then propagate those embeddings downstream through the
#### directed river graph (Veličković et al., 2018; Jia et al., 2021), so each
#### reach can receive information from its upstream neighbors — unless USE_GAT
#### is False (--no-gat, Experiment G03), in which case the LSTM embedding feeds
#### the head directly, isolating how much the graph step is actually contributing.
#### A linear head maps the final embedding to two outputs (Q, T), and a sigmoid
#### activation bounds both predictions to [0, 1] in scaled space — added after
#### observing exploding predictions during early model development.
class GraphLSTM_MTL(nn.Module):
    def __init__(self, in_dim, lstm_hidden, lstm_layers, lstm_dropout,
                 gnn_hidden, gnn_dropout, heads=4, out_dim=2,
                 use_gat=True, n_gat_layers=2):
        super().__init__()
        self.use_gat = use_gat
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            batch_first=True
        )
        self.drop_lstm = nn.Dropout(lstm_dropout)
        self.drop_gnn = nn.Dropout(gnn_dropout)

        if use_gat:
            assert n_gat_layers >= 1, "n_gat_layers must be >= 1 when use_gat=True"
            gat_in = lstm_hidden
            gat_layers = []
            for _ in range(n_gat_layers):
                gat_layers.append(GATConv(gat_in, gnn_hidden // heads, heads=heads, dropout=gnn_dropout))
                gat_in = gnn_hidden
            self.gat_layers = nn.ModuleList(gat_layers)
            head_in = gnn_hidden
        else:
            self.gat_layers = nn.ModuleList()
            head_in = lstm_hidden

        self.head = nn.Linear(head_in, out_dim)

    def forward(self, x_win, edge_index, return_logits=False):
        lstm_out, _ = self.lstm(x_win)                   # [N, lb, H_LSTM]
        h = self.drop_lstm(lstm_out[:, -1, :])            # [N, H_LSTM]

        for gat in self.gat_layers:
            h = F.elu(gat(h, edge_index))                 # [N, H_GNN]
            h = self.drop_gnn(h)

        raw = self.head(h)                              # [N, 2] unconstrained logits
        # return_logits=True is used by Graph-SHAP to avoid sigmoid saturation:
        # when predictions are near 0 or 1, individual feature contributions
        # shift the output by < 1e-7, which is below float32 precision.
        # Logit space keeps the same changes in the ±1 range where float32
        # has full precision.
        return raw if return_logits else torch.sigmoid(raw)

model = GraphLSTM_MTL(
    in_dim=len(FEATURES),
    lstm_hidden=H_LSTM, lstm_layers=LSTM_LAYERS, lstm_dropout=LSTM_DROPOUT,
    gnn_hidden=H_GNN, gnn_dropout=GNN_DROPOUT, heads=GAT_HEADS, out_dim=2,
    use_gat=USE_GAT, n_gat_layers=N_GAT_LAYERS,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

## 10. Loss function and regularization
#### Discharge uses a weighted NSE-like term (numerator = weighted MSE,
#### denominator = weighted variance of observations) so the model is penalized
#### relative to the variance of the observed signal.  Temperature uses standard
#### MSE.  Both terms are masked so only observed nodes contribute to that step's
#### gradient.  Task weights (0.7 Q / 0.3 T) keep discharge as the primary
#### objective while still requiring the shared representation to learn temperature
#### (Sadler et al., 2022).  High-flow upweighting and the optional graph
#### smoothness penalty are applied on top of this base loss.
def nse_mse_mixed_loss(yhat, ytrue, w_task, mask_t=None, mask_q=None,
                       q_thresh_scaled=0.0, q_max_mult=1.0, q_power=1.0, eps=1e-8):
    """
    yhat,ytrue: [N, 2] scaled; columns [Q, T]
    mask_t: [N] bool for Temp observed
    mask_q: [N] bool for Q observed
    """
    w_task = w_task.to(yhat.device)

    # Q term (masked)
    if mask_q is None or (mask_q.sum() == 0):
        nse_term_q = torch.tensor(0.0, device=yhat.device)
        wq_use = torch.tensor(0.0, device=yhat.device)
    else:
        q_pred = yhat[mask_q, 0]
        q_true = ytrue[mask_q, 0]

        w_q = _q_highflow_weights_from_scaled(
            q_true_scaled=q_true,
            q_thresh_scaled=q_thresh_scaled,
            max_mult=q_max_mult,
            power=q_power,
            eps=eps
        )

        q_true_mean_w = torch.sum(w_q * q_true) / (torch.sum(w_q) + eps)
        num_q = torch.sum(w_q * (q_true - q_pred) ** 2)
        den_q = torch.sum(w_q * (q_true - q_true_mean_w) ** 2) + eps
        nse_term_q = num_q / den_q
        wq_use = w_task[0]

    # Temp term (masked)
    if mask_t is None or (mask_t.sum() == 0):
        mse_t = torch.tensor(0.0, device=yhat.device)
        wt_use = torch.tensor(0.0, device=yhat.device)
    else:
        t_pred = yhat[mask_t, 1]
        t_true = ytrue[mask_t, 1]
        mse_t = torch.mean((t_true - t_pred) ** 2)
        wt_use = w_task[1]

    # Combine only active tasks
    wsum = wq_use + wt_use + eps
    total = (wq_use / wsum) * nse_term_q + (wt_use / wsum) * mse_t
    return total, nse_term_q.detach(), mse_t.detach()

## 11. Training
#### Adam optimizer with ReduceLROnPlateau scheduling.  At each epoch the script
#### iterates through all training windows, computes the multitask masked loss,
#### optionally adds the graph smoothness penalty, and updates weights.  Validation
#### is run on the independent validation windows (Section 8) after every epoch;
#### early stopping fires if validation loss does not improve within PATIENCE
#### epochs.  The best model state is retained and reloaded at the end of
#### training.  The held-out test set is never seen during this loop.
N_TRAINABLE_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {N_TRAINABLE_PARAMS:,}")

best_val = float("inf")
best_state = None
best_epoch = 0
epochs_no_improve = 0
final_epoch = 0
history_rows = []

print(f"\nTraining Graph-LSTM+GAT (multitask Q+T) on {len(train_ds)} time steps; "
      f"validating on {len(val_ds)} time steps.")
print("Output head constraint: sigmoid on BOTH Q and Temp (scaled-space [0,1])")

train_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_losses = []; tr_q = []; tr_t = []

    for x_win, y_t, m_t, m_q in train_loader:
        x_win = x_win.squeeze(0).to(device).float()  # [N, lb, F]
        y_t   = y_t.squeeze(0).to(device).float()    # [N, 2]
        m_t   = m_t.squeeze(0).to(device)            # [N] bool
        m_q   = m_q.squeeze(0).to(device)            # [N] bool

        optimizer.zero_grad()
        yhat = model(x_win, edge_index)

        loss, lq, lt = nse_mse_mixed_loss(
            yhat, y_t, TASK_WEIGHTS, mask_t=m_t, mask_q=m_q,
            q_thresh_scaled=q_thresh_scaled,
            q_max_mult=Q_HIGH_MAX_MULT,
            q_power=Q_HIGH_POWER
        )

        if USE_GRAPH_SMOOTHNESS and LAMBDA_SMOOTH > 0:
            smooth = graph_smoothness_loss(yhat[:, 0], edge_index)
            loss = loss + LAMBDA_SMOOTH * smooth

        loss.backward()
        optimizer.step()

        tr_losses.append(loss.item()); tr_q.append(lq.item()); tr_t.append(lt.item())

    model.eval()
    va_losses = []; va_q = []; va_t = []

    with torch.no_grad():
        for x_win, y_t, m_t, m_q in val_loader:
            x_win = x_win.squeeze(0).to(device).float()
            y_t   = y_t.squeeze(0).to(device).float()
            m_t   = m_t.squeeze(0).to(device)
            m_q   = m_q.squeeze(0).to(device)

            yhat  = model(x_win, edge_index)
            loss, lq, lt = nse_mse_mixed_loss(
                yhat, y_t, TASK_WEIGHTS, mask_t=m_t, mask_q=m_q,
                q_thresh_scaled=q_thresh_scaled,
                q_max_mult=Q_HIGH_MAX_MULT,
                q_power=Q_HIGH_POWER
            )

            if USE_GRAPH_SMOOTHNESS and LAMBDA_SMOOTH > 0:
                smooth = graph_smoothness_loss(yhat[:, 0], edge_index)
                loss = loss + LAMBDA_SMOOTH * smooth

            va_losses.append(loss.item()); va_q.append(lq.item()); va_t.append(lt.item())

    tr_mean = float(np.mean(tr_losses)); va_mean = float(np.mean(va_losses))
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train mix: {tr_mean:.6f} "
          f"(Q NSEterm_w {np.mean(tr_q):.6f}, T MSE {np.mean(tr_t):.6f}) "
          f"| Val mix: {va_mean:.6f} (Q NSEterm_w {np.mean(va_q):.6f}, T MSE {np.mean(va_t):.6f})")

    history_rows.append({
        "epoch": epoch,
        "train_loss": tr_mean, "train_q_nse_term": float(np.mean(tr_q)), "train_t_mse": float(np.mean(tr_t)),
        "val_loss": va_mean, "val_q_nse_term": float(np.mean(va_q)), "val_t_mse": float(np.mean(va_t)),
    })

    scheduler.step(va_mean)
    final_epoch = epoch

    if va_mean + 1e-6 < best_val:
        best_val = va_mean
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

TRAIN_ELAPSED_SEC = time.time() - train_start_time
print(f"\nTraining elapsed: {TRAIN_ELAPSED_SEC:.1f} sec over {final_epoch} epochs "
      f"(best epoch {best_epoch}, best val mix {best_val:.6f})")

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

### Persist per-epoch train/val loss curves for cross-run comparison
history_df = pd.DataFrame(history_rows)
history_df.to_csv(os.path.join(OUTPUTS_DIR, "training_history.csv"), index=False)

plt.figure(figsize=(8, 4.5))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train mix loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Val mix loss")
plt.axvline(best_epoch, linestyle="--", color="gray", alpha=0.6, label=f"Best epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Weighted NSE-term + MSE mix loss")
plt.title(f"Training/Validation Loss — {RUN_NAME}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "training_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

## 12. Evaluation in physical units
#### Predictions and targets are inverse-transformed back to original units.
#### Metrics (NSE, RMSE, MAE) are computed only at nodes and dates where
#### observations exist, using the pre-computed masks.  Stream-order breakdowns
#### let us assess whether model performance changes across network scale.
### Metric helpers (NSE, RMSE, MAE)
def nse_numpy(sim, obs):
    num = np.sum((sim - obs) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2) + 1e-12
    return 1.0 - num / den

def rmse_numpy(sim, obs):
    return float(np.sqrt(np.mean((sim - obs) ** 2)))

def mae_numpy(sim, obs):
    return float(np.mean(np.abs(sim - obs)))

def bias_numpy(sim, obs):
    """Mean bias: positive = overprediction, negative = underprediction."""
    return float(np.mean(sim - obs))

model.eval()
preds_scaled = []
obs_scaled   = []
temp_masks   = []
q_masks      = []

with torch.no_grad():
    for x_win, y_t, m_t, m_q in test_loader:
        x_win = x_win.squeeze(0).to(device).float()
        y_t_d = y_t.squeeze(0).to(device).float()
        yhat  = model(x_win, edge_index).detach().cpu().numpy()

        preds_scaled.append(yhat)
        obs_scaled.append(y_t_d.detach().cpu().numpy())
        temp_masks.append(m_t.squeeze(0).cpu().numpy().astype(bool))
        q_masks.append(m_q.squeeze(0).cpu().numpy().astype(bool))

preds_scaled = np.stack(preds_scaled, axis=0)  # [Ttest, N, 2]
obs_scaled   = np.stack(obs_scaled,   axis=0)  # [Ttest, N, 2]
temp_masks   = np.stack(temp_masks,   axis=0)  # [Ttest, N]
q_masks      = np.stack(q_masks,      axis=0)  # [Ttest, N]

print("\n=== Scaled prediction bounds check (should be within [0,1]) ===")
print(f"Q_scaled pred min/max: {preds_scaled[:,:,0].min():.6f} / {preds_scaled[:,:,0].max():.6f}")
print(f"T_scaled pred min/max: {preds_scaled[:,:,1].min():.6f} / {preds_scaled[:,:,1].max():.6f}")

### Inverse-transform scaled predictions back to physical units
def inverse_two_columns(arr_2):
    flat = arr_2.reshape(-1, 2)
    inv  = scaler_y.inverse_transform(flat)
    return inv.reshape(arr_2.shape)

pred_phys = inverse_two_columns(preds_scaled)
obs_phys  = inverse_two_columns(obs_scaled)

### Undo the log1p transform (if enabled) so metrics/plots stay in physical units
if USE_LOG_SCALE_Q:
    pred_phys[:, :, 0] = np.expm1(pred_phys[:, :, 0])
    obs_phys[:, :, 0]  = np.expm1(obs_phys[:, :, 0])

pred_phys[:, :, 0] = np.clip(pred_phys[:, :, 0], 0.0, None)

pred_q = pred_phys[:, :, 0].ravel()
true_q = obs_phys[:, :, 0].ravel()
pred_t = pred_phys[:, :, 1].ravel()
true_t = obs_phys[:, :, 1].ravel()

mask_t = temp_masks.ravel()
mask_q = q_masks.ravel()

if mask_q.any():
    q_nse  = nse_numpy(pred_q[mask_q], true_q[mask_q])
    q_rmse = rmse_numpy(pred_q[mask_q], true_q[mask_q])
    q_mae  = mae_numpy(pred_q[mask_q], true_q[mask_q])
    q_bias = bias_numpy(pred_q[mask_q], true_q[mask_q])
else:
    q_nse = q_rmse = q_mae = q_bias = np.nan

if mask_t.any():
    t_nse  = nse_numpy(pred_t[mask_t], true_t[mask_t])
    t_rmse = rmse_numpy(pred_t[mask_t], true_t[mask_t])
    t_mae  = mae_numpy(pred_t[mask_t], true_t[mask_t])
    t_bias = bias_numpy(pred_t[mask_t], true_t[mask_t])
else:
    t_nse = t_rmse = t_mae = t_bias = np.nan

print("\n=== Test Metrics (original units; observed-only) ===")
print(f"Q    -> NSE: {q_nse:.4f} | RMSE: {q_rmse:.4f} | MAE: {q_mae:.4f} | Bias: {q_bias:.4f} | n_obs={int(mask_q.sum())}")
print(f"Temp -> NSE: {t_nse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | Bias: {t_bias:.4f} | n_obs={int(mask_t.sum())}")

### Metrics by stream order
#### Stratifying by stream order reveals whether the model is stronger on
#### larger reaches (higher order) versus smaller tributaries (lower order),
#### which informs where the domain should be extended or constrained.
def calc_metrics(obs, pred):
    if len(obs) == 0:
        return {"n_obs": 0, "NSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}
    return {
        "n_obs": len(obs),
        "NSE": nse_numpy(pred, obs),
        "RMSE": rmse_numpy(pred, obs),
        "MAE": mae_numpy(pred, obs),
        "Bias": bias_numpy(pred, obs),
    }

streamorder_by_node = (
    nodes_df_raw[["comid", "streamorde"]]
    .drop_duplicates("comid")
    .set_index("comid")
    .reindex(all_comids)["streamorde"]
    .to_numpy()
)

streamorder_by_node = pd.to_numeric(streamorder_by_node, errors="coerce")
streamorder_test = np.tile(streamorder_by_node, (pred_phys.shape[0], 1))  # [Ttest, N]

so_flat   = streamorder_test.ravel()
pred_q    = pred_phys[:, :, 0].ravel()
true_q    = obs_phys[:, :, 0].ravel()
pred_t    = pred_phys[:, :, 1].ravel()
true_t    = obs_phys[:, :, 1].ravel()
mask_q_f  = q_masks.ravel()
mask_t_f  = temp_masks.ravel()

valid_so = np.isfinite(so_flat)
unique_so = sorted(np.unique(so_flat[valid_so]).astype(int))

rows = []

for so in unique_so:
    idx_q = (so_flat == so) & valid_so & mask_q_f & np.isfinite(true_q) & np.isfinite(pred_q)
    q_metrics = calc_metrics(true_q[idx_q], pred_q[idx_q])
    rows.append({
        "streamorde": so,
        "variable": "Q",
        **q_metrics
    })

    idx_t = (so_flat == so) & valid_so & mask_t_f & np.isfinite(true_t) & np.isfinite(pred_t)
    t_metrics = calc_metrics(true_t[idx_t], pred_t[idx_t])
    rows.append({
        "streamorde": so,
        "variable": "Temp",
        **t_metrics
    })

metrics_by_streamorder = pd.DataFrame(rows)

print("\n=== Test Metrics by Stream Order ===")
print(metrics_by_streamorder)

metrics_by_streamorder.to_csv(
    os.path.join(OUTPUTS_DIR, "test_metrics_by_streamorder.csv"),
    index=False
)

print("\n=== Physical plausibility diagnostics (test predictions, original units) ===")
print(f"Pred Q < 0 count: {int((pred_q < 0).sum())} / {pred_q.size}")
print(f"Pred Temp < 0 count: {int((pred_t < 0).sum())} / {pred_t.size}")
print(f"Pred Temp > 40 count: {int((pred_t > 40).sum())} / {pred_t.size}")

### Common stream-order subset metrics
#### Lets runs that model different stream-order bands (SO8-9 vs SO7-9 vs
#### SO6-9 ...) be compared on the identical set of observations, answering
#### whether expanding the graph helps or hurts the reaches already modeled
#### in the narrowest band.
COMMON_SO_MIN, COMMON_SO_MAX = args.common_so_min, args.common_so_max
common_so_mask = valid_so & (so_flat >= COMMON_SO_MIN) & (so_flat <= COMMON_SO_MAX)

idx_q_common = common_so_mask & mask_q_f & np.isfinite(true_q) & np.isfinite(pred_q)
idx_t_common = common_so_mask & mask_t_f & np.isfinite(true_t) & np.isfinite(pred_t)
common_q_metrics = calc_metrics(true_q[idx_q_common], pred_q[idx_q_common])
common_t_metrics = calc_metrics(true_t[idx_t_common], pred_t[idx_t_common])

print(f"\n=== Test Metrics on common SO{COMMON_SO_MIN}-{COMMON_SO_MAX} subset ===")
print(f"Q    -> {common_q_metrics}")
print(f"Temp -> {common_t_metrics}")

### Measured vs. interpolated Q sample counts (test period only)
#### q_avail_raw includes Q values repaired by interpolation; q_avail_measured
#### excludes those. Reported separately per the prescription's note that
#### interpolated fills shouldn't be silently counted as genuine measurements,
#### and because Q/T are sparsely supervised targets worth tracking by reach.
test_period_rows = nodes_df_raw[nodes_df_raw["date"] >= SPLIT_DATE]
n_q_any_test        = int(test_period_rows["q_avail_raw"].sum())
n_q_measured_test    = int(test_period_rows["q_avail_measured"].sum())
n_q_interpolated_test = n_q_any_test - n_q_measured_test
n_t_measured_test    = int(test_period_rows["temp_avail_raw"].sum())
n_reaches_with_q_test = int(test_period_rows.loc[test_period_rows["q_avail_measured"], "comid"].nunique())
n_reaches_with_t_test = int(test_period_rows.loc[test_period_rows["temp_avail_raw"], "comid"].nunique())

print(f"\n=== Measured vs. interpolated Q (test period, {SPLIT_DATE.date()}+) ===")
print(f"Q any-available (incl. interpolated fills): {n_q_any_test}")
print(f"Q measured (genuine, non-interpolated):      {n_q_measured_test}")
print(f"Q interpolated fills counted above:          {n_q_interpolated_test}")
print(f"T measured:                                   {n_t_measured_test}")
print(f"Distinct reaches with >=1 measured Q obs:     {n_reaches_with_q_test}")
print(f"Distinct reaches with >=1 measured T obs:     {n_reaches_with_t_test}")

### Append final metrics + run diagnostics to this run's results.txt (test
#### variables were already written at the top of the run, in Section 2).
with open(RESULTS_PATH, "a") as f:
    f.write("\n=== Run diagnostics ===\n")
    f.write(f"TRAINABLE_PARAMS: {N_TRAINABLE_PARAMS}\n")
    f.write(f"TRAIN_ELAPSED_SEC: {TRAIN_ELAPSED_SEC:.1f}\n")
    f.write(f"FINAL_EPOCH: {final_epoch}\n")
    f.write(f"BEST_EPOCH: {best_epoch}\n")
    f.write(f"BEST_VAL_LOSS: {best_val:.6f}\n")

    f.write("\n=== Test Metrics (original units; observed-only) ===\n")
    f.write(f"Q    -> NSE: {q_nse:.4f} | RMSE: {q_rmse:.4f} | MAE: {q_mae:.4f} | Bias: {q_bias:.4f} | n_obs={int(mask_q.sum())}\n")
    f.write(f"Temp -> NSE: {t_nse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | Bias: {t_bias:.4f} | n_obs={int(mask_t.sum())}\n")

    f.write(f"\n=== Test Metrics on common SO{COMMON_SO_MIN}-{COMMON_SO_MAX} subset ===\n")
    f.write(f"Q    -> NSE: {common_q_metrics['NSE']:.4f} | RMSE: {common_q_metrics['RMSE']:.4f} | "
            f"MAE: {common_q_metrics['MAE']:.4f} | Bias: {common_q_metrics['Bias']:.4f} | n_obs={common_q_metrics['n_obs']}\n")
    f.write(f"Temp -> NSE: {common_t_metrics['NSE']:.4f} | RMSE: {common_t_metrics['RMSE']:.4f} | "
            f"MAE: {common_t_metrics['MAE']:.4f} | Bias: {common_t_metrics['Bias']:.4f} | n_obs={common_t_metrics['n_obs']}\n")

    f.write("\n=== Test Metrics by Stream Order ===\n")
    f.write(metrics_by_streamorder.to_string(index=False))
    f.write("\n")

    f.write(f"\n=== Measured vs. interpolated Q (test period, {SPLIT_DATE.date()}+) ===\n")
    f.write(f"Q_N_ANY_TEST: {n_q_any_test}\n")
    f.write(f"Q_N_MEASURED_TEST: {n_q_measured_test}\n")
    f.write(f"Q_N_INTERPOLATED_TEST: {n_q_interpolated_test}\n")
    f.write(f"T_N_MEASURED_TEST: {n_t_measured_test}\n")
    f.write(f"N_REACHES_WITH_Q_TEST: {n_reaches_with_q_test}\n")
    f.write(f"N_REACHES_WITH_T_TEST: {n_reaches_with_t_test}\n")

## 12b. Per-reach diagnostics and seasonal breakdown
#### Pooled NSE (Section 12 above) uses one global observed mean across all
#### reaches; a model can score well there by learning *between-reach* size
#### differences (big rivers vs. small ones) without actually tracking
#### *within-reach* temporal fluctuations. Per-COMID NSE, computed against
#### each reach's own mean, isolates that — and the median across gauges
#### avoids one or two reaches dominating the picture. Joining raw (unscaled)
#### static attributes lets drainage-area/stream-order relationships with
#### per-reach skill be inspected directly, without re-deriving them from the
#### MinMax-scaled training features.
MIN_OBS_PER_COMID = 10  # too few points makes NSE/variance unstable to the point of meaningless

per_comid_rows = []
raw_statics_by_comid = (
    nodes_df_raw[["comid", "area_m2", "streamorde"]]
    .drop_duplicates("comid")
    .set_index("comid")
)

for j, comid in enumerate(all_comids):
    row = {"comid": int(comid)}
    if comid in raw_statics_by_comid.index:
        row["area_m2"] = float(raw_statics_by_comid.loc[comid, "area_m2"])
        row["streamorde"] = raw_statics_by_comid.loc[comid, "streamorde"]
    else:
        row["area_m2"] = np.nan
        row["streamorde"] = np.nan

    q_series_mask = q_masks[:, j] & np.isfinite(obs_phys[:, j, 0]) & np.isfinite(pred_phys[:, j, 0])
    n_q = int(q_series_mask.sum())
    row["Q_n_obs"] = n_q
    if n_q >= MIN_OBS_PER_COMID:
        q_obs_j = obs_phys[q_series_mask, j, 0]
        q_pred_j = pred_phys[q_series_mask, j, 0]
        row["Q_NSE"] = nse_numpy(q_pred_j, q_obs_j)
        row["Q_Bias"] = bias_numpy(q_pred_j, q_obs_j)
        row["Q_obs_var"] = float(np.var(q_obs_j))
    else:
        row["Q_NSE"] = row["Q_Bias"] = row["Q_obs_var"] = np.nan

    t_series_mask = temp_masks[:, j] & np.isfinite(obs_phys[:, j, 1]) & np.isfinite(pred_phys[:, j, 1])
    n_t = int(t_series_mask.sum())
    row["T_n_obs"] = n_t
    if n_t >= MIN_OBS_PER_COMID:
        t_obs_j = obs_phys[t_series_mask, j, 1]
        t_pred_j = pred_phys[t_series_mask, j, 1]
        row["T_NSE"] = nse_numpy(t_pred_j, t_obs_j)
        row["T_Bias"] = bias_numpy(t_pred_j, t_obs_j)
        row["T_obs_var"] = float(np.var(t_obs_j))
    else:
        row["T_NSE"] = row["T_Bias"] = row["T_obs_var"] = np.nan

    per_comid_rows.append(row)

per_comid_df = pd.DataFrame(per_comid_rows)
per_comid_df.to_csv(os.path.join(OUTPUTS_DIR, "per_comid_diagnostics.csv"), index=False)

q_median_pergauge_nse = float(per_comid_df["Q_NSE"].median(skipna=True))
t_median_pergauge_nse = float(per_comid_df["T_NSE"].median(skipna=True))
n_gauges_q_eligible = int(per_comid_df["Q_NSE"].notna().sum())
n_gauges_t_eligible = int(per_comid_df["T_NSE"].notna().sum())

### Quick directional read on drainage area vs. per-reach Q skill (full
#### scatter is in per_comid_diagnostics.csv for plotting).
_area_nse = per_comid_df[["area_m2", "Q_NSE"]].dropna()
if len(_area_nse) >= 3:
    area_nse_corr = float(np.corrcoef(np.log1p(_area_nse["area_m2"]), _area_nse["Q_NSE"])[0, 1])
else:
    area_nse_corr = np.nan

print(f"\n=== Per-reach (per-COMID) diagnostics, test period (>= {MIN_OBS_PER_COMID} obs required) ===")
print(f"Q median per-gauge NSE: {q_median_pergauge_nse:.4f}  ({n_gauges_q_eligible} eligible gauges)")
print(f"T median per-gauge NSE: {t_median_pergauge_nse:.4f}  ({n_gauges_t_eligible} eligible gauges)")
print(f"corr(log1p(area_m2), Q_NSE) across eligible gauges: {area_nse_corr:.4f}")
print(f"Full per-reach table written to: per_comid_diagnostics.csv")

### Seasonal Q/T skill (pooled across reaches, by meteorological season)
#### Flags whether poor performance is concentrated in particular seasons
#### (e.g. snowmelt/high-flow spring) rather than uniform year-round.
test_dates_arr = np.asarray(t_dates)[test_indices]
month_to_season = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                    6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
season_per_t = np.array([month_to_season[pd.Timestamp(d).month] for d in test_dates_arr])
season_flat = np.repeat(season_per_t, n_nodes)  # matches the T-major ravel order used for pred_q/true_q above

seasonal_rows = []
for season in ["DJF", "MAM", "JJA", "SON"]:
    season_mask = season_flat == season
    idx_q_season = season_mask & mask_q_f & np.isfinite(true_q) & np.isfinite(pred_q)
    idx_t_season = season_mask & mask_t_f & np.isfinite(true_t) & np.isfinite(pred_t)
    seasonal_rows.append({"season": season, "variable": "Q", **calc_metrics(true_q[idx_q_season], pred_q[idx_q_season])})
    seasonal_rows.append({"season": season, "variable": "Temp", **calc_metrics(true_t[idx_t_season], pred_t[idx_t_season])})

seasonal_metrics_df = pd.DataFrame(seasonal_rows)
seasonal_metrics_df.to_csv(os.path.join(OUTPUTS_DIR, "test_metrics_by_season.csv"), index=False)
print("\n=== Test Metrics by Season ===")
print(seasonal_metrics_df)

with open(RESULTS_PATH, "a") as f:
    f.write(f"\n=== Per-reach (per-COMID) diagnostics, test period (>= {MIN_OBS_PER_COMID} obs required) ===\n")
    f.write(f"Q_MEDIAN_PERGAUGE_NSE: {q_median_pergauge_nse:.4f}\n")
    f.write(f"T_MEDIAN_PERGAUGE_NSE: {t_median_pergauge_nse:.4f}\n")
    f.write(f"N_GAUGES_Q_ELIGIBLE: {n_gauges_q_eligible}\n")
    f.write(f"N_GAUGES_T_ELIGIBLE: {n_gauges_t_eligible}\n")
    f.write(f"AREA_LOG1P_VS_Q_NSE_CORR: {area_nse_corr:.4f}\n")
    f.write("(full per-reach table: per_comid_diagnostics.csv)\n")

    f.write("\n=== Test Metrics by Season ===\n")
    f.write(seasonal_metrics_df.to_string(index=False))
    f.write("\n(full table: test_metrics_by_season.csv)\n")

## 13. Diagnostic plots
#### Generates single-COMID hydrographs, observed-vs-predicted scatterplots,
#### and observed-distribution histograms.  All figures are written directly to
#### OUTPUTS_DIR so the full script can run uninterrupted without manual figure
#### closing.  The histogram view applies the same preprocessing rules used
#### during training so what you inspect matches what the model saw.
#### The default target COMID may not fall inside every stream-order band
#### tested here, and — even when it is in-domain — picking an arbitrary
#### (sorted-first) COMID almost always lands on an ungauged reach, since
#### only a handful of COMIDs per stream order have real observations (see
#### gauge_inventory.py). That silently produced an empty "Observed" line on
#### every hydrograph. Fall back to a COMID with real test-period
#### observations instead, preferring one gauged for both Q and T.
q_gauged_by_node = q_masks.any(axis=0)     # [N] bool: any test-period Q obs
t_gauged_by_node = temp_masks.any(axis=0)  # [N] bool: any test-period T obs

requested_idx = comid_to_idx.get(TARGET_COMID_FOR_PLOTS)
requested_is_gauged = requested_idx is not None and (q_gauged_by_node[requested_idx] or t_gauged_by_node[requested_idx])

if not requested_is_gauged:
    both_idx  = np.where(q_gauged_by_node & t_gauged_by_node)[0]
    q_idx_opt = np.where(q_gauged_by_node)[0]
    t_idx_opt = np.where(t_gauged_by_node)[0]

    if len(both_idx) > 0:
        fallback_idx = int(both_idx[0])
    elif len(q_idx_opt) > 0:
        fallback_idx = int(q_idx_opt[0])
    elif len(t_idx_opt) > 0:
        fallback_idx = int(t_idx_opt[0])
    else:
        fallback_idx = 0  # no gauged reach at all in this domain's test period

    fallback_comid = int(all_comids[fallback_idx])
    reason = "not in this run's domain" if requested_idx is None else "has no test-period observations"
    print(f"\n[WARNING] COMID {TARGET_COMID_FOR_PLOTS} {reason} "
          f"(streamorde {STREAMORDER_MIN}-{STREAMORDER_MAX}); "
          f"falling back to gauged COMID {fallback_comid} for single-reach plots.")
    with open(RESULTS_PATH, "a") as f:
        f.write(f"\n[WARNING] TARGET_COMID_FOR_PLOTS {TARGET_COMID_FOR_PLOTS} {reason}; "
                f"used fallback COMID {fallback_comid} for single-reach plots.\n")
    TARGET_COMID_FOR_PLOTS = fallback_comid

catch_idx = comid_to_idx[TARGET_COMID_FOR_PLOTS]

pred_q_series = pred_phys[:, catch_idx, 0]
true_q_series = obs_phys[:,  catch_idx, 0]
pred_t_series = pred_phys[:, catch_idx, 1]
true_t_series = obs_phys[:,  catch_idx, 1]

mask_q_series = q_masks[:, catch_idx]
mask_t_series = temp_masks[:, catch_idx]

plt.figure(figsize=(12, 4))
plt.plot(pred_q_series, label="Predicted Q", alpha=0.85, linewidth=1.8)
if mask_q_series.any():
    plt.plot(np.where(mask_q_series, true_q_series, np.nan), label="Observed Q (masked)", linewidth=2)
plt.legend()
plt.title(f"Hydrograph (Test Period) — COMID {TARGET_COMID_FOR_PLOTS}")
plt.xlabel("Time steps")
plt.ylabel("Discharge (m³/s)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, f"hydrograph_Q_{TARGET_COMID_FOR_PLOTS}.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(pred_t_series, label="Predicted Temp", alpha=0.85, linewidth=1.8)
if mask_t_series.any():
    plt.plot(np.where(mask_t_series, true_t_series, np.nan), label="Observed Temp (masked)", linewidth=2)
plt.legend()
plt.title(f"Water Temperature (Test Period) — COMID {TARGET_COMID_FOR_PLOTS}")
plt.xlabel("Time steps")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, f"hydrograph_T_{TARGET_COMID_FOR_PLOTS}.png"), dpi=150, bbox_inches="tight")
plt.close()

valid_q = mask_q & np.isfinite(true_q) & np.isfinite(pred_q)
plt.figure(figsize=(5, 5))
plt.scatter(true_q[valid_q], pred_q[valid_q], alpha=0.3, s=10)
if valid_q.any():
    max_q = max(true_q[valid_q].max(), pred_q[valid_q].max())
    min_q = min(true_q[valid_q].min(), pred_q[valid_q].min())
    plt.plot([min_q, max_q], [min_q, max_q], linestyle="--")
plt.xlabel("Observed Q (m³/s)")
plt.ylabel("Predicted Q (m³/s)")
plt.title("Discharge: Observed vs Predicted (Observed-only)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "scatter_Q.png"), dpi=150, bbox_inches="tight")
plt.close()

valid_t = mask_t & np.isfinite(true_t) & np.isfinite(pred_t)
plt.figure(figsize=(5, 5))
plt.scatter(true_t[valid_t], pred_t[valid_t], alpha=0.3, s=10)
if valid_t.any():
    max_t = max(true_t[valid_t].max(), pred_t[valid_t].max())
    min_t = min(true_t[valid_t].min(), pred_t[valid_t].min())
    plt.plot([min_t, max_t], [min_t, max_t], linestyle="--")
plt.xlabel("Observed Temp (°C)")
plt.ylabel("Predicted Temp (°C)")
plt.title("Water Temperature: Observed vs Predicted (Observed-only)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "scatter_T.png"), dpi=150, bbox_inches="tight")
plt.close()

### Observed-distribution histograms (log-y)
def plot_obs_histograms_logy(nodes_df_in, q_col="q_m3s_obs", t_col="water_temp_C",
                             bins_q=250, bins_t=200, title_suffix="(all observations)",
                             save_dir=None):
    q = pd.to_numeric(nodes_df_in[q_col], errors="coerce").to_numpy()
    q = q[np.isfinite(q)]
    plt.figure(figsize=(9, 4))
    plt.hist(q, bins=bins_q)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.title(f"Observed Discharge Histogram — log y {title_suffix}")
    plt.xlabel("Observed Q (m³/s)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_Q.png"), dpi=150, bbox_inches="tight")
    plt.close()

    t = pd.to_numeric(nodes_df_in[t_col], errors="coerce").to_numpy()
    t = t[np.isfinite(t)]
    plt.figure(figsize=(9, 4))
    plt.hist(t, bins=bins_t)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.title(f"Observed Water Temperature Histogram — log y {title_suffix}")
    plt.xlabel("Observed Temp (°C)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_T.png"), dpi=150, bbox_inches="tight")
    plt.close()

### nodes_df_raw (Section 4) is already filtered to this run's stream-order
#### band and already has the same Q repair / temperature QC applied, so the
#### histogram view can reuse it directly instead of re-reading the ~12GB
#### NODES_PATH file and redoing that work a second time per run.
plot_obs_histograms_logy(
    nodes_df_raw,
    q_col=Q_COL,
    t_col=T_COL,
    bins_q=250,
    bins_t=200,
    title_suffix=f"(stream order {STREAMORDER_MIN}-{STREAMORDER_MAX})",
    save_dir=OUTPUTS_DIR,
)

## 14. Save the model
#### Model weights, both scalers, and the feature list (order-critical for
#### inference) are written to SAVE_DIR — but only when --save-model is
#### passed. While sweeping experiments, results.txt + the plots in
#### OUTPUTS_DIR are the record; weights are only worth keeping once a
#### config is settled on and GraphSHAP.py needs them to run.
if SAVE_MODEL:
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_FILENAME))
    joblib.dump(scaler_X, os.path.join(SAVE_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(SAVE_DIR, "scaler_y.pkl"))
    joblib.dump(FEATURES, FEATURES_PATH)
    print(f"\nSaved model/scalers/features to:\n  {SAVE_DIR}\n  model: {MODEL_FILENAME}\n  features: {FEATURES_PATH}")
else:
    print("\n--save-model not set: skipping weight/scaler/feature save for this run.")

print(f"Graph smoothness: {USE_GRAPH_SMOOTHNESS} (lambda={LAMBDA_SMOOTH})")
print("Output head: sigmoid on BOTH Q and Temp (scaled-space bounded)")

## 15. Graph-SHAP interpretation
#### Graph-SHAP is intentionally NOT run here. Explaining a trained graph
#### model (feature / temporal / node×feature / structural Shapley) is a
#### separate pass from training, and is moved out to GraphSHAP.py so that
#### training runs aren't slowed down by SHAP's Monte Carlo permutations.
#### Run SRC/GraphSHAP.py after this script finishes, once a model is saved
#### to SAVE_DIR, to interpret it.
