#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DDNN_Model.py  (the model file)
# STEP 8 of the pipeline (last script; runs AFTER Q_Processing_python.py).
#
# Purpose : train and evaluate a multitask Graph LSTM-GAT that predicts daily discharge (Q)
#           and water temperature (T) on the routed NHDPlus network, then explain it with
#           Graph-SHAP (feature, lag, node x feature and structural levels).
# Inputs  : hydro_temp_data_trainval_2013plus.parquet   (NODES_PATH)
#           edge_list.parquet                           (EDGES_PATH)
# Outputs : trained model weights, scalers and features.pkl in SAVE_DIR;
#           diagnostic figures and Graph-SHAP results in OUTPUTS_DIR
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
from torch_geometric.utils import k_hop_subgraph

### Sklearn
from sklearn.preprocessing import MinMaxScaler
import shap  # optional — retained for potential kernel-SHAP comparisons


## 2. Configuration

### Input data paths
#### Node table: daily COMID×date records with met drivers, discharge, and temperature.
#### Edge table: NHDPlus fromnode/tonode routing used to build the directed graph.
#### NOTE: NODES_PATH, EDGES_PATH, SAVE_DIR and OUTPUTS_DIR are hard-coded absolute paths; point them at your own project folders
NODES_PATH = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/hydro_temp_data_trainval_2013plus.parquet"
EDGES_PATH = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/edge_list.parquet"

### Temporal split & sequence
#### Temporal split: all windows whose target date falls before SPLIT_DATE go to
#### training; windows on or after go to the held-out test set. A 7-day lookback
#### gives the model enough recent memory without overextending sequence length.
SPLIT_DATE   = pd.Timestamp("2025-01-01")
LOOKBACK     = 7                            # days of antecedent input per window

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
EPOCHS       = 200
LR           = 1e-4
BATCH_SIZE   = 1                            # one full graph window per step
PATIENCE     = 10                           # early-stopping patience (epochs)

### Modeling domain
#### Restricting to stream order ≥4 or 5 focuses the model on larger, better-observed
#### reaches to pair with remote sensing derived chl-a predictions
STREAMORDER_MIN = 5

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
SEASONAL_FEATURES = ["DOY_sin", "DOY_cos"]

### Targets
Q_COL    = "q_m3s_obs"
T_COL    = "water_temp_C"
TARGETS  = [Q_COL, T_COL]

### Task weights
#### Fixed task weights: discharge is the primary target (0.7); temperature is
#### secondary (0.3) but still requires the shared representation to learn it.
TASK_WEIGHTS = torch.tensor([0.7, 0.3], dtype=torch.float32)

### High-flow upweighting
#### Ramps residual weight up to Q_HIGH_MAX_MULT for observations above the
#### threshold, helping the model better represent hydrologically important peaks.
Q_HIGH_THRESH_M3S = 100.0
Q_HIGH_MAX_MULT   = 5.0
Q_HIGH_POWER      = 1.0

### Graph smoothness regularization
#### Optional L2 penalty on predicted Q differences along directed edges.
#### Discourages physically implausible jumps in discharge across connected reaches.
USE_GRAPH_SMOOTHNESS = True
LAMBDA_SMOOTH        = 1e-3                 # tune between 1e-4 and 1e-2

### Output paths
TARGET_COMID_FOR_PLOTS = 14787569           # COMID used for single-reach diagnostics

SAVE_DIR    = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/models"
OUTPUTS_DIR = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/outputs"
FEATURES_PATH       = os.path.join(SAVE_DIR, "features.pkl")
EDGE_SUSPICIOUS_OUT = os.path.join(SAVE_DIR, "edge_suspicious_checks.csv")

### Graph coarsening
#### When enabled, the full NHDPlus network is collapsed so edges connect only
#### modeled COMIDs; intermediate unobserved reaches are skipped by walking
#### downstream until the nearest observed connection is found.
USE_COARSENED_GRAPH = True
MAX_HOPS_COARSEN    = 5000
COARSE_MODE         = "nearest"

MODEL_FILENAME = "graph_lstm_gat_multitask_q_t_maskedloss_sigmoid_head.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
):
    """
    Repair negative discharge values by COMID time interpolation.

    Steps:
      1) Coerce q to numeric
      2) Mark q < q_floor as NaN
      3) Interpolate linearly within each COMID over time (inside only)
      4) Clip Q to >= 0 for numerical safety

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

    # interpolate per COMID in time order; no endpoint extrapolation
    out[q_col] = (
        out.groupby(comid_col, group_keys=False)[q_col]
           .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
    )

    # safety clip
    out[q_col] = out[q_col].clip(lower=0.0)

    summary = {
        "q_negatives_marked_nan_before_interp": n_neg_before,
        "q_nan_after_interp": int(out[q_col].isna().sum()),
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

nodes_df_raw = pd.read_parquet(NODES_PATH)
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
#### treated as true supervision signal.
nodes_df_raw["temp_avail_raw"] = ~nodes_df_raw[T_COL].isna()
nodes_df_raw["q_avail_raw"]    = ~nodes_df_raw[Q_COL].isna()

### Modeling domain: stream order filter
#### Applied to both train and test to keep the domain consistent.
#### Restricting to order ≥ STREAMORDER_MIN focuses the model on larger,
#### better-observed reaches during initial development.
so_filter = pd.to_numeric(nodes_df_raw["streamorde"], errors="coerce")

keep_rows = so_filter >= STREAMORDER_MIN

n_before = len(nodes_df_raw)
nodes_df_raw = nodes_df_raw.loc[keep_rows].copy()
nodes_df_raw = nodes_df_raw.sort_values(["date", "comid"]).reset_index(drop=True)
n_after = len(nodes_df_raw)

print("\n=== Applied stream order filter ===")
print(f"Rule: keep streamorde >= {STREAMORDER_MIN}")
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
os.makedirs(SAVE_DIR, exist_ok=True)

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

joblib.dump(FEATURES, FEATURES_PATH)
print(f"Using explicit FEATURES (n={len(FEATURES)}), saved to: {FEATURES_PATH}")
print(FEATURES)

keep_cols = ["comid", "date"] + FEATURES + TARGETS + ["temp_avail_raw", "q_avail_raw"]
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

train_mask_rows = nodes_df["date"] < SPLIT_DATE
if train_mask_rows.sum() == 0:
    raise ValueError("No training rows found before SPLIT_DATE. Check SPLIT_DATE or data date range.")

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
q_thresh_scaled = float((Q_HIGH_THRESH_M3S - q_min) / q_range)
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

for d in dates:
    t = date_to_idx[d]
    sub = nodes_df[nodes_df["date"] == d]
    idxs = sub["comid"].map(comid_to_idx).to_numpy()

    X_TNF[t, idxs, :] = sub[FEATURES].to_numpy(dtype=np.float32)
    Y_TN2[t, idxs, 0] = sub[[Q_COL]].to_numpy(dtype=np.float32).ravel()
    Y_TN2[t, idxs, 1] = sub[[T_COL]].to_numpy(dtype=np.float32).ravel()

    # masks come from raw availability columns (not scaled)
    Tmask_TN[t, idxs] = sub["temp_avail_raw"].to_numpy(dtype=bool)
    Qmask_TN[t, idxs] = sub["q_avail_raw"].to_numpy(dtype=bool)

## 8. Train–test datasets
#### Each sample is a 7-day input window across all nodes ending at time t,
#### paired with targets for that final day.  The split is strictly temporal:
#### windows whose target date falls before SPLIT_DATE go to training; all
#### others go to the held-out test set.  Because each batch spans the full
#### graph, BATCH_SIZE = 1 (one window = one graph time step).
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
train_indices = [i for i, d in enumerate(t_dates) if d < SPLIT_DATE]
test_indices  = [i for i, d in enumerate(t_dates) if d >= SPLIT_DATE]

assert len(train_indices) > 0, "No training samples before split date."
assert len(test_indices)  > 0, "No testing samples on/after split date."

train_ds = Subset(dataset, train_indices)
test_ds  = Subset(dataset,  test_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining on {len(train_ds)} time steps (< {SPLIT_DATE.date()}); "
      f"testing on {len(test_ds)} time steps (≥ {SPLIT_DATE.date()}).")

## 9. Model architecture (Graph LSTM + GAT multitask)
#### For each node, the 7-day feature sequence is passed through an LSTM to
#### encode short-term temporal dynamics (Kratzert et al., 2018).  The final
#### hidden state becomes that node's embedding for the current day.  Two
#### stacked GAT layers then propagate those embeddings downstream through the
#### directed river graph (Veličković et al., 2018; Jia et al., 2021), so each
#### reach can receive information from its upstream neighbors.  A linear head
#### maps the updated embeddings to two outputs (Q, T), and a sigmoid activation
#### bounds both predictions to [0, 1] in scaled space — added after observing
#### exploding predictions during early model development.
class GraphLSTM_MTL(nn.Module):
    def __init__(self, in_dim, lstm_hidden, lstm_layers, lstm_dropout,
                 gnn_hidden, gnn_dropout, heads=4, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            batch_first=True
        )
        self.drop_lstm = nn.Dropout(lstm_dropout)
        self.gat1 = GATConv(lstm_hidden, gnn_hidden // heads, heads=heads, dropout=gnn_dropout)
        self.gat2 = GATConv(gnn_hidden, gnn_hidden // heads, heads=heads, dropout=gnn_dropout)
        self.drop_gnn = nn.Dropout(gnn_dropout)
        self.head = nn.Linear(gnn_hidden, out_dim)

    def forward(self, x_win, edge_index, return_logits=False):
        lstm_out, _ = self.lstm(x_win)                   # [N, lb, H_LSTM]
        node_embed = self.drop_lstm(lstm_out[:, -1, :]) # [N, H_LSTM]

        h = F.elu(self.gat1(node_embed, edge_index))    # [N, H_GNN]
        h = self.drop_gnn(h)
        h = F.elu(self.gat2(h, edge_index))             # [N, H_GNN]
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
    gnn_hidden=H_GNN, gnn_dropout=GNN_DROPOUT, heads=GAT_HEADS, out_dim=2
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
#### is run on the held-out test windows after every epoch; early stopping fires
#### if validation loss does not improve within PATIENCE epochs.  The best model
#### state is retained and reloaded at the end of training.
best_val = float("inf")
best_state = None
epochs_no_improve = 0

print(f"\nTraining Graph-LSTM+GAT (multitask Q+T) on {len(train_ds)} time steps; "
      f"validating on {len(test_ds)} time steps.")
print("Output head constraint: sigmoid on BOTH Q and Temp (scaled-space [0,1])")

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
        for x_win, y_t, m_t, m_q in test_loader:
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

    scheduler.step(va_mean)

    if va_mean + 1e-6 < best_val:
        best_val = va_mean
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

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
else:
    q_nse = q_rmse = q_mae = np.nan

if mask_t.any():
    t_nse  = nse_numpy(pred_t[mask_t], true_t[mask_t])
    t_rmse = rmse_numpy(pred_t[mask_t], true_t[mask_t])
    t_mae  = mae_numpy(pred_t[mask_t], true_t[mask_t])
else:
    t_nse = t_rmse = t_mae = np.nan

print("\n=== Test Metrics (original units; observed-only) ===")
print(f"Q    -> NSE: {q_nse:.4f} | RMSE: {q_rmse:.4f} | MAE: {q_mae:.4f} | n_obs={int(mask_q.sum())}")
print(f"Temp -> NSE: {t_nse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | n_obs={int(mask_t.sum())}")

### Metrics by stream order
#### Stratifying by stream order reveals whether the model is stronger on
#### larger reaches (higher order) versus smaller tributaries (lower order),
#### which informs where the domain should be extended or constrained.
def calc_metrics(obs, pred):
    if len(obs) == 0:
        return {"n_obs": 0, "NSE": np.nan, "RMSE": np.nan, "MAE": np.nan}
    return {
        "n_obs": len(obs),
        "NSE": nse_numpy(pred, obs),
        "RMSE": rmse_numpy(pred, obs),
        "MAE": mae_numpy(pred, obs)
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
    os.path.join(SAVE_DIR, "test_metrics_by_streamorder.csv"),
    index=False
)

print("\n=== Physical plausibility diagnostics (test predictions, original units) ===")
print(f"Pred Q < 0 count: {int((pred_q < 0).sum())} / {pred_q.size}")
print(f"Pred Temp < 0 count: {int((pred_t < 0).sum())} / {pred_t.size}")
print(f"Pred Temp > 40 count: {int((pred_t > 40).sum())} / {pred_t.size}")

## 13. Diagnostic plots
#### Generates single-COMID hydrographs, observed-vs-predicted scatterplots,
#### and observed-distribution histograms.  All figures are written directly to
#### OUTPUTS_DIR so the full script can run uninterrupted without manual figure
#### closing.  The histogram view applies the same preprocessing rules used
#### during training so what you inspect matches what the model saw.
if TARGET_COMID_FOR_PLOTS not in comid_to_idx:
    raise ValueError(f"COMID {TARGET_COMID_FOR_PLOTS} not found in dataset.")

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

nodes_raw = pd.read_parquet(NODES_PATH)
nodes_raw["date"] = pd.to_datetime(nodes_raw["date"])
nodes_raw = replace_sentinel_with_nan(nodes_raw, [Q_COL, T_COL])

### Match modeling-domain filter
nodes_raw["streamorde"] = pd.to_numeric(nodes_raw["streamorde"], errors="coerce")
nodes_raw = nodes_raw.loc[nodes_raw["streamorde"] >= STREAMORDER_MIN].copy()

### Apply same Q repair to histogram view so what you inspect matches training preprocessing
nodes_raw, _ = interpolate_negative_q_by_comid(
    nodes_raw, q_col=Q_COL, date_col="date", comid_col="comid", q_floor=0.0
)

plot_obs_histograms_logy(
    nodes_raw,
    q_col=Q_COL,
    t_col=T_COL,
    bins_q=250,
    bins_t=200,
    title_suffix=f"(stream order >= {STREAMORDER_MIN})",
    save_dir=OUTPUTS_DIR,
)

## 14. Save the model
#### Model weights, both scalers, and the feature list (order-critical for
#### inference) are written to SAVE_DIR.  Reloading all three is necessary to
#### reproduce predictions outside this script.
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_FILENAME))
joblib.dump(scaler_X, os.path.join(SAVE_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(SAVE_DIR, "scaler_y.pkl"))
joblib.dump(FEATURES, FEATURES_PATH)

print(f"\nSaved model/scalers/features to:\n  {SAVE_DIR}\n  model: {MODEL_FILENAME}\n  features: {FEATURES_PATH}")
print(f"Graph smoothness: {USE_GRAPH_SMOOTHNESS} (lambda={LAMBDA_SMOOTH})")
print("Output head: sigmoid on BOTH Q and Temp (scaled-space bounded)")


## 15. Graph-SHAP interpretation
#### Standard SHAP is insufficient for a graph model because once message
#### passing is used, the prediction at a node is influenced by upstream
#### neighbors — not just that node's own inputs.  This block implements a
#### three-level Monte Carlo Shapley framework that respects graph structure:
#
####   Level 1  Feature Graph-SHAP   — MC Shapley over feature blocks at the
####            target node (all lags of pr, all lags of tmax, etc.)
####   Level 2  Temporal Graph-SHAP  — MC Shapley over lag blocks
####            (all features at t-6, t-5, … t)
####   Level 3  Node×Feature Heatmap — MC Shapley over every
####            (upstream_node, feature) pair in the k-hop subgraph
####   Structural                    — MC Shapley over incoming edges,
####            capturing synergy / redundancy that leave-one-out misses
#
#### Key design choices
####  • Players are grouped blocks rather than raw lb×F scalar cells,
####    reducing dimensionality and keeping results interpretable.
####  • Absent players are replaced with randomly drawn seasonal background
####    windows (not a fixed mean), making the masking more realistic.
####  • All explanations are restricted to a k-hop upstream subgraph so the
####    explanation respects the local routed neighborhood the model uses.
####  • Returns signed mean + 95 % CI so direction and uncertainty are visible.
####  • compute_edge_ablation_importance is retained as a fast sanity check;
####    compute_structural_graphshap_mc is the true Shapley version.

### 15.1 Tuneable knobs
GRAPHSHAP_N_HOPS         = 2      # upstream subgraph depth
GRAPHSHAP_N_BACKGROUND   = 40     # distinct windows in masker pool
GRAPHSHAP_N_EXPLAIN      = 25     # windows to average Shapley over
GRAPHSHAP_N_MC_PERMS     = 120    # permutations per explain window
GRAPHSHAP_N_MC_STRUCT    = 100    # permutations for structural SHAP
GRAPHSHAP_N_MC_NF        = 60     # permutations for node×feature
GRAPHSHAP_MAX_NF_PLAYERS = 100    # skip node×feature if larger
GRAPHSHAP_SEASON_DAYS    = 45     # ±days for seasonal background


### 15.2 Local upstream subgraph extraction
#### Extract a k-hop upstream neighborhood around the target node.  Because the
#### graph is directed upstream → downstream, only nodes that can influence the
#### target through routing are included — downstream nodes are excluded.  This
#### keeps the explanation local while preserving the relevant routed structure.

def extract_upstream_subgraph(edge_index, target_node_idx, n_hops=2):
    """
    k-hop upstream subgraph, flow = source_to_target (upstream → target).

    Returns
    -------
    sub_nodes_global : LongTensor  [Ns]  global node indices
    edge_index_sub   : LongTensor  [2, Es]  relabeled to subgraph
    target_sub_idx   : int         position of target in sub_nodes_global
    """
    sub_nodes, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx      = target_node_idx,
        num_hops      = n_hops,
        edge_index    = edge_index,
        flow          = "source_to_target",
        relabel_nodes = True,
    )
    target_sub_idx = int(mapping.item()) if mapping.numel() == 1 \
        else int((sub_nodes == target_node_idx).nonzero(as_tuple=True)[0].item())
    return sub_nodes, edge_index_sub, target_sub_idx


### 15.3 Player (block) definitions
#### Players are grouped blocks rather than raw lb×F scalar cells, which reduces
#### the coalition space and makes results interpretable at the feature, lag, or
#### node×feature level.  Each player is a 3-tuple:
####   (name, global_node_idx, [flat_indices])
#### where flat_index = lag * n_features + feat_idx  (row-major over [lb, F])

def make_feature_blocks(target_node_idx, lookback, features):
    """One player per feature — pools all lags of that feature."""
    F = len(features)
    return [
        (f_name, target_node_idx,
         [lag * F + f_idx for lag in range(lookback)])
        for f_idx, f_name in enumerate(features)
    ]

### Lag blocks (one player per time lag)
def make_lag_blocks(target_node_idx, lookback, features):
    """One player per lag — pools all features at that time step."""
    F = len(features)
    return [
        (f"t-{lookback-1-lag}" if lag < lookback - 1 else "t",
         target_node_idx,
         list(range(lag * F, (lag + 1) * F)))
        for lag in range(lookback)
    ]

### Node x feature blocks (one player per node-feature pair)
def make_node_feature_blocks(sub_nodes_global, features, target_node_idx, lookback):
    """One player per (node, feature) — all lags of feature f at node n."""
    F = len(features)
    blocks = []
    for n_g in sub_nodes_global.tolist():
        label = "target" if n_g == target_node_idx else f"up_{n_g}"
        for f_idx, f_name in enumerate(features):
            blocks.append((
                f"{label}/{f_name}",
                int(n_g),
                [lag * F + f_idx for lag in range(lookback)]
            ))
    return blocks


### 15.4 Seasonal background pool
#### Background windows are drawn from seasonally similar periods (±season_window_days
#### of day-of-year) rather than random dates, making the absent-player baseline
#### more realistic than a fixed mean or a fully random sample.

def build_background_pool(X_TNF, test_indices, dates, sub_nodes_global,
                          lookback, n_samples, target_date=None,
                          season_window_days=45, seed=42):
    """
    Returns bg_pool: {global_node_idx: list of [lb*F] float32 arrays}
    Drawn from seasonally similar test windows (falls back to random).
    """
    sub_set = sub_nodes_global.tolist()

    # Seasonal selection
    if target_date is not None:
        td  = pd.Timestamp(target_date)
        tdoy = td.day_of_year
        pool_idx = [
            ti for ti in test_indices
            if abs(pd.Timestamp(dates[ti + lookback]).day_of_year - tdoy)
               <= season_window_days
        ]
        if len(pool_idx) < max(10, n_samples // 2):
            pool_idx = list(test_indices)
    else:
        pool_idx = list(test_indices)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool_idx,
                        size=min(n_samples, len(pool_idx)),
                        replace=False)

    bg_pool = {int(ng): [] for ng in sub_set}
    for ti in chosen:
        t0 = ti + 1
        t1 = ti + lookback + 1
        x_np = X_TNF[t0:t1].transpose(1, 0, 2).astype(np.float32)  # [N, lb, F]
        for ng in sub_set:
            bg_pool[ng].append(x_np[ng].reshape(-1))  # [lb*F]

    return bg_pool, list(chosen)


### 15.5 Graph-aware coalition reconstructor
#### Absent players → background values; present players → actual explain values.
#### Masking is graph-aware: all subgraph nodes start from background, then only
#### the coalition members have their explain-window values restored.  This keeps
#### the explanation grounded in the local routed neighborhood.

#### NOTE: this function is defined a second time in the Graph-SHAP utilities below; the later definition is the one in effect
def reconstruct_from_coalition(
    coalition_mask,      # bool [n_players]
    player_blocks,       # list of (name, node_g, flat_idxs)
    x_explain_full,      # np.ndarray [N_global, lb, F]  explain window
    bg_by_node,          # {node_g: np.ndarray [lb*F]} one drawn background
    sub_nodes_global,    # node indices that are "in play"
    lookback, n_features, device
):
    """
    Start from explain data for ALL nodes.
    For nodes in the subgraph, replace with background (absent = background).
    For players in the coalition (mask=True), restore their explain values.
    """
    x_win = torch.FloatTensor(x_explain_full).to(device)   # [N, lb, F]

    # Step 1: all subgraph nodes → background
    for ng in sub_nodes_global.tolist():
        x_win[ng] = torch.FloatTensor(
            bg_by_node[ng].reshape(lookback, n_features)
        ).to(device)

    # Step 2: coalition players → restore explain values
    F = n_features
    for p_idx, (_, node_g, flat_idxs) in enumerate(player_blocks):
        if coalition_mask[p_idx]:
            ex_flat = x_explain_full[node_g].reshape(-1)  # [lb*F]
            for fi in flat_idxs:
                lag_i  = fi // F
                feat_i = fi %  F
                x_win[node_g, lag_i, feat_i] = float(ex_flat[fi])

    return x_win   # [N, lb, F]  on device


### 15.6 Monte Carlo Shapley core
#### Shared estimator used by all three feature levels.  For each explain window,
#### random player orderings are sampled and the marginal contribution of each
#### player is accumulated as it enters the coalition.  Repeating across windows
#### yields a signed mean with a 95 % CI so both magnitude and direction are
#### visible in the final output.

def mc_shapley_over_blocks(
    model, edge_index,
    X_TNF, test_indices, dates,
    player_blocks,
    sub_nodes_global, target_node_idx,
    lookback, n_features,
    n_background, n_explain, n_mc_perms,
    target_output, device,
    background_indices=None,
    season_window_days=45, seed=0,
):
    """
    Generic MC Shapley estimator over any grouped player set.

    background_indices : list of dataset indices used to build the background
        reference pool.  Should be the TRAINING indices so the baseline
        represents the model's learned distribution, not the same test windows
        being explained.  If None, falls back to test_indices (not recommended).

    Returns
    -------
    shap_mean  : [n_players]  mean signed Shapley (in scaled prediction space)
    shap_std   : [n_players]  std error across explain windows
    shap_ci95  : [n_players]  95 % CI half-width (1.96 × std)
    """
    model.eval()
    n_players  = len(player_blocks)
    rng        = np.random.default_rng(seed)
    F          = n_features

    # Explain windows drawn from test period
    exp_sample = rng.choice(len(test_indices),
                            size=min(n_explain, len(test_indices)),
                            replace=False)
    exp_tis = [test_indices[int(i)] for i in exp_sample]

    # Background pool drawn from training period (separate from explain windows)
    bg_source = background_indices if background_indices is not None else test_indices
    bg_pool_all, _ = build_background_pool(
        X_TNF, bg_source, dates, sub_nodes_global,
        lookback, n_background,
        target_date=None,
        season_window_days=season_window_days,
        seed=seed + 1,
    )
    n_bg_samples = min(len(v) for v in bg_pool_all.values() if v)

    shap_per_window = np.zeros((n_explain, n_players), dtype=np.float64)

    for ei, ti in enumerate(exp_tis):
        t0 = ti + 1
        t1 = ti + lookback + 1
        x_explain_full = X_TNF[t0:t1].transpose(1, 0, 2).astype(np.float32)  # [N, lb, F]

        sv = np.zeros(n_players, dtype=np.float64)

        for _ in range(n_mc_perms):
            perm         = rng.permutation(n_players)
            bg_draw_idx  = int(rng.integers(0, n_bg_samples))
            bg_by_node   = {ng: bg_pool_all[ng][bg_draw_idx]
                            for ng in sub_nodes_global.tolist()}

            coalition = np.zeros(n_players, dtype=bool)
            x_prev    = reconstruct_from_coalition(
                coalition, player_blocks, x_explain_full,
                bg_by_node, sub_nodes_global, lookback, F, device
            )
            with torch.no_grad():
                prev_pred = model(x_prev, edge_index, return_logits=True)[target_node_idx, target_output].item()

            for pos in perm:
                coalition[pos] = True
                x_new = reconstruct_from_coalition(
                    coalition, player_blocks, x_explain_full,
                    bg_by_node, sub_nodes_global, lookback, F, device
                )
                with torch.no_grad():
                    new_pred = model(x_new, edge_index, return_logits=True)[target_node_idx, target_output].item()

                sv[pos] += (new_pred - prev_pred)
                prev_pred = new_pred

        sv /= n_mc_perms
        shap_per_window[ei] = sv

    shap_mean = np.mean(shap_per_window, axis=0)
    shap_std  = np.std(shap_per_window, axis=0) / np.sqrt(max(n_explain, 1))
    shap_ci95 = 1.96 * shap_std

    return shap_mean, shap_std, shap_ci95


### 15.7 Structural Graph-SHAP — MC Shapley over incoming edges
#### Separates the role of graph structure from node features by computing
#### true Shapley values over the set of incoming edges to the target node.
#### Captures synergy and redundancy among upstream connections that a simple
#### leave-one-edge-out ablation cannot detect.

def compute_structural_graphshap_mc(
    model, edge_index,
    X_TNF, test_indices,
    target_node_idx,
    lookback,
    n_explain=20, n_mc_perms=100,
    target_output=0, device="cpu", seed=10,
):
    """
    True MC Shapley over incoming edges to target_node_idx.
    Captures edge synergy / redundancy that leave-one-out misses.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    incoming_ids = torch.where(edge_index[1] == target_node_idx)[0].tolist()
    if not incoming_ids:
        print(f"[Structural SHAP] No incoming edges to node {target_node_idx}.")
        return np.array([]), np.array([]), np.array([]), []

    n_players    = len(incoming_ids)
    edge_players = [(int(edge_index[0, ei]), int(edge_index[1, ei]))
                    for ei in incoming_ids]

    cand_mask_base = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)
    for ei in incoming_ids:
        cand_mask_base[ei] = True

    exp_tis = [test_indices[int(i)]
               for i in rng.choice(len(test_indices),
                                   size=min(n_explain, len(test_indices)),
                                   replace=False)]

    shap_per_window = np.zeros((n_explain, n_players), dtype=np.float64)

    for ei_w, ti in enumerate(exp_tis):
        t0 = ti + 1
        t1 = ti + lookback + 1
        x_win = torch.FloatTensor(
            X_TNF[t0:t1].transpose(1, 0, 2).astype(np.float32)
        ).to(device)  # [N, lb, F]

        sv = np.zeros(n_players, dtype=np.float64)

        for _ in range(n_mc_perms):
            perm   = rng.permutation(n_players)
            active = []

            keep = ~cand_mask_base.clone()
            with torch.no_grad():
                prev_pred = model(x_win, edge_index[:, keep], return_logits=True)[target_node_idx, target_output].item()

            for pos in perm:
                active.append(pos)
                keep = ~cand_mask_base.clone()
                for ap in active:
                    keep[incoming_ids[ap]] = True

                with torch.no_grad():
                    new_pred = model(x_win, edge_index[:, keep], return_logits=True)[target_node_idx, target_output].item()

                sv[pos] += (new_pred - prev_pred)
                prev_pred = new_pred

        sv /= n_mc_perms
        shap_per_window[ei_w] = sv

    shap_mean = np.mean(shap_per_window, axis=0)
    shap_std  = np.std(shap_per_window, axis=0) / np.sqrt(max(n_explain, 1))
    shap_ci95 = 1.96 * shap_std

    return shap_mean, shap_std, shap_ci95, edge_players


### 15.8 Edge ablation — fast structural sanity check
#### Single-edge leave-one-out ablation: much faster than full Shapley but
#### cannot capture edge interactions.  Use this to quickly verify that the
#### broader pattern of structural Shapley results is consistent, or when the
#### number of incoming edges makes MC Shapley too expensive.

def compute_edge_ablation_importance(
    model, edge_index, x_win_sample,
    target_node_idx, target_output=0, device="cpu"
):
    """
    Fast single-edge leave-one-out ablation (NOT full Shapley).
    Use as a quick sanity check or when n_incoming_edges > 30.
    """
    model.eval()
    x = x_win_sample.to(device)
    with torch.no_grad():
        baseline = model(x, edge_index, return_logits=True)[target_node_idx, target_output].item()

    incoming = torch.where(edge_index[1] == target_node_idx)[0].tolist()
    scores   = {}
    for ei in incoming:
        keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
        keep[ei] = False
        with torch.no_grad():
            # return_logits=True keeps units consistent with the MC-Shapley baseline
            p = model(x, edge_index[:, keep], return_logits=True)[target_node_idx, target_output].item()
        src = int(edge_index[0, ei])
        dst = int(edge_index[1, ei])
        scores[(src, dst)] = baseline - p

    return scores, baseline


### 15.9 Graph-SHAP visualization helpers
#### Ranked horizontal bar charts with 95 % CI error bars for all Shapley levels.
#### Node×feature results also render as a diverging-color heatmap so the spatial
#### pattern of upstream influence is immediately visible.  All outputs are saved
#### to OUTPUTS_DIR; plt.show() is replaced with plt.close() so plots never block
#### script execution.

def plot_graphshap_bars(shap_mean, shap_ci95, player_names,
                        title, xlabel="Mean Shapley value (signed)",
                        max_show=30, save_path=None):
    sort_idx = np.argsort(np.abs(shap_mean))[::-1][:max_show]
    names = [player_names[i] for i in sort_idx]
    means = shap_mean[sort_idx]
    cis   = shap_ci95[sort_idx]
    colors = ["steelblue" if m >= 0 else "coral" for m in means]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.38)))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=cis, color=colors, align="center", height=0.7,
            error_kw=dict(ecolor="black", capsize=3, linewidth=1.1))
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # Value labels on each bar so small contributions are legible
    x_range = max(np.abs(means).max(), 1e-12)
    for yi, m in enumerate(means):
        offset = x_range * 0.02
        ha = "left" if m >= 0 else "right"
        ax.text(m + (offset if m >= 0 else -offset), yi,
                f"{m:+.2e}", va="center", ha=ha, fontsize=6.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


### Node x feature heatmap plot
def plot_node_feature_heatmap(shap_mean, player_blocks_nf,
                               sub_nodes_global, features_list,
                               target_node_idx, idx_to_comid,
                               title, save_path=None):
    sub_nodes = [int(n) for n in sub_nodes_global.tolist()]
    F  = len(features_list)
    ns = len(sub_nodes)
    mat = np.zeros((ns, F), dtype=float)

    node_labels = []
    for ni, ng in enumerate(sub_nodes):
        cid  = idx_to_comid.get(ng, ng)
        role = f"★ target ({cid})" if ng == target_node_idx else str(cid)
        node_labels.append(role)

    for pi, (_, node_g, flat_idxs) in enumerate(player_blocks_nf):
        ni    = sub_nodes.index(node_g)
        f_idx = flat_idxs[0] % F
        mat[ni, f_idx] = shap_mean[pi]

    vmax = max(np.abs(mat).max(), 1e-12)
    fig, ax = plt.subplots(figsize=(max(12, F * 0.65), max(4, ns * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(F))
    ax.set_xticklabels(features_list, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(ns))
    ax.set_yticklabels(node_labels, fontsize=8)
    fig.colorbar(im, ax=ax, label="Shapley value (signed, logit space)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig

### 15.10 Graph-SHAP utilities

#### NOTE: redefinition of reconstruct_from_coalition from 15.5; this later version overrides the earlier one
def reconstruct_from_coalition(
    coalition_mask,
    player_blocks,
    x_explain,
    bg_draw,
    sub_nodes,
    lookback,
    n_features,
    device,
):
    """
    Reconstruct a masked sample for Graph-SHAP.

    Expected inputs
    ---------------
    coalition_mask : array-like of bool, shape [n_players]
        True  -> keep that player block from x_explain
        False -> replace that player block with background values

    player_blocks : list
        Each block must be of the form:
            (name, node_idx, flat_positions)
        or
            (name, node_idx, flat_positions, *extras)

        where flat_idx = lag_idx * n_features + feat_idx

    x_explain : np.ndarray or torch.Tensor, shape [N, lookback, F]
        The explained sample in node-major format.

    bg_draw : dict
        Background draw keyed by local node index.
        Each value may be:
            - [lookback, F]
            - [F, lookback]
            - [1, lookback, F]
            - [lookback * F]
        This function standardizes to [lookback, F].

    sub_nodes : array-like
        Present for interface compatibility; not used here except for length checks.

    Returns
    -------
    torch.FloatTensor, shape [N, lookback, F]
    """
    import numpy as np
    import torch

    if isinstance(x_explain, torch.Tensor):
        x_rec = x_explain.detach().cpu().numpy().copy()
    else:
        x_rec = np.array(x_explain, dtype=np.float32, copy=True)

    if x_rec.ndim != 3:
        raise ValueError(f"x_explain must be 3D [N, lookback, F], got shape {x_rec.shape}")

    n_nodes, lb, nf = x_rec.shape
    if lb != lookback or nf != n_features:
        raise ValueError(
            f"x_explain shape mismatch: got {x_rec.shape}, expected [N, {lookback}, {n_features}]"
        )

    coalition_mask = np.asarray(coalition_mask).astype(bool)
    if len(coalition_mask) != len(player_blocks):
        raise ValueError(
            f"coalition_mask length {len(coalition_mask)} != number of blocks {len(player_blocks)}"
        )

    bg_std = {}
    for node_idx, arr in bg_draw.items():
        node_idx = int(node_idx)

        if isinstance(arr, torch.Tensor):
            a = arr.detach().cpu().numpy()
        else:
            a = np.asarray(arr)

        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]

        if a.ndim == 1:
            if a.size != lookback * n_features:
                raise ValueError(
                    f"Background for node {node_idx} has flat size {a.size}, "
                    f"expected {lookback * n_features}"
                )
            a = a.reshape(lookback, n_features)

        elif a.ndim == 2 and a.shape == (n_features, lookback):
            a = a.T

        elif a.ndim == 2 and a.shape == (lookback, n_features):
            pass

        else:
            raise ValueError(
                f"Background for node {node_idx} has unsupported shape {a.shape}; "
                f"expected [lookback, F], [F, lookback], [1, lookback, F], or flat."
            )

        bg_std[node_idx] = a.astype(np.float32, copy=False)

    for p_idx, block in enumerate(player_blocks):
        if len(block) < 3:
            raise ValueError(f"Malformed block at index {p_idx}: {block}")

        _, node_idx, flat_idxs = block[:3]
        node_idx = int(node_idx)

        if coalition_mask[p_idx]:
            continue

        if node_idx < 0 or node_idx >= n_nodes:
            continue

        if node_idx not in bg_std:
            raise KeyError(f"Missing background draw for node {node_idx}")

        bg_node = bg_std[node_idx]

        for flat_idx in flat_idxs:
            flat_idx = int(flat_idx)
            lag_idx = flat_idx // n_features
            feat_idx = flat_idx % n_features

            if not (0 <= lag_idx < lookback and 0 <= feat_idx < n_features):
                raise ValueError(
                    f"Flat index {flat_idx} out of range -> (lag={lag_idx}, feat={feat_idx})"
                )

            x_rec[node_idx, lag_idx, feat_idx] = bg_node[lag_idx, feat_idx]

    return torch.tensor(x_rec, dtype=torch.float32, device=device)


### 15.11 Graph-SHAP runner
#### Orchestrates all four explanation levels for each target variable (Q, Temp)
#### at TARGET_COMID_FOR_PLOTS. The upstream subgraph is extracted once and then
#### all block-based Graph-SHAP runs are performed in LOCAL subgraph index space
#### to avoid global/local node-index mismatches.
#
#### IMPORTANT:
#### The internal Graph-SHAP utilities expect block definitions in the form:
####     (name, node_idx, flat_positions)
#### where:
####     flat_idx = lag_idx * n_features + feat_idx
#
#### Therefore we DO NOT convert blocks to (node, lag, feat) triplets here.
#### We only remap node_idx from global graph indexing to local subgraph indexing.

print("\n" + "=" * 65)
print("FULL GRAPH-SHAP  —  three-level analysis")
print("=" * 65)

idx_to_comid      = {v: k for k, v in comid_to_idx.items()}
target_node_idx_g = comid_to_idx[TARGET_COMID_FOR_PLOTS]


### 15.12 Helper utilities for consistent local-subgraph Graph-SHAP
def build_local_subgraph_inputs(X_TNF, sub_nodes_g):
    """
    Restrict full [T, N, F] tensor to local upstream subgraph order.
    Returns [T, N_sub, F].
    """
    sub_nodes_np = np.asarray(sub_nodes_g, dtype=np.int64)
    return X_TNF[:, sub_nodes_np, :].astype(np.float32)


### Decode flat positions into (lag, feature) for diagnostics
def flat_positions_to_lag_feat(flat_positions, n_features):
    """
    Decode flat positions into (lag_idx, feat_idx) for diagnostics only.
    Expected flattening:
        flat_idx = lag_idx * n_features + feat_idx
    """
    out = []
    for fp in flat_positions:
        fp = int(fp)
        lag_idx  = fp // n_features
        feat_idx = fp % n_features
        out.append((lag_idx, feat_idx))
    return out


### Remap blocks from global to local subgraph node indexing
def remap_any_blocks_to_local(blocks_global, global_to_local, label="blocks"):
    """
    Remap block definitions from global node indexing to local subgraph indexing
    while preserving the ORIGINAL block structure expected by the SHAP functions.

    Supported observed format:
      (name, global_node_idx, flat_positions)
      (name, global_node_idx, flat_positions, *extras)

    Output format:
      (name, local_node_idx, flat_positions)
      (name, local_node_idx, flat_positions, *extras)
    """
    blocks_local = []

    for bi, block in enumerate(blocks_global):
        if not isinstance(block, (tuple, list)):
            raise TypeError(f"[{label}] block {bi} is not tuple/list: {type(block)}")

        if len(block) < 3:
            raise ValueError(f"[{label}] expected at least 3 elements, got: {block}")

        block_name = block[0]
        node_idx_g = int(block[1])
        flat_positions = block[2]
        extras = list(block[3:]) if len(block) > 3 else []

        if not isinstance(flat_positions, (list, tuple, np.ndarray)):
            raise TypeError(
                f"[{label}] block '{block_name}' flat positions are not iterable: "
                f"{type(flat_positions)}"
            )

        if node_idx_g not in global_to_local:
            node_idx_l = -1
            flat_positions_l = list(flat_positions)
        else:
            node_idx_l = global_to_local[node_idx_g]
            flat_positions_l = list(flat_positions)

        block_local = tuple([block_name, node_idx_l, flat_positions_l] + extras)
        blocks_local.append(block_local)

    return blocks_local


### Validate block indices
def check_block_indices(blocks, n_nodes, lookback, n_features, label="blocks"):
    """
    Validate blocks in ORIGINAL flat-index schema:
      (name, node_idx, flat_positions)
    """
    bad = 0
    total_flat = 0

    for block in blocks:
        if not isinstance(block, (tuple, list)) or len(block) < 3:
            print(f"[WARN:{label}] malformed block header: {block}")
            bad += 1
            continue

        name = block[0]
        node_idx = int(block[1])
        flat_positions = block[2]

        if not isinstance(flat_positions, (list, tuple, np.ndarray)):
            print(f"[WARN:{label}] flat_positions malformed in '{name}': {type(flat_positions)}")
            bad += 1
            continue

        if not (0 <= node_idx < n_nodes):
            bad += 1
            if bad <= 10:
                print(f"[WARN:{label}] node index out of bounds in '{name}': {node_idx}")

        for fp in flat_positions:
            total_flat += 1
            fp = int(fp)
            lag_idx  = fp // n_features
            feat_idx = fp % n_features

            ok = (0 <= lag_idx < lookback) and (0 <= feat_idx < n_features)
            if not ok:
                bad += 1
                if bad <= 10:
                    print(f"[WARN:{label}] bad flat index in '{name}': "
                          f"flat={fp} -> (lag={lag_idx}, feat={feat_idx})")

    if bad == 0:
        print(f"[OK:{label}] all {total_flat} flat indices are valid for local subgraph tensor.")
    else:
        print(f"[WARN:{label}] {bad}/{max(total_flat,1)} issues found while checking block indices.")


### Summarize how much a reconstructed tensor changed
def summarize_tensor_change(x_a, x_b, name_a="A", name_b="B"):
    """
    Quick numeric diagnostic for reconstructed tensors.
    Accepts numpy arrays or torch tensors.
    """
    if isinstance(x_a, torch.Tensor):
        xa = x_a.detach().cpu().numpy()
    else:
        xa = np.asarray(x_a)

    if isinstance(x_b, torch.Tensor):
        xb = x_b.detach().cpu().numpy()
    else:
        xb = np.asarray(x_b)

    d = np.abs(xa - xb)
    print(f"[Tensor diff] {name_a} vs {name_b}: "
          f"sum={d.sum():.6f}  mean={d.mean():.6f}  max={d.max():.6f}")


### Sensitivity diagnostic on FULL graph
#### This confirms the trained model itself is responsive to the target node.
model.eval()
_ti = test_indices[0]
_t0, _t1 = _ti + 1, _ti + LOOKBACK + 1
with torch.no_grad():
    _x_orig  = torch.FloatTensor(X_TNF[_t0:_t1].transpose(1, 0, 2)).to(device)  # [N, lb, F]
    _x_bg    = _x_orig.clone()
    _x_bg[target_node_idx_g] = 0.0
    _p_orig  = model(_x_orig, edge_index, return_logits=True)[target_node_idx_g, 0].item()
    _p_bg    = model(_x_bg,   edge_index, return_logits=True)[target_node_idx_g, 0].item()

print(f"\n[Sensitivity check]  Q  target node (logit space):")
print(f"  logit (original input) = {_p_orig:.6f}")
print(f"  logit (target zeroed)  = {_p_bg:.6f}")
print(f"  Δ = {abs(_p_bg - _p_orig):.6f}  "
      f"{'✓ model is sensitive' if abs(_p_bg - _p_orig) > 1e-4 else '✗ WARNING: model output unchanged in logit space — SHAP values will be zero'}")
del _ti, _t0, _t1, _x_orig, _x_bg, _p_orig, _p_bg


### Extract local subgraph once
sub_nodes_g, edge_index_sub_g, target_sub_idx_g = extract_upstream_subgraph(
    edge_index      = edge_index,
    target_node_idx = target_node_idx_g,
    n_hops          = GRAPHSHAP_N_HOPS,
)
sub_nodes_g = np.asarray(sub_nodes_g, dtype=np.int64)
n_sub = sub_nodes_g.shape[0]

global_to_local = {int(g): i for i, g in enumerate(sub_nodes_g.tolist())}
local_to_global = {i: int(g) for i, g in enumerate(sub_nodes_g.tolist())}

print(f"\n{GRAPHSHAP_N_HOPS}-hop upstream subgraph around COMID {TARGET_COMID_FOR_PLOTS}:")
print(f"  Nodes : {n_sub}  |  Edges : {edge_index_sub_g.shape[1]}")
up_comids = [idx_to_comid.get(int(n), n) for n in sub_nodes_g.tolist()
             if int(n) != target_node_idx_g]
print(f"  Upstream COMIDs: {up_comids}")
print(f"  Global target idx : {target_node_idx_g}")
print(f"  Local  target idx : {target_sub_idx_g}")

### Build local [T, N_sub, F] tensor for all block-based Graph-SHAP
X_TNF_sub = build_local_subgraph_inputs(X_TNF, sub_nodes_g)

### Block-level perturbation check in LOCAL subgraph space
#### This tests the exact masking route used by block-based Graph-SHAP.
_feat_blocks_chk_global = make_feature_blocks(target_node_idx_g, LOOKBACK, FEATURES)

print("\n[Debug] first 3 raw feature blocks:")
for _b in _feat_blocks_chk_global[:3]:
    print("  ", type(_b), _b if len(str(_b)) < 250 else str(_b)[:250] + " ...")

_feat_blocks_chk_local = remap_any_blocks_to_local(
    _feat_blocks_chk_global,
    global_to_local,
    label="feature check blocks"
)

print("\n[Debug] first 3 remapped feature blocks:")
for _b in _feat_blocks_chk_local[:3]:
    print("  ", type(_b), _b if len(str(_b)) < 250 else str(_b)[:250] + " ...")
    _nm, _nd, _fp = _b[:3]
    print("     decoded:", flat_positions_to_lag_feat(_fp[:min(5, len(_fp))], len(FEATURES)))

check_block_indices(
    _feat_blocks_chk_local,
    n_nodes=n_sub,
    lookback=LOOKBACK,
    n_features=len(FEATURES),
    label="L1 feature blocks"
)

_ti_chk = test_indices[0]
_t0_chk, _t1_chk = _ti_chk + 1, _ti_chk + LOOKBACK + 1

### local explained sample: [N_sub, lb, F]
_x_exp_chk = X_TNF_sub[_t0_chk:_t1_chk].transpose(1, 0, 2).astype(np.float32)

### training-period background pool built in LOCAL subgraph space
_bg_pool_chk, _ = build_background_pool(
    X_TNF_sub,
    train_indices,
    dates,
    np.arange(n_sub, dtype=np.int64),
    LOOKBACK,
    n_samples=5,
    seed=999,
)

_bg_draw_chk = {ng: _bg_pool_chk[ng][0] for ng in range(n_sub)}

print("\n[Block-level perturbation check]  (LOCAL subgraph, logit space)")
for _tgt_o, _tgt_n in [(0, "Q"), (1, "Temp")]:
    _all_present = np.ones(len(_feat_blocks_chk_local), dtype=bool)
    _x_full = reconstruct_from_coalition(
        _all_present,
        _feat_blocks_chk_local,
        _x_exp_chk,
        _bg_draw_chk,
        np.arange(n_sub, dtype=np.int64),
        LOOKBACK,
        len(FEATURES),
        device
    )

    _all_absent = np.zeros(len(_feat_blocks_chk_local), dtype=bool)
    _x_absent = reconstruct_from_coalition(
        _all_absent,
        _feat_blocks_chk_local,
        _x_exp_chk,
        _bg_draw_chk,
        np.arange(n_sub, dtype=np.int64),
        LOOKBACK,
        len(FEATURES),
        device
    )

    summarize_tensor_change(_x_full, _x_absent, "full", "absent")

    with torch.no_grad():
        _p_full   = model(_x_full,   edge_index_sub_g, return_logits=True)[target_sub_idx_g, _tgt_o].item()
        _p_absent = model(_x_absent, edge_index_sub_g, return_logits=True)[target_sub_idx_g, _tgt_o].item()

    _delta = abs(_p_full - _p_absent)
    _ok = "✓" if _delta > 1e-4 else "✗ WARNING — masking has no effect"
    print(f"  {_tgt_n}  full={_p_full:+.5f}  absent={_p_absent:+.5f}  Δ={_delta:.5f}  {_ok}")

print("\n[Sanity] target-node explained slice, first 2 lags × first 5 feats:")
print(_x_exp_chk[target_sub_idx_g, :2, :5])

print("\n[Sanity] target-node FULL reconstructed slice:")
print(_x_full.detach().cpu().numpy()[target_sub_idx_g, :2, :5])

print("\n[Sanity] target-node ABSENT reconstructed slice:")
print(_x_absent.detach().cpu().numpy()[target_sub_idx_g, :2, :5])

del (
    _feat_blocks_chk_global, _feat_blocks_chk_local,
    _ti_chk, _t0_chk, _t1_chk,
    _x_exp_chk, _bg_pool_chk, _bg_draw_chk,
    _x_full, _x_absent, _p_full, _p_absent, _delta, _ok,
    _all_present, _all_absent
)


### Main Graph-SHAP loop
#### NOTE:
####   L1/L2/L3 are now all run in LOCAL subgraph space.
####   Structural Graph-SHAP remains on the FULL graph.
for tgt_out, tgt_name in [(0, "Q"), (1, "Temp")]:
    print(f"\n{'─'*60}")
    print(f"  {tgt_name}  at  COMID {TARGET_COMID_FOR_PLOTS}")
    print(f"{'─'*60}")

    ### Level 1: Feature blocks at target node
    print("\n[L1] Feature blocks (MC Shapley, LOCAL subgraph) …")
    feat_blocks_global = make_feature_blocks(target_node_idx_g, LOOKBACK, FEATURES)
    feat_blocks_local = remap_any_blocks_to_local(
        feat_blocks_global,
        global_to_local,
        label=f"L1 feature blocks ({tgt_name})"
    )

    check_block_indices(
        feat_blocks_local,
        n_nodes=n_sub,
        lookback=LOOKBACK,
        n_features=len(FEATURES),
        label=f"L1 feature blocks ({tgt_name})"
    )

    sv_f_mean, sv_f_std, sv_f_ci = mc_shapley_over_blocks(
        model=model,
        edge_index=edge_index_sub_g,
        X_TNF=X_TNF_sub,
        test_indices=test_indices,
        dates=dates,
        player_blocks=feat_blocks_local,
        sub_nodes_global=np.arange(n_sub, dtype=np.int64),
        target_node_idx=target_sub_idx_g,
        lookback=LOOKBACK,
        n_features=len(FEATURES),
        n_background=GRAPHSHAP_N_BACKGROUND,
        n_explain=GRAPHSHAP_N_EXPLAIN,
        n_mc_perms=GRAPHSHAP_N_MC_PERMS,
        target_output=tgt_out,
        device=str(device),
        background_indices=train_indices,
        season_window_days=GRAPHSHAP_SEASON_DAYS,
    )

    feat_names = [b[0] for b in feat_blocks_local]
    print("  Top features:")
    for ri in np.argsort(np.abs(sv_f_mean))[::-1][:5]:
        s = "↑" if sv_f_mean[ri] >= 0 else "↓"
        print(f"    {s} {feat_names[ri]:>12s}  φ={sv_f_mean[ri]:+.8f}  ±{sv_f_ci[ri]:.8f}")

    plot_graphshap_bars(
        sv_f_mean,
        sv_f_ci,
        feat_names,
        title=f"Feature Graph-SHAP  |  {tgt_name}  |  COMID {TARGET_COMID_FOR_PLOTS}",
        save_path=os.path.join(OUTPUTS_DIR, f"graphshap_L1_feat_{tgt_name}_{TARGET_COMID_FOR_PLOTS}.png"),
    )

    np.save(os.path.join(SAVE_DIR, f"graphshap_L1_feat_{tgt_name}_mean.npy"), sv_f_mean)
    np.save(os.path.join(SAVE_DIR, f"graphshap_L1_feat_{tgt_name}_ci95.npy"), sv_f_ci)

    ### Level 2: Lag blocks
    print("\n[L2] Lag blocks (MC Shapley, LOCAL subgraph) …")
    lag_blocks_global = make_lag_blocks(target_node_idx_g, LOOKBACK, FEATURES)
    lag_blocks_local = remap_any_blocks_to_local(
        lag_blocks_global,
        global_to_local,
        label=f"L2 lag blocks ({tgt_name})"
    )

    check_block_indices(
        lag_blocks_local,
        n_nodes=n_sub,
        lookback=LOOKBACK,
        n_features=len(FEATURES),
        label=f"L2 lag blocks ({tgt_name})"
    )

    sv_l_mean, sv_l_std, sv_l_ci = mc_shapley_over_blocks(
        model=model,
        edge_index=edge_index_sub_g,
        X_TNF=X_TNF_sub,
        test_indices=test_indices,
        dates=dates,
        player_blocks=lag_blocks_local,
        sub_nodes_global=np.arange(n_sub, dtype=np.int64),
        target_node_idx=target_sub_idx_g,
        lookback=LOOKBACK,
        n_features=len(FEATURES),
        n_background=GRAPHSHAP_N_BACKGROUND,
        n_explain=GRAPHSHAP_N_EXPLAIN,
        n_mc_perms=GRAPHSHAP_N_MC_PERMS,
        target_output=tgt_out,
        device=str(device),
        background_indices=train_indices,
        season_window_days=GRAPHSHAP_SEASON_DAYS,
    )

    lag_names = [b[0] for b in lag_blocks_local]
    plot_graphshap_bars(
        sv_l_mean,
        sv_l_ci,
        lag_names,
        title=f"Temporal Graph-SHAP  |  {tgt_name}  |  COMID {TARGET_COMID_FOR_PLOTS}",
        save_path=os.path.join(OUTPUTS_DIR, f"graphshap_L2_lags_{tgt_name}_{TARGET_COMID_FOR_PLOTS}.png"),
        max_show=LOOKBACK,
    )

    np.save(os.path.join(SAVE_DIR, f"graphshap_L2_lags_{tgt_name}_mean.npy"), sv_l_mean)
    np.save(os.path.join(SAVE_DIR, f"graphshap_L2_lags_{tgt_name}_ci95.npy"), sv_l_ci)

    ### Level 3: Node × Feature heatmap
    n_nf = n_sub * len(FEATURES)
    if n_nf <= GRAPHSHAP_MAX_NF_PLAYERS:
        print(f"\n[L3] Node×Feature blocks ({n_nf} players, MC Shapley, LOCAL subgraph) …")

        nf_blocks_global = make_node_feature_blocks(
            sub_nodes_g,
            FEATURES,
            target_node_idx_g,
            LOOKBACK
        )
        nf_blocks_local = remap_any_blocks_to_local(
            nf_blocks_global,
            global_to_local,
            label=f"L3 node-feature blocks ({tgt_name})"
        )

        check_block_indices(
            nf_blocks_local,
            n_nodes=n_sub,
            lookback=LOOKBACK,
            n_features=len(FEATURES),
            label=f"L3 node-feature blocks ({tgt_name})"
        )

        sv_nf_mean, sv_nf_std, sv_nf_ci = mc_shapley_over_blocks(
            model=model,
            edge_index=edge_index_sub_g,
            X_TNF=X_TNF_sub,
            test_indices=test_indices,
            dates=dates,
            player_blocks=nf_blocks_local,
            sub_nodes_global=np.arange(n_sub, dtype=np.int64),
            target_node_idx=target_sub_idx_g,
            lookback=LOOKBACK,
            n_features=len(FEATURES),
            n_background=GRAPHSHAP_N_BACKGROUND,
            n_explain=GRAPHSHAP_N_EXPLAIN,
            n_mc_perms=GRAPHSHAP_N_MC_NF,
            target_output=tgt_out,
            device=str(device),
            background_indices=train_indices,
            season_window_days=GRAPHSHAP_SEASON_DAYS,
            seed=20,
        )

        nf_names = [b[0] for b in nf_blocks_local]
        plot_graphshap_bars(
            sv_nf_mean,
            sv_nf_ci,
            nf_names,
            title=f"Node×Feature Graph-SHAP  |  {tgt_name}  |  COMID {TARGET_COMID_FOR_PLOTS}",
            save_path=os.path.join(OUTPUTS_DIR, f"graphshap_L3_nf_{tgt_name}_{TARGET_COMID_FOR_PLOTS}.png"),
        )

        plot_node_feature_heatmap(
            sv_nf_mean,
            nf_blocks_local,
            np.arange(n_sub, dtype=np.int64),
            FEATURES,
            target_sub_idx_g,
            {i: idx_to_comid[local_to_global[i]] for i in range(n_sub)},
            title=f"Node×Feature Shapley Heatmap  |  {tgt_name}  |  COMID {TARGET_COMID_FOR_PLOTS}",
            save_path=os.path.join(OUTPUTS_DIR, f"graphshap_L3_heatmap_{tgt_name}_{TARGET_COMID_FOR_PLOTS}.png"),
        )

        np.save(os.path.join(SAVE_DIR, f"graphshap_L3_nf_{tgt_name}_mean.npy"), sv_nf_mean)
        np.save(os.path.join(SAVE_DIR, f"graphshap_L3_nf_{tgt_name}_ci95.npy"), sv_nf_ci)

    else:
        print(f"\n[L3] Skipped: {n_nf} players exceeds GRAPHSHAP_MAX_NF_PLAYERS={GRAPHSHAP_MAX_NF_PLAYERS}. "
              f"Reduce GRAPHSHAP_N_HOPS or increase the limit.")

    ### Structural Graph-SHAP on FULL graph
    print(f"\n[Structural] MC Shapley over incoming edges ({tgt_name}) …")
    sv_e_mean, sv_e_std, sv_e_ci, edge_players = compute_structural_graphshap_mc(
        model=model,
        edge_index=edge_index,
        X_TNF=X_TNF,
        test_indices=test_indices,
        target_node_idx=target_node_idx_g,
        lookback=LOOKBACK,
        n_explain=GRAPHSHAP_N_EXPLAIN,
        n_mc_perms=GRAPHSHAP_N_MC_STRUCT,
        target_output=tgt_out,
        device=str(device),
        seed=30,
    )

    if len(sv_e_mean) > 0:
        edge_names = [f"{idx_to_comid.get(s, s)} → {idx_to_comid.get(d, d)}"
                      for s, d in edge_players]

        plot_graphshap_bars(
            sv_e_mean,
            sv_e_ci,
            edge_names,
            title=f"Structural Graph-SHAP  |  {tgt_name}  |  COMID {TARGET_COMID_FOR_PLOTS}",
            xlabel="Mean Shapley value: edge contribution (signed)",
            save_path=os.path.join(OUTPUTS_DIR, f"graphshap_structural_{tgt_name}_{TARGET_COMID_FOR_PLOTS}.png"),
        )

        np.save(os.path.join(SAVE_DIR, f"graphshap_structural_{tgt_name}_mean.npy"), sv_e_mean)
        np.save(os.path.join(SAVE_DIR, f"graphshap_structural_{tgt_name}_ci95.npy"), sv_e_ci)

        sample_x, *_ = next(iter(test_loader))
        sample_x = sample_x.squeeze(0).float()
        ablation, bl = compute_edge_ablation_importance(
            model,
            edge_index,
            sample_x,
            target_node_idx_g,
            tgt_out,
            str(device)
        )
        print(f"  Ablation baseline ({tgt_name}): {bl:.6f}")
        for (s, d), v in sorted(ablation.items(), key=lambda x: -abs(x[1])):
            mc_idx = edge_players.index((s, d)) if (s, d) in edge_players else -1
            mc_val = sv_e_mean[mc_idx] if mc_idx >= 0 else float("nan")
            print(f"    {idx_to_comid.get(s,s)} → {idx_to_comid.get(d,d)}"
                  f"  ablation={v:+.5f}  MC-Shapley={mc_val:+.5f}")

print(f"\nAll Graph-SHAP outputs saved to: {SAVE_DIR}")