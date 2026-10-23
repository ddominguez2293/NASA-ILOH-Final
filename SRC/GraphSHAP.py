#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GraphSHAP.py
# STEP 9 of the pipeline (runs AFTER DDNN_Model.py has trained and saved a model).
#
# Purpose : explain a trained Graph LSTM-GAT multitask model with Graph-SHAP
#           (feature, temporal, node x feature, and structural levels). Split
#           out of DDNN_Model.py so training runs aren't slowed down by SHAP's
#           Monte Carlo permutations — run this separately once a model exists.
# Inputs  : hydro_temp_data_trainval_2013plus.parquet   (NODES_PATH)
#           edge_list.parquet                           (EDGES_PATH)
#           outputs/<run-name>/results.txt              (recorded run config)
#           trained model weights, scaler_X.pkl, features.pkl in SAVE_DIR
#           (all written by DDNN_Model.py --save-model)
# Outputs : Graph-SHAP figures and .npy arrays, written to OUTPUTS_DIR / SAVE_DIR
# Usage   : python SRC/GraphSHAP.py --run-name <run-name>
#
# Note    : the domain filter, graph construction (coarsening/bidirectional),
#           and architecture (GAT depth / LSTM-only) are loaded from
#           outputs/<run-name>/results.txt at runtime (see _load_run_config
#           in Section 2) rather than hand-copied into constants here — that
#           hand-copying is what let this file drift out of sync with
#           DDNN_Model.py every time a new experiment flag was added.
#           DDNN_Model.py can't be imported directly (it's a CLI script that
#           parses argv and runs training at import time), so results.txt is
#           the shared contract between the two scripts instead.
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code
#
"""
GraphSHAP — Graph-SHAP Interpretation for the Graph LSTM-GAT Multitask Model
Illinois River Basin (HUC 0712-0713) | Daily

Loads a model trained by DDNN_Model.py and runs the full three-level Monte
Carlo Shapley framework (feature, temporal, node×feature) plus structural
Graph-SHAP over incoming edges, for a single target COMID.
"""

## 1. Imports
### Standard library
import os
import re
import argparse
from pathlib import Path

### Scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

### PyTorch + graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph

### Sklearn
import shap  # optional — retained for potential kernel-SHAP comparisons


## 2. Configuration

### Which run to explain
#### Everything below that describes the training run (domain, lookback,
#### graph construction, architecture) is loaded from that run's
#### outputs/<run-name>/results.txt rather than hand-copied into constants
#### here — hand-copied constants are exactly what let this file drift out
#### of sync with DDNN_Model.py every time a new experiment flag was added
#### (graph coarsening, bidirectional edges, GAT depth, LSTM-only mode).
#### Only run this against a run that was trained with --save-model, since
#### that's what writes the .pth/scaler/features artifacts this script loads.
parser = argparse.ArgumentParser(description="Graph-SHAP interpretation for a trained DDNN_Model.py run.")
parser.add_argument("--run-name", type=str, required=True,
                     help="outputs/<run-name> folder of the training run to explain "
                          "(must have been trained with --save-model)")
parser.add_argument("--target-comid", type=int, default=None,
                     help="COMID to explain; defaults to the run's TARGET_COMID_FOR_PLOTS")
args = parser.parse_args()

### Input data paths — same inputs DDNN_Model.py reads.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

NODES_PATH = PROJECT_ROOT / "data" / "daily_drivers" / "hydro_temp_data_trainval_2013plus.parquet"
EDGES_PATH = PROJECT_ROOT / "data" / "daily_drivers" / "edge_list.parquet"


### Parse the training run's recorded config out of its results.txt
#### Only the "=== Test variables ===" block is needed here (plain KEY: value
#### lines); metric sections below it use a different format and are ignored.
def _load_run_config(run_name: str) -> dict:
    results_path = PROJECT_ROOT / "outputs" / run_name / "results.txt"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No results.txt for run '{run_name}' at {results_path}. "
            "GraphSHAP.py needs the training run's recorded config to reconstruct "
            "the same domain/graph/architecture — pass the exact --run-name used in DDNN_Model.py."
        )
    kv = {}
    kv_re = re.compile(r"^([A-Z][A-Z0-9_]*):\s*(.*)$")
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("===") and "Test variables" not in line and kv:
                break  # stop once we've left the test-variables block
            m = kv_re.match(line)
            if m:
                kv[m.group(1)] = m.group(2)
    return kv


def _as_bool(s, default):
    if s is None:
        return default
    return s.strip().lower() in ("true", "1", "yes")


RUN_CONFIG = _load_run_config(args.run_name)

### Temporal split & sequence — loaded from the training run's config.
SPLIT_DATE = pd.Timestamp(RUN_CONFIG.get("SPLIT_DATE", "2025-01-01"))
LOOKBACK   = int(RUN_CONFIG.get("LOOKBACK", 30))

### Model hyperparameters — must match the architecture of the saved weights,
#### since load_state_dict requires identical layer shapes.
H_LSTM       = int(RUN_CONFIG.get("H_LSTM", 64))
LSTM_LAYERS  = int(RUN_CONFIG.get("LSTM_LAYERS", 1))
LSTM_DROPOUT = float(RUN_CONFIG.get("LSTM_DROPOUT", 0.2))

H_GNN        = int(RUN_CONFIG.get("H_GNN", 64))
GNN_DROPOUT  = float(RUN_CONFIG.get("GNN_DROPOUT", 0.2))
GAT_HEADS    = int(RUN_CONFIG.get("GAT_HEADS", 4))

### Graph attention depth / ablation — defaults match DDNN_Model.py's
#### pre-ablation behavior, for results.txt files written before these flags existed.
USE_GAT      = _as_bool(RUN_CONFIG.get("USE_GAT"), True)
N_GAT_LAYERS = int(RUN_CONFIG.get("N_GAT_LAYERS", 2))

### Modeling domain — must match the stream-order band used for training.
STREAMORDER_MIN = int(RUN_CONFIG.get("STREAMORDER_MIN", 8))
STREAMORDER_MAX = int(RUN_CONFIG.get("STREAMORDER_MAX", 9))

### Target COMID for this explanation pass.
TARGET_COMID_FOR_PLOTS = args.target_comid or int(RUN_CONFIG.get("TARGET_COMID_FOR_PLOTS", 14787569))

### Trained-artifact + output paths.
SAVE_DIR    = PROJECT_ROOT / "data" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH  = os.path.join(SAVE_DIR, "features.pkl")
SCALER_X_PATH  = os.path.join(SAVE_DIR, "scaler_X.pkl")
MODEL_FILENAME = "graph_lstm_gat_multitask_q_t_maskedloss_sigmoid_head.pth"

### Graph coarsening / bidirectionality — must match the training run's settings.
USE_COARSENED_GRAPH  = _as_bool(RUN_CONFIG.get("USE_COARSENED_GRAPH"), True)
MAX_HOPS_COARSEN     = int(RUN_CONFIG.get("MAX_HOPS_COARSEN", 5000))
COARSE_MODE          = RUN_CONFIG.get("COARSE_MODE", "nearest")
BIDIRECTIONAL_EDGES  = _as_bool(RUN_CONFIG.get("BIDIRECTIONAL_EDGES"), False)

print(f"Explaining run: {args.run_name}")
print(f"Loaded config: SO{STREAMORDER_MIN}-{STREAMORDER_MAX}, lookback={LOOKBACK}, "
      f"use_gat={USE_GAT}, n_gat_layers={N_GAT_LAYERS}, coarsened={USE_COARSENED_GRAPH}, "
      f"bidirectional={BIDIRECTIONAL_EDGES}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Graph-SHAP tuneable knobs
GRAPHSHAP_N_HOPS         = 2      # upstream subgraph depth
GRAPHSHAP_N_BACKGROUND   = 40     # distinct windows in masker pool
GRAPHSHAP_N_EXPLAIN      = 25     # windows to average Shapley over
GRAPHSHAP_N_MC_PERMS     = 120    # permutations per explain window
GRAPHSHAP_N_MC_STRUCT    = 100    # permutations for structural SHAP
GRAPHSHAP_N_MC_NF        = 60     # permutations for node×feature
GRAPHSHAP_MAX_NF_PLAYERS = 100    # skip node×feature if larger
GRAPHSHAP_SEASON_DAYS    = 45     # ±days for seasonal background


## 3. Utility functions (data loading / graph construction)
#### Duplicated from DDNN_Model.py — kept identical so the reconstructed
#### domain and graph match the model being explained.

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


## 4. Load trained artifacts & rebuild the modeling domain
#### FEATURES comes from features.pkl (saved by DDNN_Model.py) so the feature
#### order exactly matches what the trained scaler_X and model expect — only
#### the stream-order band is re-applied here to rebuild the same row set.

nodes_df_raw = pd.read_parquet(NODES_PATH)
edges_df = pd.read_parquet(EDGES_PATH)

nodes_df_raw["date"] = pd.to_datetime(nodes_df_raw["date"])
nodes_df_raw = nodes_df_raw.sort_values(["date", "comid"]).reset_index(drop=True)

### Modeling domain: stream order filter (must match the training run)
so_filter = pd.to_numeric(nodes_df_raw["streamorde"], errors="coerce")
keep_rows = (so_filter >= STREAMORDER_MIN) & (so_filter <= STREAMORDER_MAX)
nodes_df_raw = nodes_df_raw.loc[keep_rows].copy()
nodes_df_raw = nodes_df_raw.sort_values(["date", "comid"]).reset_index(drop=True)

print(f"Rows after stream order filter ({STREAMORDER_MIN} <= streamorde <= {STREAMORDER_MAX}): {len(nodes_df_raw):,}")
print(f"Unique COMIDs: {nodes_df_raw['comid'].nunique():,}")

### Feature list — loaded, not redefined, so the column order matches training
FEATURES = joblib.load(FEATURES_PATH)
print(f"Loaded FEATURES (n={len(FEATURES)}) from: {FEATURES_PATH}")

for c in FEATURES:
    nodes_df_raw[c] = pd.to_numeric(nodes_df_raw[c], errors="coerce")

nodes_df = nodes_df_raw[["comid", "date"] + FEATURES].copy()
nodes_df = replace_sentinel_with_nan(nodes_df, FEATURES)


## 5. Graph construction
#### Identical approach to DDNN_Model.py Section 5: directed upstream ->
#### downstream COMID graph, optionally coarsened to observed nodes only.

all_comids = np.sort(nodes_df["comid"].unique())
comid_to_idx = {c: i for i, c in enumerate(all_comids)}
idx_to_comid = {v: k for k, v in comid_to_idx.items()}
n_nodes = len(all_comids)

edge_pairs_all = build_edge_pairs_from_nodes(edges_df)

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
else:
    edge_pairs = edge_pairs_all[
        edge_pairs_all["comid_up"].isin(all_comids) & edge_pairs_all["comid_dn"].isin(all_comids)
    ].drop_duplicates().reset_index(drop=True)

### Optional bidirectional edges — must match the training run (G06-style).
if BIDIRECTIONAL_EDGES and len(edge_pairs) > 0:
    reversed_pairs = edge_pairs.rename(columns={"comid_up": "comid_dn", "comid_dn": "comid_up"})
    edge_pairs = pd.concat([edge_pairs, reversed_pairs], ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"[BidirectionalEdges] Added reverse edges: {len(edge_pairs)} total directed edges.")

if len(edge_pairs) > 0:
    src = edge_pairs["comid_up"].map(comid_to_idx).to_numpy()
    dst = edge_pairs["comid_dn"].map(comid_to_idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).to(device)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} directed edges")
edge_qa_checks(edge_pairs, all_comids)

if edge_index.numel() == 0:
    raise RuntimeError("edge_index is empty. Check USE_COARSENED_GRAPH settings and the domain filter.")


## 6. Feature scaling
#### Loads the scaler fit during training rather than refitting, so inputs
#### are scaled exactly as the model was trained to expect.

scaler_X = joblib.load(SCALER_X_PATH)
X_all = np.nan_to_num(nodes_df[FEATURES].values, nan=0.0)
nodes_df[FEATURES] = scaler_X.transform(X_all)


## 7. Tensor assembly
#### Only the feature tensor is needed here — Graph-SHAP explains the
#### model's inputs (FEATURES), not the targets, so Q/T columns and masks
#### from DDNN_Model.py's Section 7 are not rebuilt.

dates = np.sort(nodes_df["date"].unique())
Tsteps = len(dates)
Fdim = len(FEATURES)

X_TNF = np.zeros((Tsteps, n_nodes, Fdim), dtype=np.float32)
date_to_idx = {d: i for i, d in enumerate(dates)}

for d in dates:
    t = date_to_idx[d]
    sub = nodes_df[nodes_df["date"] == d]
    idxs = sub["comid"].map(comid_to_idx).to_numpy()
    X_TNF[t, idxs, :] = sub[FEATURES].to_numpy(dtype=np.float32)


## 8. Train / test split indices
#### Same temporal split as DDNN_Model.py Section 8, without building the
#### Dataset/DataLoader machinery — Graph-SHAP only needs the index lists.

t_dates = dates[LOOKBACK:]
train_indices = [i for i, d in enumerate(t_dates) if d < SPLIT_DATE]
test_indices  = [i for i, d in enumerate(t_dates) if d >= SPLIT_DATE]

assert len(train_indices) > 0, "No training-period windows found before split date."
assert len(test_indices)  > 0, "No test-period windows found on/after split date."

print(f"\n{len(train_indices)} training-period windows; {len(test_indices)} test-period windows.")


## 9. Load the trained model
#### Same architecture as DDNN_Model.py Section 9; weights are loaded from
#### the .pth saved by that script.

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
        # return_logits=True avoids sigmoid saturation: near 0/1, individual
        # feature contributions shift the output by < 1e-7 (below float32
        # precision). Logit space keeps changes in the ±1 range where
        # float32 has full precision — this is why Graph-SHAP always uses it.
        return raw if return_logits else torch.sigmoid(raw)

model = GraphLSTM_MTL(
    in_dim=len(FEATURES),
    lstm_hidden=H_LSTM, lstm_layers=LSTM_LAYERS, lstm_dropout=LSTM_DROPOUT,
    gnn_hidden=H_GNN, gnn_dropout=GNN_DROPOUT, heads=GAT_HEADS, out_dim=2,
    use_gat=USE_GAT, n_gat_layers=N_GAT_LAYERS,
).to(device)

model.load_state_dict(torch.load(os.path.join(SAVE_DIR, MODEL_FILENAME), map_location=device))
model.eval()
print(f"\nLoaded trained model from: {os.path.join(SAVE_DIR, MODEL_FILENAME)}")


## 10. Graph-SHAP interpretation
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

### 10.1 Local upstream subgraph extraction
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


### 10.2 Player (block) definitions
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


### 10.3 Seasonal background pool
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


### 10.4 Graph-aware coalition reconstructor
#### Absent players → background values; present players → actual explain values.
#### Masking is graph-aware: all subgraph nodes start from background, then only
#### the coalition members have their explain-window values restored.  This keeps
#### the explanation grounded in the local routed neighborhood.

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


### 10.5 Monte Carlo Shapley core
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


### 10.6 Structural Graph-SHAP — MC Shapley over incoming edges
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


### 10.7 Edge ablation — fast structural sanity check
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


### 10.8 Graph-SHAP visualization helpers
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


### 10.9 Local-subgraph helper utilities
#### Keeps all block-based Graph-SHAP runs in LOCAL subgraph index space to
#### avoid global/local node-index mismatches.

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


print("\n" + "=" * 65)
print("FULL GRAPH-SHAP  —  three-level analysis")
print("=" * 65)

target_node_idx_g = comid_to_idx[TARGET_COMID_FOR_PLOTS]

### Sensitivity diagnostic on FULL graph
#### Confirms the loaded model is responsive to the target node before
#### spending time on the full Monte Carlo Shapley runs.
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
####   L1/L2/L3 are all run in LOCAL subgraph space.
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

        ### Edge ablation sanity check, using the first test-period window directly
        _ti_s = test_indices[0]
        _t0_s, _t1_s = _ti_s + 1, _ti_s + LOOKBACK + 1
        sample_x = torch.FloatTensor(X_TNF[_t0_s:_t1_s].transpose(1, 0, 2)).float()

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
