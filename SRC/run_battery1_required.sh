#!/usr/bin/env bash
# run_battery1_required.sh
#
# Purpose : NASA-ILOH Battery I -- the "Required" runs only (I01-I04), on the
#           fully reconstructed/standardized dataset (post pipeline rerun +
#           corrections #4/#5/#10A) with a 30-day lookback. Establishes the
#           corrected baseline and tests whether graph attention earns its
#           cost at SO7-9 and SO6-9, before any conditional expansion
#           (I05-I07) or reproducibility runs (I08-I09, after architecture
#           selection). Do NOT run this until the pipeline rerun has
#           completed and the new data has been validated (Phase 3 checks).
# Usage   : bash SRC/run_battery1_required.sh
#
# Standard config: 30-day lookback, raw Q, Q:T 0.80:0.20, 3 GAT layers (when
# enabled), 4 attention heads (default), DOY sine-only seasonal predictor
# (per instruction, not the sine+cosine default), graph smoothness
# lambda=0.001 (default), high-flow thresh=100/mult=5.0 (default), seed 0,
# 180-day validation window, test from 2025-01-01, corrected coarsened graph.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

run() {
  echo
  echo "=============================================="
  echo "Starting: $*"
  echo "=============================================="
  python3 SRC/DDNN_Model.py "$@"
}

COMMON="--lookback 30 --q-weight 0.80 --t-weight 0.20 --doy-component sin"

# I01: SO7-9, directed 3-layer GAT -- corrected 30-day reference
run --streamorder-min 7 --streamorder-max 9 $COMMON --gat-layers 3 \
    --run-name I01_so7-9_lb30_gat3

# I02: SO7-9, LSTM-only -- does graph attention add value?
run --streamorder-min 7 --streamorder-max 9 $COMMON --no-gat \
    --run-name I02_so7-9_lb30_lstmOnly

# I03: SO6-9, directed 3-layer GAT -- does adding SO6 help SO7-9?
run --streamorder-min 6 --streamorder-max 9 $COMMON --gat-layers 3 \
    --run-name I03_so6-9_lb30_gat3

# I04: SO6-9, LSTM-only -- does the graph contribute more with smaller tributaries included?
run --streamorder-min 6 --streamorder-max 9 $COMMON --no-gat \
    --run-name I04_so6-9_lb30_lstmOnly

echo
echo "Battery I required runs (I01-I04) complete. Run: python3 SRC/aggregate_results.py"
echo "Remaining decisions: I05-I07 (conditional on I01-I04 results), I08-I09 (after architecture selection)."
