#!/usr/bin/env bash
# run_phase3_architecture_battery.sh
#
# Purpose : chain the architecture-diagnostic battery proposed after Phase I
#           (Oct 2026): Phase I found pooled Q NSE positive while individual
#           stream orders (esp. SO8) were often negative -- i.e. the model may
#           be learning regional/between-reach discharge differences without
#           adequately representing within-reach temporal dynamics. These runs
#           isolate whether the GAT / graph construction / smoothness penalty
#           are helping, hurting, or irrelevant to that per-reach skill, plus
#           a DOY single-component ablation (ex- series) run alongside it.
#           Stops at the first failing run (set -e).
# Usage   : bash SRC/run_phase3_architecture_battery.sh
#
# Runs    : ex01-ex02 (DOY single-component, SO8-9 baseline, for direct
#           comparison against A01/C02/C04), then E01 (new SO7-9 + 80:20
#           reference) and G01-G05 (one-flag diffs off E01).

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

# --- ex-series: single DOY component, same domain/weights as A01/C01/C02 ---
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --doy-component sin --run-name ex01_so8-9_lb7_doySinOnly
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --doy-component cos --run-name ex02_so8-9_lb7_doyCosOnly

# --- E01: new candidate reference (SO7-9, raw Q, lb7, Q:T 0.80:0.20) ---
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --run-name E01_so7-9_lb7_qt80-20

# --- G-series: one-flag architecture diffs off E01 ---
# G01: does the graph-smoothness penalty hurt reach-specific Q?
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --lambda-smooth 0 --run-name G01_so7-9_lb7_qt80-20_noSmooth

# G02: sensitivity to coarsened vs. direct-edge graph topology
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --use-direct-edges --run-name G02_so7-9_lb7_qt80-20_directEdges

# G03: does graph message passing help at all? (LSTM-only ablation)
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --no-gat --run-name G03_so7-9_lb7_qt80-20_lstmOnly

# G04: shallower propagation (1 GAT layer)
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 1 --run-name G04_so7-9_lb7_qt80-20_gat1

# G05: deeper propagation (3 GAT layers) -- watch for oversmoothing
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 3 --run-name G05_so7-9_lb7_qt80-20_gat3

echo
echo "Architecture battery complete. Run: python3 SRC/aggregate_results.py"
echo "Per-reach diagnostics for each run are in outputs/<run>/per_comid_diagnostics.csv"
