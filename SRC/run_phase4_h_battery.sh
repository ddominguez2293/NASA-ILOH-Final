#!/usr/bin/env bash
# run_phase4_h_battery.sh
#
# Purpose : Phase II testing battery (Oct 2026) built on G05 (SO7-9, lb7, raw
#           Q, Q:T 0.80:0.20, 3 GAT layers -- the strongest config from the
#           architecture-diagnostic battery). Tests longer temporal memory,
#           single-DOY-component Q sensitivity, deeper message passing, seed
#           reproducibility, and lower-order network expansion.
#           Stops at the first failing run (set -e).
# Usage   : bash SRC/run_phase4_h_battery.sh
#
# Runs    : H01 (lb30), H02 (doy-sin-only), H03 (4 GAT layers),
#           H07-H08 (seeds 1-2, reproducibility), H05 (SO6-9 expansion).
#           H04 (5 GAT layers, conditional on H03), H06 (SO5-9, conditional
#           on H05), and H09 (final LSTM-vs-GAT comparison, depends on which
#           config wins here) are intentionally NOT auto-launched -- those
#           are scope/cost decisions to make after seeing these results.

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

# --- H01: G05 config with 30-day lookback (longer temporal memory) ---
run --streamorder-min 7 --streamorder-max 9 --lookback 30 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 3 --run-name H01_so7-9_lb30_gat3

# --- H02: G05, DOY sine-only (does dropping cosine help Q, as seen at SO8-9 in ex01?) ---
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 3 --doy-component sin --run-name H02_so7-9_lb7_gat3_doySinOnly

# --- H03: G05, 4 GAT layers (does the G04->G05 depth trend continue?) ---
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 4 --run-name H03_so7-9_lb7_gat4

# --- H07-H08: G05, alternate seeds (is G05's strong result reproducible?) ---
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 3 --seed 1 --run-name H07_so7-9_lb7_gat3_seed1
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 3 --seed 2 --run-name H08_so7-9_lb7_gat3_seed2

# --- H05: G05, expand to SO6-9 (does adding SO6 help SO7-9 skill, as SO7-9 helped SO8-9?) ---
run --streamorder-min 6 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 \
    --gat-layers 3 --run-name H05_so6-9_lb7_gat3

echo
echo "H-battery (fixed portion) complete. Run: python3 SRC/aggregate_results.py"
echo "Remaining decisions: H04 (5 GAT layers, conditional on H03), H06 (SO5-9, conditional on H05),"
echo "and H09 (final LSTM-only vs GAT comparison, config TBD from these results) were intentionally skipped."
