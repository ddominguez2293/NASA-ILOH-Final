#!/usr/bin/env bash
# run_phase1_experiments.sh
#
# Purpose : chain the Phase I local-screening experiments (NASA-ILOH Modeling
#           Prescription, Oct 2026) from the repo root, so each run's own
#           outputs/<run>/results.txt lands on disk before the next starts.
#           Stops at the first failing run (set -e) rather than silently
#           skipping ahead on a broken config.
# Usage   : bash SRC/run_phase1_experiments.sh
#
# Runs    : A01-A04 (stream-order x Q-transform), B02-B03 (Q:T weighting),
#           C02 + C04 (DOY ablation), D02 (uniform high-flow weighting).
#           A01==B01==C01==D01 by construction (all defaults), so it is only
#           run once, as A01; B03==C03, so only run once, as B03.

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

# --- Experiment A: stream-order x Q-transform sensitivity ---
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --run-name A01_so8-9_lb7_rawQ
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --log-scale-q --run-name A02_so8-9_lb7_logQ
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --run-name A03_so7-9_lb7_rawQ
run --streamorder-min 7 --streamorder-max 9 --lookback 7 --log-scale-q --run-name A04_so7-9_lb7_logQ

# --- Experiment B: multitask Q:T loss-weight sensitivity (SO8-9, lb7) ---
# B01 reuses A01 (identical config).
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --q-weight 0.80 --t-weight 0.20 --run-name B02_so8-9_lb7_qt80-20
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --q-weight 0.85 --t-weight 0.15 --run-name B03_so8-9_lb7_qt85-15

# --- Experiment C: seasonal (DOY) feature ablation (SO8-9, lb7) ---
# C01 reuses A01 (DOY included, 0.70:0.30); C03 reuses B03 (DOY included, 0.85:0.15).
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --no-include-doy --run-name C02_so8-9_lb7_noDOY_qt70-30
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --no-include-doy --q-weight 0.85 --t-weight 0.15 --run-name C04_so8-9_lb7_noDOY_qt85-15

# --- Experiment D: high-flow discharge weighting (SO8-9, lb7) ---
# D01 reuses A01 (max_mult=5.0, thresh=100 m3/s -- the current default).
run --streamorder-min 8 --streamorder-max 9 --lookback 7 --q-high-max-mult 1.0 --run-name D02_so8-9_lb7_hfm1.0

echo
echo "Phase I battery complete. Run: python3 SRC/aggregate_results.py"
