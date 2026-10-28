#!/usr/bin/env bash
# run_gridmet_and_battery.sh
#
# Purpose : the correct single-script version of what run_after_gridmet.sh tried to do by polling
#           an external PID (which can't observe that process's exit code and silently proceeded
#           past a real crash -- see git history). Runs meteo -> q_processing -> Battery I
#           continuation (I05-I08) as one chain under `set -e`, so a real failure in any step stops
#           everything immediately instead of continuing on stale data.
# Usage   : bash SRC/run_gridmet_and_battery.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Starting: Meteo_gridmet.py (resumes from cached chunks automatically)"
echo "=============================================="
source .venv_pipeline/bin/activate
python3 SRC/Meteo_gridmet.py

echo
echo "=============================================="
echo "Starting: Q_Processing_python.py"
echo "=============================================="
python3 SRC/Q_Processing_python.py
deactivate

echo
echo "Meteo + q_processing complete. Final model inputs rebuilt from GridMET in data/daily_drivers/."
echo "Starting Battery I continuation (I05-I08) on the GridMET-based dataset."
echo

source .venv/bin/activate

run() {
  echo
  echo "=============================================="
  echo "Starting: $*"
  echo "=============================================="
  python3 SRC/DDNN_Model.py "$@"
}

COMMON="--lookback 30 --q-weight 0.80 --t-weight 0.20"
# --doy-component defaults to "both" (sine+cosine), matching historical H01.

# I05 (GridMET): SO7-9, 3-layer GAT, both DOY -- the direct H01-architecture retest. Priority.
run --streamorder-min 7 --streamorder-max 9 $COMMON --gat-layers 3 \
    --run-name I05_gridmet_so7-9_lb30_gat3_bothDOY

# I06 (GridMET): SO7-9, LSTM-only, both DOY, graph smoothness off -- matched benchmark
run --streamorder-min 7 --streamorder-max 9 $COMMON --no-gat --lambda-smooth 0 \
    --run-name I06_gridmet_so7-9_lb30_lstmOnly_bothDOY_noSmooth

# I07 (GridMET): SO6-9, 3-layer GAT, both DOY
run --streamorder-min 6 --streamorder-max 9 $COMMON --gat-layers 3 \
    --run-name I07_gridmet_so6-9_lb30_gat3_bothDOY

# I08 (GridMET): SO6-9, LSTM-only, both DOY, graph smoothness off
run --streamorder-min 6 --streamorder-max 9 $COMMON --no-gat --lambda-smooth 0 \
    --run-name I08_gridmet_so6-9_lb30_lstmOnly_bothDOY_noSmooth

echo
echo "run_gridmet_and_battery.sh complete. Run: python3 SRC/aggregate_results.py"
echo "Compare I05_gridmet/I06_gridmet/I07_gridmet/I08_gridmet against I05/I06/I07/I08 (ERA5-based)"
echo "and against historical H01 (Q NSE 0.805, T NSE 0.877)."
