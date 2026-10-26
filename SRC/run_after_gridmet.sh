#!/usr/bin/env bash
# run_after_gridmet.sh
#
# Purpose : queued to run once the GridMET meteo pull finishes -- rebuilds the final model-input
#           parquet (q_processing, now driven by GridMET instead of ERA5-Land) and then reruns the
#           Battery I continuation (I05-I08) against it, so the H01-architecture comparison (I05 in
#           particular -- SO7-9, 3-layer GAT, both DOY, the direct match to the original historical
#           H01 setup) is retested on the meteorology source that reference actually used.
#
#           Run names get a "_gridmet" suffix rather than reusing I05-I08 as-is: those names
#           already hold the ERA5-based results, and DDNN_Model.py overwrites OUTPUTS_DIR
#           unconditionally, so reusing them would destroy exactly the ERA5-vs-GridMET comparison
#           this rerun exists to make.
# Usage   : bash SRC/run_after_gridmet.sh <meteo_pid>

set -euo pipefail
cd "$(dirname "$0")/.."

METEO_PID="${1:-}"
if [ -n "$METEO_PID" ]; then
  echo "Waiting for PID $METEO_PID (Meteo_gridmet.py) to finish..."
  while kill -0 "$METEO_PID" 2>/dev/null; do
    sleep 10
  done
  echo "PID $METEO_PID finished."
fi

echo
echo "=============================================="
echo "Starting: q_processing (rebuild model inputs from GridMET drivers)"
echo "=============================================="
source .venv_pipeline/bin/activate
python3 SRC/Q_Processing_python.py
deactivate

echo
echo "q_processing complete. Final model inputs rebuilt in data/daily_drivers/."
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
echo "run_after_gridmet.sh complete. Run: python3 SRC/aggregate_results.py"
echo "Compare I05_gridmet/I06_gridmet/I07_gridmet/I08_gridmet against I05/I06/I07/I08 (ERA5-based)"
echo "and against historical H01 (Q NSE 0.805, T NSE 0.877)."
