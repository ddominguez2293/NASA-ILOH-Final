#!/usr/bin/env bash
# run_h09.sh
#
# Purpose : H09 -- final LSTM-only vs. GAT comparison at 30-day lookback.
#           H01 (SO7-9, lb30, Q:T 0.80:0.20, 3-layer GAT) is the strongest
#           result in the project so far (Q NSE 0.805, SO8 Q NSE 0.392,
#           median per-gauge Q NSE -0.071). H09 reruns the identical config
#           with --no-gat to ask directly: does graph message passing still
#           earn its cost once the model already has 30 days of antecedent
#           memory, or does temporal memory alone explain H01's jump?
#           Queued to start after G06 finishes rather than interrupting it.
# Usage   : bash SRC/run_h09.sh [wait_pid]

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "Waiting for PID $WAIT_PID (G06) to finish before starting H09..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 10
  done
  echo "PID $WAIT_PID finished. Starting H09."
fi

echo
echo "=============================================="
echo "Starting: H09 (30-day lookback, LSTM-only, matched to H01)"
echo "=============================================="
python3 SRC/DDNN_Model.py \
  --streamorder-min 7 --streamorder-max 9 --lookback 30 \
  --q-weight 0.80 --t-weight 0.20 --no-gat \
  --run-name H09_so7-9_lb30_lstmOnly

echo
echo "H09 complete. Run: python3 SRC/aggregate_results.py"
echo "Compare directly against H01_so7-9_lb30_gat3 (same config, with GAT)."
