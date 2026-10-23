#!/usr/bin/env bash
# run_g06.sh
#
# Purpose : G06 -- bidirectional-edge sensitivity test, queued to run right
#           after the current H-battery (run_phase4_h_battery.sh) finishes,
#           without interrupting it. G05 restricts attention to upstream
#           neighbors only (routing direction); G06 also lets downstream
#           context propagate back toward smaller tributaries, to see if
#           that specifically helps the weak SO7/SO8 reaches.
# Usage   : bash SRC/run_g06.sh [wait_pid]
#           wait_pid: optional PID to wait on before starting (defaults to
#           the PID of the currently running H-battery, passed at launch time).

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "Waiting for PID $WAIT_PID (H-battery) to finish before starting G06..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 10
  done
  echo "PID $WAIT_PID finished. Starting G06."
fi

echo
echo "=============================================="
echo "Starting: G06 (bidirectional edges)"
echo "=============================================="
python3 SRC/DDNN_Model.py \
  --streamorder-min 7 --streamorder-max 9 --lookback 7 \
  --q-weight 0.80 --t-weight 0.20 --gat-layers 3 --bidirectional-edges \
  --run-name G06_so7-9_lb7_gat3_bidirEdges

echo
echo "G06 complete. Run: python3 SRC/aggregate_results.py"
