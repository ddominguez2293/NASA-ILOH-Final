#!/usr/bin/env bash
# run_ex03-ex04_nopet.sh
#
# Purpose : ex03/ex04 -- PET-removal ablation, requested while investigating
#           the gap between historical H01 (0.805, pre-correction pipeline)
#           and I01 (0.573, corrected pipeline + ERA5 meteorology + new
#           features). Matched exactly to I01/I02 otherwise (sin-only DOY,
#           default graph smoothness) so PET inclusion is the only variable
#           that changes relative to those two runs.
# Usage   : bash SRC/run_ex03-ex04_nopet.sh

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

COMMON="--streamorder-min 7 --streamorder-max 9 --lookback 30 --q-weight 0.80 --t-weight 0.20 --doy-component sin --exclude-pet"

# ex03: matched to I01 (3-layer GAT) minus PET
run $COMMON --gat-layers 3 --run-name ex03_so7-9_lb30_gat3_noPET

# ex04: matched to I02 (LSTM-only) minus PET
run $COMMON --no-gat --run-name ex04_so7-9_lb30_lstmOnly_noPET

echo
echo "ex03-ex04 (PET ablation) complete. Run: python3 SRC/aggregate_results.py"
echo "Compare directly against I01 (GAT, with PET) and I02 (LSTM-only, with PET)."
