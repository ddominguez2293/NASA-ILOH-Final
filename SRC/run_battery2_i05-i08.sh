#!/usr/bin/env bash
# run_battery2_i05-i08.sh
#
# Purpose : Battery I continuation (I05-I08) -- reproduce H01's architecture
#           (both DOY components, not sine-only) on the corrected pipeline's
#           dataset to isolate whether I01's drop vs. historical H01 came
#           from the DOY ablation, the new predictors/masking/graph fixes,
#           or some mix -- then the matched SO6-9 comparison. Keeps both
#           the GAT and LSTM-only tracks alive scientifically (graph
#           structure has interpretability value beyond raw NSE) rather
#           than eliminating GAT after I01-I04.
#
#           LSTM-only runs (I06, I08) additionally disable the graph
#           smoothness penalty (--lambda-smooth 0): that loss term acts on
#           predicted Q differences across edges regardless of whether the
#           model itself does graph message passing, so --no-gat alone
#           (as used for I02/I04) was not a fully graph-independent
#           benchmark. GAT runs (I05, I07) keep the default lambda=0.001,
#           matching H01's original setup.
# Usage   : bash SRC/run_battery2_i05-i08.sh

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

COMMON="--lookback 30 --q-weight 0.80 --t-weight 0.20"
# --doy-component defaults to "both" (sine+cosine) -- deliberately NOT sin-only here.

# I05: SO7-9, 3-layer GAT, both DOY -- reproduce H01's architecture on the corrected dataset
run --streamorder-min 7 --streamorder-max 9 $COMMON --gat-layers 3 \
    --run-name I05_so7-9_lb30_gat3_bothDOY

# I06: SO7-9, LSTM-only, both DOY, graph smoothness OFF -- genuinely graph-independent benchmark
run --streamorder-min 7 --streamorder-max 9 $COMMON --no-gat --lambda-smooth 0 \
    --run-name I06_so7-9_lb30_lstmOnly_bothDOY_noSmooth

# I07: SO6-9, 3-layer GAT, both DOY
run --streamorder-min 6 --streamorder-max 9 $COMMON --gat-layers 3 \
    --run-name I07_so6-9_lb30_gat3_bothDOY

# I08: SO6-9, LSTM-only, both DOY, graph smoothness OFF
run --streamorder-min 6 --streamorder-max 9 $COMMON --no-gat --lambda-smooth 0 \
    --run-name I08_so6-9_lb30_lstmOnly_bothDOY_noSmooth

echo
echo "Battery I continuation (I05-I08) complete. Run: python3 SRC/aggregate_results.py"
echo "Remaining: I09/I10 (365-day, pending throughput check), I11-I14 (seed replication, after selection)."
