#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# aggregate_results.py
#
# Purpose : collect every outputs/<run>/results.txt (+ test_metrics_by_streamorder.csv)
#           written by DDNN_Model.py into two comparison tables, so repeat
#           experiments (A01-A04, B01-B03, C01-C04, D01-D02, ...) can be
#           compared side by side without opening each run's results.txt by hand.
# Inputs  : outputs/<run>/results.txt, outputs/<run>/test_metrics_by_streamorder.csv
# Outputs : outputs/experiment_comparison.csv              (one row per run)
#           outputs/experiment_comparison_by_streamorder.csv (one row per run x streamorder x variable)
#
# Usage   : python SRC/aggregate_results.py
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code

## 1. Imports
import re
from pathlib import Path
from typing import Optional

import pandas as pd

## 2. Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

## 3. Parsing helpers
#### results.txt is written by DDNN_Model.py as a mix of:
####   - "KEY: value" lines (test variables, run diagnostics, measured/interpolated counts)
####   - "=== Section header ===" lines
####   - "Q    -> NSE: .. | RMSE: .. | MAE: .. | Bias: .. | n_obs=.." metric lines,
####     which repeat under both the full-domain and common-SO-subset sections,
####     so the active section header disambiguates which block a metric line belongs to.
KV_RE = re.compile(r"^([A-Z][A-Z0-9_]*):\s*(.*)$")
METRIC_RE = re.compile(
    r"^(Q|Temp)\s*-> NSE:\s*(-?[\d.]+|nan)\s*\|\s*RMSE:\s*(-?[\d.]+|nan)\s*\|\s*"
    r"MAE:\s*(-?[\d.]+|nan)\s*\|\s*Bias:\s*(-?[\d.]+|nan)\s*\|\s*n_obs=(\d+)"
)

### Which "=== ... ===" section a metric line belongs to
def _section_tag(section_header: Optional[str]) -> Optional[str]:
    if section_header is None:
        return None
    if section_header.startswith("Test Metrics on common SO"):
        return "common_so"
    if section_header.startswith("Test Metrics (original units"):
        return "full_domain"
    return None


def _to_float(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


### Parse one run's results.txt into a flat dict (one row of the comparison table)
def parse_results_txt(path: Path) -> dict:
    row = {}
    section = None

    with open(path) as f:
        lines = f.readlines()

    if lines:
        m = re.match(r"^Run:\s*(.*)$", lines[0].strip())
        if m:
            row["run_name"] = m.group(1)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("===") and line.endswith("==="):
            section = line.strip("= ").strip()
            continue

        kv = KV_RE.match(line)
        if kv:
            key, val = kv.groups()
            # don't clobber a later same-named key with an earlier one (shouldn't happen, but be safe)
            row.setdefault(key, val)
            continue

        mm = METRIC_RE.match(line)
        if mm:
            var, nse, rmse, mae, bias, n_obs = mm.groups()
            prefix = "Q" if var == "Q" else "T"
            tag = _section_tag(section) or "other"
            row[f"{tag}_{prefix}_NSE"]   = _to_float(nse)
            row[f"{tag}_{prefix}_RMSE"]  = _to_float(rmse)
            row[f"{tag}_{prefix}_MAE"]   = _to_float(mae)
            row[f"{tag}_{prefix}_Bias"]  = _to_float(bias)
            row[f"{tag}_{prefix}_n_obs"] = int(n_obs)

    return row


## 4. Collect every run under outputs/
def main():
    rows = []
    so_frames = []

    run_dirs = sorted(p for p in OUTPUTS_DIR.iterdir() if p.is_dir())

    for run_dir in run_dirs:
        results_path = run_dir / "results.txt"
        if not results_path.exists():
            continue

        row = parse_results_txt(results_path)
        row["run_dir"] = run_dir.name
        row.setdefault("run_name", run_dir.name)
        rows.append(row)

        so_csv = run_dir / "test_metrics_by_streamorder.csv"
        if so_csv.exists():
            df_so = pd.read_csv(so_csv)
            df_so.insert(0, "run_dir", run_dir.name)
            df_so.insert(1, "run_name", row.get("run_name", run_dir.name))
            so_frames.append(df_so)

    if not rows:
        print(f"No outputs/*/results.txt found under {OUTPUTS_DIR}. Nothing to aggregate.")
        return

    ### Put the most comparison-relevant columns first; leave everything else
    #### (full test-variable dump) trailing so the CSV is still complete.
    comparison_df = pd.DataFrame(rows)
    priority_cols = [
        "run_name", "run_dir", "Timestamp",
        "STREAMORDER_MIN", "STREAMORDER_MAX", "LOOKBACK", "SEED",
        "USE_LOG_SCALE_Q", "INCLUDE_DOY", "DOY_COMPONENT", "Q_WEIGHT", "T_WEIGHT",
        "Q_HIGH_THRESH_M3S", "Q_HIGH_MAX_MULT",
        "USE_COARSENED_GRAPH", "LAMBDA_SMOOTH", "USE_GAT", "N_GAT_LAYERS",
        "full_domain_Q_NSE", "full_domain_Q_RMSE", "full_domain_Q_MAE", "full_domain_Q_Bias", "full_domain_Q_n_obs",
        "full_domain_T_NSE", "full_domain_T_RMSE", "full_domain_T_MAE", "full_domain_T_Bias", "full_domain_T_n_obs",
        "common_so_Q_NSE", "common_so_Q_RMSE", "common_so_Q_MAE", "common_so_Q_Bias", "common_so_Q_n_obs",
        "common_so_T_NSE", "common_so_T_RMSE", "common_so_T_MAE", "common_so_T_Bias", "common_so_T_n_obs",
        "TRAINABLE_PARAMS", "TRAIN_ELAPSED_SEC", "FINAL_EPOCH", "BEST_EPOCH", "BEST_VAL_LOSS",
        "N_REACHES_WITH_Q_TEST", "N_REACHES_WITH_T_TEST",
        "Q_N_ANY_TEST", "Q_N_MEASURED_TEST", "Q_N_INTERPOLATED_TEST", "T_N_MEASURED_TEST",
        "Q_MEDIAN_PERGAUGE_NSE", "T_MEDIAN_PERGAUGE_NSE",
        "N_GAUGES_Q_ELIGIBLE", "N_GAUGES_T_ELIGIBLE", "AREA_LOG1P_VS_Q_NSE_CORR",
    ]
    ordered_cols = [c for c in priority_cols if c in comparison_df.columns]
    remaining_cols = [c for c in comparison_df.columns if c not in ordered_cols]
    comparison_df = comparison_df[ordered_cols + remaining_cols]
    comparison_df = comparison_df.sort_values("run_name", na_position="last")

    comparison_path = OUTPUTS_DIR / "experiment_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Wrote {comparison_path} ({len(comparison_df)} runs)")

    if so_frames:
        so_df = pd.concat(so_frames, ignore_index=True)
        so_path = OUTPUTS_DIR / "experiment_comparison_by_streamorder.csv"
        so_df.to_csv(so_path, index=False)
        print(f"Wrote {so_path} ({len(so_df)} rows)")


if __name__ == "__main__":
    main()
