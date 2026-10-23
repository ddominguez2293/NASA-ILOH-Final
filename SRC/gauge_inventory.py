#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gauge_inventory.py
#
# Purpose : answer "how much observational data do we actually have?" for the
#           NASA-ILOH paper's methods section. Distinguishes total modeled
#           reaches, gauged reaches (>=1 real measurement), and observation
#           records (individual measured values) by stream order -- for the
#           full 2013+ history and for the train/val/test periods separately,
#           so the severity of sparse supervision is documented precisely
#           rather than treated as a preprocessing footnote.
# Inputs  : hydro_temp_data_trainval_2013plus.parquet (NODES_PATH)
# Outputs : outputs/gauge_inventory.csv (one row per period x stream order)
#
# Note    : QC/repair logic (sentinel removal, negative-Q interpolation
#           repair, temperature plausibility range) mirrors DDNN_Model.py
#           Sections 3-4 exactly, so "measured" means the same thing in both
#           scripts. DDNN_Model.py can't be imported directly (it's a CLI
#           script that parses argv and runs training at import time), so
#           the small set of needed helpers is duplicated here deliberately.
#
# Usage   : python SRC/gauge_inventory.py
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code

## 1. Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd

## 2. Paths and constants (mirrors DDNN_Model.py Section 2 defaults)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = PROJECT_ROOT / "data" / "daily_drivers" / "hydro_temp_data_trainval_2013plus.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

Q_COL = "q_m3s_obs"
T_COL = "water_temp_C"
SENTINEL_MISS = -9998.0
TEMP_QC_MIN_C, TEMP_QC_MAX_C = -5.0, 60.0

SPLIT_DATE = pd.Timestamp("2025-01-01")
VAL_DAYS = 180
VAL_START_DATE = SPLIT_DATE - pd.Timedelta(days=VAL_DAYS)


## 3. QC helpers (copied from DDNN_Model.py Section 3/4 -- keep in sync)
def replace_sentinel_with_nan(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            df.loc[x == SENTINEL_MISS, c] = np.nan
    return df


def interpolate_negative_q_by_comid(df, q_col, date_col="date", comid_col="comid",
                                     q_floor=0.0, flag_col="q_was_interpolated"):
    out = df.copy()
    out = out.sort_values([comid_col, date_col]).reset_index(drop=True)
    out[q_col] = pd.to_numeric(out[q_col], errors="coerce")
    neg_mask_before = out[q_col].notna() & (out[q_col] < q_floor)
    out.loc[neg_mask_before, q_col] = np.nan
    missing_before_interp = out[q_col].isna()
    out[q_col] = (
        out.groupby(comid_col, group_keys=False)[q_col]
           .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
    )
    out[flag_col] = missing_before_interp & out[q_col].notna()
    out[q_col] = out[q_col].clip(lower=0.0)
    return out


## 4. Load + QC (same steps as DDNN_Model.py Section 4, full SO5-9 domain)
print("Loading:", NODES_PATH)
nodes = pd.read_parquet(NODES_PATH, columns=["comid", "date", "streamorde", Q_COL, T_COL])
nodes["date"] = pd.to_datetime(nodes["date"])
nodes["streamorde"] = pd.to_numeric(nodes["streamorde"], errors="coerce")
print(f"Rows loaded: {len(nodes):,}  |  unique COMIDs: {nodes['comid'].nunique():,}")

nodes = replace_sentinel_with_nan(nodes, [Q_COL, T_COL])
nodes = interpolate_negative_q_by_comid(nodes, q_col=Q_COL, date_col="date", comid_col="comid", q_floor=0.0)

t_raw = pd.to_numeric(nodes[T_COL], errors="coerce")
bad_t = t_raw.notna() & ((t_raw < TEMP_QC_MIN_C) | (t_raw > TEMP_QC_MAX_C))
nodes.loc[bad_t, T_COL] = np.nan

nodes["q_avail_any"] = ~nodes[Q_COL].isna()
nodes["q_avail_measured"] = nodes["q_avail_any"] & ~nodes["q_was_interpolated"]
nodes["t_avail_measured"] = ~nodes[T_COL].isna()


## 5. Per-period, per-stream-order coverage table
def summarize_period(df_period: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for so in sorted(df_period["streamorde"].dropna().unique()):
        sub = df_period[df_period["streamorde"] == so]
        total_comids = sub["comid"].nunique()

        q_gauged_comids = set(sub.loc[sub["q_avail_measured"], "comid"].unique())
        t_gauged_comids = set(sub.loc[sub["t_avail_measured"], "comid"].unique())
        both_comids = q_gauged_comids & t_gauged_comids

        n_q_gauged = len(q_gauged_comids)
        n_t_gauged = len(t_gauged_comids)

        total_q_records = int(sub["q_avail_measured"].sum())
        total_t_records = int(sub["t_avail_measured"].sum())

        if n_q_gauged > 0:
            median_q_days = (
                sub.loc[sub["comid"].isin(q_gauged_comids) & sub["q_avail_measured"], "comid"]
                   .value_counts().median()
            )
        else:
            median_q_days = np.nan

        if n_t_gauged > 0:
            median_t_days = (
                sub.loc[sub["comid"].isin(t_gauged_comids) & sub["t_avail_measured"], "comid"]
                   .value_counts().median()
            )
        else:
            median_t_days = np.nan

        rows.append({
            "streamorde": int(so),
            "total_comids": total_comids,
            "q_gauged_reaches": n_q_gauged,
            "t_gauged_reaches": n_t_gauged,
            "q_ungauged_reaches": total_comids - n_q_gauged,
            "pct_reaches_q_gauged": round(100.0 * n_q_gauged / total_comids, 2) if total_comids else np.nan,
            "pct_reaches_t_gauged": round(100.0 * n_t_gauged / total_comids, 2) if total_comids else np.nan,
            "total_q_records_measured": total_q_records,
            "total_t_records_measured": total_t_records,
            "median_measured_days_per_q_gauge": median_q_days,
            "median_measured_days_per_t_gauge": median_t_days,
            "reaches_with_both_q_and_t": len(both_comids),
        })
    return pd.DataFrame(rows)


periods = {
    "full_history_2013plus": nodes,
    "train": nodes[nodes["date"] < VAL_START_DATE],
    "val": nodes[(nodes["date"] >= VAL_START_DATE) & (nodes["date"] < SPLIT_DATE)],
    "test": nodes[nodes["date"] >= SPLIT_DATE],
}

all_rows = []
for period_name, df_period in periods.items():
    summary = summarize_period(df_period)
    summary.insert(0, "period", period_name)
    all_rows.append(summary)
    print(f"\n=== {period_name} ({df_period['date'].min().date()} .. {df_period['date'].max().date()}) ===")
    print(summary.to_string(index=False))

inventory_df = pd.concat(all_rows, ignore_index=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
out_path = OUTPUTS_DIR / "gauge_inventory.csv"
inventory_df.to_csv(out_path, index=False)
print(f"\nWrote {out_path} ({len(inventory_df)} rows)")
