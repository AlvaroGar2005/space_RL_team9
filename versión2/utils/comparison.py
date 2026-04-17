"""
utils/comparison.py
-------------------
Generates V1 vs V2 comparative analysis CSV and summary tables.

This is the main academic contribution of V2:
    - Load metrics from both versions
    - Align them by episode
    - Compute differences and ratios
    - Save a comparison CSV for notebook analysis
"""

import os
import sys
import numpy as np
import pandas as pd

from config import V1_METRICS_CSV, V1_EVAL_CSV, METRICS_CSV_PATH, EVAL_CSV_PATH, COMPARISON_CSV_PATH


def load_v1_metrics(path: str = V1_METRICS_CSV) -> pd.DataFrame:
    """Load V1 training metrics CSV."""
    try:
        df = pd.read_csv(path)
        # Normalise column names: V1 may use 'td_errors' or 'td_error'
        df = df.rename(columns={"td_errors": "td_error"})
        print(f"[Compare] V1 metrics loaded ← {path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"[WARN] V1 metrics not found at '{path}'. Run V1 training first for full comparison.")
        return None


def build_comparison_csv(df_v1: pd.DataFrame,
                          df_v2: pd.DataFrame,
                          out_path: str = COMPARISON_CSV_PATH,
                          window: int = 200) -> pd.DataFrame:
    """
    Build a comparison DataFrame with smoothed metrics for both versions.
    Only uses columns present in both DataFrames.

    Saves result to out_path and returns the DataFrame.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    common_cols = ["reward", "steps", "td_error", "epsilon"]
    # Add success if available in both
    if "success" in df_v1.columns and "success" in df_v2.columns:
        common_cols.append("success")

    # Align to the shorter dataset
    n = min(len(df_v1), len(df_v2))
    episodes = np.arange(1, n + 1)

    rows = {"episode": episodes}
    for col in common_cols:
        if col in df_v1.columns and col in df_v2.columns:
            v1_smooth = _smooth(df_v1[col].values[:n], window)
            v2_smooth = _smooth(df_v2[col].values[:n], window)
            rows[f"v1_{col}"]   = v1_smooth
            rows[f"v2_{col}"]   = v2_smooth
            rows[f"delta_{col}"] = v2_smooth - v1_smooth

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(out_path, index=False)
    print(f"[Compare] Comparison CSV saved → {out_path}")
    return df_cmp


def comparison_summary(df_v1: pd.DataFrame,
                        df_v2: pd.DataFrame,
                        last_n: int = 500) -> pd.DataFrame:
    """
    Build a one-table summary comparing the last N episodes of V1 and V2.
    """
    def _tail_stats(df, col):
        s = df[col].tail(last_n)
        return s.mean(), s.std()

    rows = []
    for col, label in [("reward", "Mean Reward"), ("steps", "Mean Steps"),
                        ("td_error", "Mean TD Error")]:
        v1_m, v1_s = _tail_stats(df_v1, col) if col in df_v1.columns else (float("nan"), float("nan"))
        v2_m, v2_s = _tail_stats(df_v2, col) if col in df_v2.columns else (float("nan"), float("nan"))
        rows.append({
            "Metric"   : label,
            "V1 mean"  : round(v1_m, 3),
            "V1 std"   : round(v1_s, 3),
            "V2 mean"  : round(v2_m, 3),
            "V2 std"   : round(v2_s, 3),
            "Δ (V2−V1)": round(v2_m - v1_m, 3),
        })

    # Success rate
    if "success" in df_v1.columns and "success" in df_v2.columns:
        v1_sr = df_v1["success"].tail(last_n).mean() * 100
        v2_sr = df_v2["success"].tail(last_n).mean() * 100
        rows.append({
            "Metric"   : "Success Rate (%)",
            "V1 mean"  : round(v1_sr, 1),
            "V1 std"   : float("nan"),
            "V2 mean"  : round(v2_sr, 1),
            "V2 std"   : float("nan"),
            "Δ (V2−V1)": round(v2_sr - v1_sr, 1),
        })

    return pd.DataFrame(rows)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    kernel = np.ones(window) / window
    smooth = np.convolve(values, kernel, mode="valid")
    pad_l = (len(values) - len(smooth)) // 2
    pad_r = len(values) - len(smooth) - pad_l
    return np.pad(smooth, (pad_l, pad_r), mode="edge")
