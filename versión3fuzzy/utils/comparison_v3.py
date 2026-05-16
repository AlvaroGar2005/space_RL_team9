"""
utils/comparison_v3.py
-----------------------
Three-way comparison: V1 vs V2 vs V3.
Loads CSVs from all three versions and produces a unified comparison.
"""

import os
import numpy as np
import pandas as pd

from config import V1_METRICS_CSV, V2_METRICS_CSV, COMPARISON_CSV, WINDOW_SIZE


def _smooth(values, window):
    if window >= len(values):
        return np.full_like(values, np.mean(values), dtype=float)
    k = np.ones(window) / window
    s = np.convolve(values, k, mode="valid")
    pl = (len(values) - len(s)) // 2
    pr = len(values) - len(s) - pl
    return np.pad(s, (pl, pr), mode="edge")


def load_version_metrics(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df = df.rename(columns={"td_errors": "td_error"})
        print(f"[Compare] {label} metrics loaded ← {path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"[WARN] {label} metrics not found at '{path}'")
        return None


def build_three_way_comparison(df_v1, df_v2, df_v3,
                                out_path: str = COMPARISON_CSV) -> pd.DataFrame:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = min(
        len(df_v1) if df_v1 is not None else 99999,
        len(df_v2) if df_v2 is not None else 99999,
        len(df_v3),
    )
    episodes = np.arange(1, n + 1)

    rows = {"episode": episodes}
    for col in ["reward", "steps"]:
        if df_v1 is not None and col in df_v1.columns:
            rows[f"v1_{col}"] = _smooth(df_v1[col].values[:n], WINDOW_SIZE)
        if df_v2 is not None and col in df_v2.columns:
            rows[f"v2_{col}"] = _smooth(df_v2[col].values[:n], WINDOW_SIZE)
        rows[f"v3_{col}"] = _smooth(df_v3[col].values[:n], WINDOW_SIZE)

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(out_path, index=False)
    print(f"[Compare] Three-way comparison saved → {out_path}")
    return df_cmp


def three_way_summary(df_v1, df_v2, df_v3, last_n: int = 1000) -> pd.DataFrame:
    rows = []
    for metric, label in [("reward", "Mean Reward"), ("steps", "Mean Steps")]:
        row = {"Metric": label}
        for df, name in [(df_v1, "V1"), (df_v2, "V2"), (df_v3, "V3")]:
            if df is not None and metric in df.columns:
                row[name] = round(df[metric].tail(last_n).mean(), 2)
            else:
                row[name] = "N/A"
        rows.append(row)

    # Success rate
    row = {"Metric": "Success Rate (%)"}
    for df, name in [(df_v1, "V1"), (df_v2, "V2"), (df_v3, "V3")]:
        if df is not None and "success" in df.columns:
            row[name] = round(df["success"].tail(last_n).mean() * 100, 1)
        else:
            row[name] = "N/A"
    rows.append(row)

    return pd.DataFrame(rows)
