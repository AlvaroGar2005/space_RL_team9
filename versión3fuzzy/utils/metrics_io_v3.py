"""
utils/metrics_io_v3.py
-----------------------
Save and load V3 training/evaluation metrics using pandas CSV.
"""

import os
import sys
import numpy as np
import pandas as pd

from config import METRICS_CSV_PATH, EVAL_CSV_PATH


def save_training_metrics_v3(metrics: dict,
                               csv_path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    n = len(metrics["rewards"])

    df = pd.DataFrame({
        "episode"      : np.arange(1, n + 1),
        "reward"       : metrics["rewards"],
        "steps"        : metrics["steps"],
        "td_error"     : metrics["td_errors"],
        "epsilon"      : metrics["epsilon"],
        "success"      : metrics["success"].astype(int),
        "term_reason"  : metrics["term_reasons"],
    })
    df.to_csv(csv_path, index=False)
    print(f"[IO] Training metrics saved → {csv_path}")
    return df


def load_training_metrics_v3(csv_path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(f"[IO] Training metrics loaded ← {csv_path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] '{csv_path}' not found.\n  → Train first: python train_v3.py\n")
        sys.exit(1)


def save_evaluation_metrics_v3(rewards, steps, successes,
                                 csv_path: str = EVAL_CSV_PATH) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame({
        "episode" : np.arange(1, len(rewards) + 1),
        "reward"  : rewards,
        "steps"   : steps,
        "success" : successes.astype(int),
    })
    df.to_csv(csv_path, index=False)
    print(f"[IO] Evaluation metrics saved → {csv_path}")
    return df


def load_evaluation_metrics_v3(csv_path: str = EVAL_CSV_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(f"[IO] Evaluation metrics loaded ← {csv_path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] '{csv_path}' not found.\n  → Evaluate first: python evaluate_v3.py\n")
        sys.exit(1)


def training_summary_v3(df: pd.DataFrame, last_n: int = 1000) -> pd.DataFrame:
    tail = df.tail(last_n)
    return pd.DataFrame([{
        "last_n_episodes" : last_n,
        "mean_reward"     : tail["reward"].mean(),
        "std_reward"      : tail["reward"].std(),
        "mean_steps"      : tail["steps"].mean(),
        "mean_td_error"   : tail["td_error"].mean(),
        "success_rate_%"  : tail["success"].mean() * 100,
        "final_epsilon"   : df["epsilon"].iloc[-1],
    }])


def evaluation_summary_v3(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_episodes"     : len(df),
        "mean_reward"    : df["reward"].mean(),
        "std_reward"     : df["reward"].std(),
        "min_reward"     : df["reward"].min(),
        "max_reward"     : df["reward"].max(),
        "mean_steps"     : df["steps"].mean(),
        "success_rate_%" : df["success"].mean() * 100,
    }])
