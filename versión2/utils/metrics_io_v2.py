"""
utils/metrics_io_v2.py
----------------------
Save and load V2 training/evaluation metrics.

Supports:
    - .npz  (numpy binary, fast)
    - .csv  (pandas, human-readable, used in notebook)

V2 adds wrapper-specific columns to every CSV, making the files
directly useful for comparing V1 vs V2 and for analysing wrapper impact.
"""

import os
import sys
import numpy as np
import pandas as pd

from config import METRICS_NPZ_PATH, METRICS_CSV_PATH, EVAL_CSV_PATH


# ── Training ──────────────────────────────────────────────────────────────────

def save_training_metrics_v2(metrics: dict,
                              npz_path: str = METRICS_NPZ_PATH,
                              csv_path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    """
    Save all V2 per-episode training metrics to both .npz and .csv.

    Args:
        metrics : dict of numpy arrays returned by run_training_v2()
        npz_path: binary output path
        csv_path: CSV output path

    Returns:
        DataFrame with all metrics (one row per episode).
    """
    n = len(metrics["rewards"])

    # ── NPZ ──
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **metrics)
    print(f"[IO] Training metrics saved (npz) → {npz_path}")

    # ── CSV ──
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame({
        "episode"              : np.arange(1, n + 1),
        "reward"               : metrics["rewards"],
        "steps"                : metrics["steps"],
        "td_error"             : metrics["td_errors"],
        "epsilon"              : metrics["epsilon"],
        "success"              : metrics["success"].astype(int),
        "invalid_actions"      : metrics["invalid_actions"],
        "wrapper_penalty"      : metrics["wrapper_penalty"],
        "wrapper_bonus"        : metrics["wrapper_bonus"],
        "blocked_actions"      : metrics["blocked_actions"],
        "transformed_actions"  : metrics["transformed_actions"],
        "risky_action_ratio"   : metrics["risky_action_ratio"],
        "useful_actions_ratio" : metrics["useful_actions_ratio"],
    })
    df.to_csv(csv_path, index=False)
    print(f"[IO] Training metrics saved (csv) → {csv_path}")
    return df


def load_training_metrics_v2(csv_path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    """Load V2 training metrics from CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"[IO] Training metrics loaded ← {csv_path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] '{csv_path}' not found.\n  → Train first: python train_taxi_v2.py\n")
        sys.exit(1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def save_evaluation_metrics_v2(rewards, steps, successes,
                                 invalid_arr, penalty_arr, bonus_arr,
                                 blocked_arr, useful_arr,
                                 csv_path: str = EVAL_CSV_PATH) -> pd.DataFrame:
    """Save per-episode evaluation metrics to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame({
        "episode"             : np.arange(1, len(rewards) + 1),
        "reward"              : rewards,
        "steps"               : steps,
        "success"             : successes.astype(int),
        "invalid_actions"     : invalid_arr,
        "wrapper_penalty"     : penalty_arr,
        "wrapper_bonus"       : bonus_arr,
        "blocked_actions"     : blocked_arr,
        "useful_actions_ratio": useful_arr,
    })
    df.to_csv(csv_path, index=False)
    print(f"[IO] Evaluation metrics saved (csv) → {csv_path}")
    return df


def load_evaluation_metrics_v2(csv_path: str = EVAL_CSV_PATH) -> pd.DataFrame:
    """Load V2 evaluation metrics from CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"[IO] Evaluation metrics loaded ← {csv_path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] '{csv_path}' not found.\n  → Evaluate first: python evaluate_taxi_v2.py\n")
        sys.exit(1)


# ── Summary helpers ───────────────────────────────────────────────────────────

def training_summary_v2(df: pd.DataFrame, last_n: int = 500) -> pd.DataFrame:
    tail = df.tail(last_n)
    return pd.DataFrame([{
        "last_n_episodes"       : last_n,
        "mean_reward"           : tail["reward"].mean(),
        "std_reward"            : tail["reward"].std(),
        "mean_steps"            : tail["steps"].mean(),
        "mean_td_error"         : tail["td_error"].mean(),
        "success_rate_%"        : tail["success"].mean() * 100,
        "mean_invalid_actions"  : tail["invalid_actions"].mean(),
        "mean_wrapper_penalty"  : tail["wrapper_penalty"].mean(),
        "mean_wrapper_bonus"    : tail["wrapper_bonus"].mean(),
        "mean_useful_ratio"     : tail["useful_actions_ratio"].mean(),
        "final_epsilon"         : df["epsilon"].iloc[-1],
    }])


def evaluation_summary_v2(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_episodes"            : len(df),
        "mean_reward"           : df["reward"].mean(),
        "std_reward"            : df["reward"].std(),
        "min_reward"            : df["reward"].min(),
        "max_reward"            : df["reward"].max(),
        "mean_steps"            : df["steps"].mean(),
        "success_rate_%"        : df["success"].mean() * 100,
        "mean_invalid_actions"  : df["invalid_actions"].mean(),
        "mean_wrapper_penalty"  : df["wrapper_penalty"].mean(),
        "mean_useful_ratio"     : df["useful_actions_ratio"].mean(),
    }])
