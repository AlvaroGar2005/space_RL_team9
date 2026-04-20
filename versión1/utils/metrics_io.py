"""
utils/metrics_io.py
-------------------
Handles saving and loading of training and evaluation metrics.

Supports two formats in parallel:
    - .npz  (numpy archive)  → fast binary, used internally
    - .csv  (pandas)         → human-readable, shareable, used in notebooks

Both files are written on every save so the project always has both formats
available without extra steps.
"""

import os
import numpy as np
import pandas as pd

from config import METRICS_NPZ_PATH, METRICS_CSV_PATH, EVAL_CSV_PATH


#Training metrics

def save_training_metrics(
    rewards: np.ndarray,
    steps: np.ndarray,
    td_errors: np.ndarray,
    epsilon_history: np.ndarray,
    npz_path: str = METRICS_NPZ_PATH,
    csv_path: str = METRICS_CSV_PATH,
):
    """
    Persist per-episode training metrics to both .npz and .csv.

    Args:
        rewards         : total reward per episode
        steps           : number of steps per episode
        td_errors       : mean absolute TD error per episode
        epsilon_history : epsilon value at the end of each episode
        npz_path        : output path for the numpy archive
        csv_path        : output path for the CSV file
    """
    #Numpy archive
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(
        npz_path,
        rewards=rewards,
        steps=steps,
        td_errors=td_errors,
        epsilon=epsilon_history,
    )
    print(f"[IO] Training metrics saved (npz) → {npz_path}")

    #CSV via pandas
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(
        {
            "episode"   : np.arange(1, len(rewards) + 1),
            "reward"    : rewards,
            "steps"     : steps,
            "td_error"  : td_errors,
            "epsilon"   : epsilon_history,
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"[IO] Training metrics saved (csv) → {csv_path}")
    return df


def load_training_metrics(
    npz_path: str = METRICS_NPZ_PATH,
    csv_path: str = METRICS_CSV_PATH,
    prefer_csv: bool = False,
) -> pd.DataFrame:
    """
    Load training metrics. Returns a pandas DataFrame.

    By default uses the .npz file (faster) and wraps it in a DataFrame.
    Pass prefer_csv=True to load directly from the CSV.
    """
    import sys

    if prefer_csv:
        try:
            df = pd.read_csv(csv_path)
            print(f"[IO] Training metrics loaded (csv) ← {csv_path}  ({len(df)} episodes)")
            return df
        except FileNotFoundError:
            pass  # fall through to npz

    try:
        data = np.load(npz_path)
        df = pd.DataFrame(
            {
                "episode"  : np.arange(1, len(data["rewards"]) + 1),
                "reward"   : data["rewards"],
                "steps"    : data["steps"],
                "td_error" : data["td_errors"],
                "epsilon"  : data["epsilon"] if "epsilon" in data else np.zeros(len(data["rewards"])),
            }
        )
        print(f"[IO] Training metrics loaded (npz) ← {npz_path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(
            f"\n[ERROR] Metrics file not found: '{npz_path}'\n"
            "  → Train first:  python train_taxi_qlearning.py\n"
        )
        sys.exit(1)


#Evaluation metrics

def save_evaluation_metrics(
    rewards: np.ndarray,
    steps: np.ndarray,
    successes: np.ndarray,
    csv_path: str = EVAL_CSV_PATH,
) -> pd.DataFrame:
    """
    Persist per-episode evaluation metrics to CSV.

    Returns the DataFrame for further analysis in notebooks.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(
        {
            "episode" : np.arange(1, len(rewards) + 1),
            "reward"  : rewards,
            "steps"   : steps,
            "success" : successes.astype(int),
        }
    )
    df.to_csv(csv_path, index=False)
    print(f"[IO] Evaluation metrics saved (csv) → {csv_path}")
    return df


def load_evaluation_metrics(csv_path: str = EVAL_CSV_PATH) -> pd.DataFrame:
    """Load evaluation metrics from CSV."""
    import sys
    try:
        df = pd.read_csv(csv_path)
        print(f"[IO] Evaluation metrics loaded ← {csv_path}  ({len(df)} episodes)")
        return df
    except FileNotFoundError:
        print(
            f"\n[ERROR] Evaluation CSV not found: '{csv_path}'\n"
            "  → Evaluate first:  python evaluate_taxi_qlearning.py\n"
        )
        sys.exit(1)


#Summary statistics

def training_summary(df: pd.DataFrame, last_n: int = 500) -> pd.DataFrame:
    """
    Compute a summary of training performance over the last N episodes.
    Returns a one-row DataFrame, useful for notebooks and reports.
    """
    tail = df.tail(last_n)
    summary = pd.DataFrame(
        [
            {
                "last_n_episodes" : last_n,
                "mean_reward"     : tail["reward"].mean(),
                "std_reward"      : tail["reward"].std(),
                "mean_steps"      : tail["steps"].mean(),
                "mean_td_error"   : tail["td_error"].mean(),
                "final_epsilon"   : df["epsilon"].iloc[-1],
            }
        ]
    )
    return summary


def evaluation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a summary of evaluation performance.
    Returns a one-row DataFrame.
    """
    summary = pd.DataFrame(
        [
            {
                "n_episodes"    : len(df),
                "mean_reward"   : df["reward"].mean(),
                "std_reward"    : df["reward"].std(),
                "min_reward"    : df["reward"].min(),
                "max_reward"    : df["reward"].max(),
                "mean_steps"    : df["steps"].mean(),
                "success_rate"  : df["success"].mean() * 100,
            }
        ]
    )
    return summary