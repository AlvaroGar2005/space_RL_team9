"""
plot_metrics.py
---------------
Entry point for generating training plots from saved metrics.

Usage:
    python plot_metrics.py

Outputs:
    results/figures/training_metrics.png

Also generates a quick evaluation summary if evaluation.csv exists.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from config import FIGURE_PATH, EVAL_CSV_PATH
from evaluation.plot_metrics import load_metrics_df, build_figure
from utils import load_evaluation_metrics, evaluation_summary


if __name__ == "__main__":
    print("=" * 60)
    print("  SpaceRL – Version 1: Taxi-v3  |  Metrics Plot")
    print("=" * 60)

    # Training curves
    df_train = load_metrics_df()
    build_figure(df_train, save_path=FIGURE_PATH, show=True)

    # Evaluation summary (if available)
    try:
        df_eval = load_evaluation_metrics(EVAL_CSV_PATH)
        print("\n[INFO] Evaluation summary from last run:")
        print(evaluation_summary(df_eval).to_string(index=False))
    except SystemExit:
        print("[INFO] No evaluation.csv found — run evaluate_taxi_qlearning.py first.")

    print("\n[DONE]")