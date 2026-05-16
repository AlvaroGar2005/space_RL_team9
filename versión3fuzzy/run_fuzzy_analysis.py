"""
run_fuzzy_analysis.py
----------------------
Entry point for the fuzzy logic analysis of SpaceRL V3.

Runs independently from training and evaluation.
Requires only:
    - results/models/qtable_v3.npy  (trained Q-table)

Usage:
    python run_fuzzy_analysis.py

Outputs:
    results/comparisons/fuzzy_analysis.csv
    results/figures/fuzzy_membership_functions.png
    results/figures/fuzzy_mhi_distribution.png
    results/figures/fuzzy_mhi_vs_qvalue.png
    results/figures/fuzzy_rules_example.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import QTABLE_PATH
from agent import QLearningAgent
from fuzzy import (
    decode_all_states,
    qtable_fuzzy_analysis,
    save_fuzzy_analysis,
    plot_membership_functions,
    plot_mhi_distribution,
    plot_mhi_vs_qvalue,
    plot_fuzzy_rules_example,
    evaluate_mission_state,
)


if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceRL V3 — Fuzzy Logic Mission Analysis")
    print("=" * 65)

    # 1. Load trained Q-table
    agent = QLearningAgent()
    agent.load(QTABLE_PATH)

    # 2. Decode all states and compute fuzzy MHI
    print("\n[Fuzzy] Evaluating all 972 states through fuzzy system...")
    df_fuzzy = qtable_fuzzy_analysis(agent.qtable)
    print(f"[Fuzzy] Evaluation complete. Sample:")
    print(df_fuzzy[["state_id", "energy", "hull", "mhi", "status",
                     "greedy_action", "greedy_qvalue"]].head(10).to_string(index=False))

    # 3. Save CSV
    save_fuzzy_analysis(df_fuzzy, path="results/comparisons/fuzzy_analysis.csv")

    # 4. Summary statistics
    print("\n[Fuzzy] Mission Health Index summary:")
    print(f"  Mean MHI        : {df_fuzzy['mhi'].mean():.4f}")
    print(f"  Min MHI         : {df_fuzzy['mhi'].min():.4f}")
    print(f"  Max MHI         : {df_fuzzy['mhi'].max():.4f}")
    print(f"  States OPTIMAL  : {(df_fuzzy['status']=='OPTIMAL').sum()}")
    print(f"  States STABLE   : {(df_fuzzy['status']=='STABLE').sum()}")
    print(f"  States WARNING  : {(df_fuzzy['status']=='WARNING').sum()}")
    print(f"  States CRITICAL : {(df_fuzzy['status']=='CRITICAL').sum()}")
    print(f"  Appropriate actions (CRITICAL states): "
          f"{df_fuzzy[df_fuzzy['status']=='CRITICAL']['appropriate_action'].mean()*100:.1f}%")

    # 5. Manual example — show fuzzy evaluation for 3 key states
    print("\n[Fuzzy] Example evaluations:")
    for vals, label in [
        ((0, 0, 0, 0, 0), "All resources CRITICAL"),
        ((1, 1, 1, 1, 1), "All resources STABLE"),
        ((2, 2, 2, 2, 2), "All resources OPTIMAL"),
        ((0, 2, 1, 1, 2), "Energy LOW, rest OK"),
    ]:
        r = evaluate_mission_state(*vals)
        print(f"  {label:<35} → MHI={r['mhi']:.3f}  [{r['status']}]"
              f"  risk={r['crisp_risk']:.3f}")

    # 6. Generate figures
    print("\n[Fuzzy] Generating figures...")
    plot_membership_functions(save_path="results/figures/fuzzy_membership_functions.png")
    plot_mhi_distribution(df_fuzzy, save_path="results/figures/fuzzy_mhi_distribution.png")
    plot_mhi_vs_qvalue(df_fuzzy,    save_path="results/figures/fuzzy_mhi_vs_qvalue.png")
    plot_fuzzy_rules_example(       save_path="results/figures/fuzzy_rules_example.png")

    print("\n[DONE] Fuzzy analysis complete.")
    print("  CSV     : results/comparisons/fuzzy_analysis.csv")
    print("  Figures : results/figures/fuzzy_*.png")
