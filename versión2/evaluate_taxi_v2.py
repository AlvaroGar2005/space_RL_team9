"""
evaluate_taxi_v2.py
--------------------
Entry point for evaluating the trained V2 Q-learning agent.

Usage:
    python evaluate_taxi_v2.py

To activate visual simulation set RENDER_DEMO = True in config.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import QTABLE_PATH, EVAL_CSV_PATH, N_EVAL_EPS, RENDER_DEMO, DEMO_EPS
from agent import QLearningAgent
from environment import make_env_v2
from evaluation.eval_runner_v2 import evaluate_agent_v2, run_greedy_episode_v2
from utils import save_evaluation_metrics_v2, evaluation_summary_v2


def run_demo(agent: QLearningAgent, n_demo: int = DEMO_EPS):
    """Render n_demo episodes with render_mode='human'."""
    print(f"\n[DEMO] Rendering {n_demo} episode(s) in human mode...")
    env = make_env_v2(render_mode="human")

    for ep in range(n_demo):
        r, s, ok = run_greedy_episode_v2(env, agent)
        status   = "SUCCESS ✓" if ok else "FAILED  ✗"
        print(f"  Demo ep {ep + 1}/{n_demo} | reward: {r:.0f} | steps: {s} | {status}")

    env.close()
    print("[DEMO] Done.")


if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceRL – Version 2: Wrapped Taxi-v3  |  Evaluation")
    print("=" * 65)

    
    env   = make_env_v2()
    agent = QLearningAgent(env)
    env.close()
    agent.load(QTABLE_PATH)

    
    results = evaluate_agent_v2(agent, n_episodes=N_EVAL_EPS)

    
    df_eval = save_evaluation_metrics_v2(
        results["rewards"], results["steps"], results["successes"],
        results["invalid_actions"], results["wrapper_penalty"],
        results["wrapper_bonus"], results["blocked_actions"],
        results["useful_actions_ratio"],
        csv_path=EVAL_CSV_PATH,
    )

    
    summary = evaluation_summary_v2(df_eval)
    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)
    print(summary.to_string(index=False))
    print("=" * 55)

    
    if RENDER_DEMO:
        run_demo(agent)
    else:
        print("\n[TIP] Set RENDER_DEMO = True in config.py to watch the agent play.")

    print(f"\n[DONE] Results saved → {EVAL_CSV_PATH}")
