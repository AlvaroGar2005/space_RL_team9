"""
evaluate_taxi_qlearning.py
--------------------------
Evaluates a trained Q-learning agent on Taxi-v3.
Loads qtable.npy and runs purely greedy episodes (epsilon = 0).

Configuration:
    RENDER_DEMO  (bool) : set to True to watch a few episodes in the terminal.
    N_EVAL_EPS   (int)  : number of evaluation episodes to run.
    DEMO_EPS     (int)  : number of episodes rendered when RENDER_DEMO = True.
"""

import sys
import numpy as np
import gymnasium as gym
import time

# ──────────────────────────────────────────────
# Evaluation configuration
# ──────────────────────────────────────────────
N_EVAL_EPS  = 100    # episodes used to compute statistics
DEMO_EPS    = 3      # episodes rendered when RENDER_DEMO = True
RENDER_DEMO = True  # set to True to activate visual demonstration


def load_qtable(path: str = "qtable.npy") -> np.ndarray:
    """
    Load the Q-table from disk.
    Raises a clear error if the file does not exist.
    """
    try:
        qtable = np.load(path)
        print(f"[INFO] Q-table loaded from '{path}'  shape: {qtable.shape}")
        return qtable
    except FileNotFoundError:
        print(
            f"\n[ERROR] File '{path}' not found.\n"
            "  → You must train the agent first.\n"
            "  → Run:  python train_taxi_qlearning.py\n"
        )
        sys.exit(1)


def greedy_action(qtable: np.ndarray, state: int) -> int:
    """Select the action with the highest Q-value for the given state."""
    return int(np.argmax(qtable[state]))


def run_episode(
    env: gym.Env,
    qtable: np.ndarray,
    max_steps: int = 200,
    render: bool = False,
    delay: float = 0.3,
) -> tuple:
    """
    Run a single greedy evaluation episode.

    Returns:
        total_reward (float): cumulative reward obtained.
        steps        (int)  : number of steps taken.
        success      (bool) : True if the taxi delivered the passenger
                              (reward == 20 is the terminal success reward in Taxi-v3).
    """
    obs, _info = env.reset()
    state = int(obs)

    total_reward = 0.0
    success = False

    if render:
        env.render()
        time.sleep(delay)

    for step in range(max_steps):
        action = greedy_action(qtable, state)
        next_obs, reward, terminated, truncated, _info = env.step(action)

        total_reward += reward
        done = terminated or truncated
        state = int(next_obs)

        if render:
            env.render()
            time.sleep(delay)

        # Taxi-v3: successful delivery gives reward = 20
        if terminated and reward == 20:
            success = True

        if done:
            break

    return total_reward, step + 1, success

def evaluate(qtable: np.ndarray):
    """
    Run N_EVAL_EPS greedy episodes and print summary statistics.
    """
    env = gym.make("Taxi-v3")

    rewards  = np.zeros(N_EVAL_EPS)
    steps    = np.zeros(N_EVAL_EPS, dtype=int)
    successes = np.zeros(N_EVAL_EPS, dtype=bool)

    print(f"\n[INFO] Running {N_EVAL_EPS} evaluation episodes (greedy policy)...")

    for ep in range(N_EVAL_EPS):
        r, s, ok          = run_episode(env, qtable)
        rewards[ep]       = r
        steps[ep]         = s
        successes[ep]     = ok

    env.close()

    # ── Print results ──────────────────────────────
    print("\n" + "=" * 45)
    print("  Evaluation Results")
    print("=" * 45)
    print(f"  Episodes evaluated : {N_EVAL_EPS}")
    print(f"  Mean reward        : {np.mean(rewards):.2f}")
    print(f"  Std reward         : {np.std(rewards):.2f}")
    print(f"  Min reward         : {np.min(rewards):.2f}")
    print(f"  Max reward         : {np.max(rewards):.2f}")
    print(f"  Mean steps         : {np.mean(steps):.1f}")
    print(f"  Success rate       : {np.mean(successes) * 100:.1f}%")
    print("=" * 45)

    return rewards, steps, successes


def render_demo(qtable: np.ndarray):
    """
    Run DEMO_EPS episodes with render_mode='human' for visual inspection.
    Only called when RENDER_DEMO = True.
    """
    print(f"\n[DEMO] Rendering {DEMO_EPS} episode(s) with human mode...")
    env = gym.make("Taxi-v3", render_mode="human")

    for ep in range(DEMO_EPS):
        r, s, ok = run_episode(env, qtable, render=True, delay=0.35)
        status = "SUCCESS ✓" if ok else "FAILED  ✗"
        print(f"  Demo ep {ep + 1}/{DEMO_EPS} | reward: {r:.0f} | steps: {s} | {status}")

    env.close()
    print("[DEMO] Done.")

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SpaceRL – Version 1: Taxi-v3 Q-learning Evaluation")
    print("=" * 55)

    qtable = load_qtable("qtable.npy")

    evaluate(qtable)

    if RENDER_DEMO:
        render_demo(qtable)
    else:
        print("\n[TIP] Set RENDER_DEMO = True in this script to watch the agent play.")