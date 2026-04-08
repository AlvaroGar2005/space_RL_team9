"""
evaluate_taxi_qlearning.py
==========================
SpaceRL – Version 1: Standard Gymnasium Environment
----------------------------------------------------
Evaluates the trained Q-learning agent on Taxi-v3 using a pure greedy policy
(no exploration). Loads the Q-table produced by train_taxi_qlearning.py.

Metrics reported:
    - Mean total reward
    - Standard deviation of total reward
    - Mean number of steps per episode
    - Success rate (episodes where the passenger was delivered correctly)

Optional visual demo:
    Set RENDER_DEMO = True to watch a few episodes rendered in the terminal.
"""

import os
import sys
import numpy as np
import gymnasium as gym

# =============================================================================
# Evaluation configuration
# =============================================================================
N_EVAL_EPISODES = 1_000   # Number of episodes used to compute statistics
MAX_STEPS       = 200     # Matches the truncation limit used during training
RENDER_DEMO     = False   # Set True to watch a few rendered episodes
N_DEMO_EPISODES = 3       # How many episodes to render when RENDER_DEMO=True

# =============================================================================
# Load trained Q-table
# =============================================================================
QTABLE_PATH = "qtable.npy"

if not os.path.exists(QTABLE_PATH):
    print(
        "[ERROR] qtable.npy not found.\n"
        "        Please run train_taxi_qlearning.py first to train the agent."
    )
    sys.exit(1)

q_table = np.load(QTABLE_PATH)
print(f"[INFO] Q-table loaded from '{QTABLE_PATH}'  shape={q_table.shape}")
print(f"[INFO] Evaluating over {N_EVAL_EPISODES} episodes (greedy policy)...\n")

# =============================================================================
# Evaluation loop  –  pure greedy policy (ε = 0)
# =============================================================================
env = gym.make("Taxi-v3", render_mode=None)

all_rewards = np.zeros(N_EVAL_EPISODES)
all_steps   = np.zeros(N_EVAL_EPISODES)
successes   = 0

for episode in range(N_EVAL_EPISODES):

    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    done = False
    episode_success = False

    while not done and step < MAX_STEPS:

        # Greedy action: always pick the best-known action (no randomness)
        action = int(np.argmax(q_table[obs]))

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step         += 1
        obs           = next_obs

        # Taxi-v3 gives +20 reward for a correct drop-off (terminal state)
        if terminated and reward == 20:
            episode_success = True

    all_rewards[episode] = total_reward
    all_steps[episode]   = step
    if episode_success:
        successes += 1

env.close()

# =============================================================================
# Print summary statistics
# =============================================================================
mean_reward  = np.mean(all_rewards)
std_reward   = np.std(all_rewards)
mean_steps   = np.mean(all_steps)
success_rate = successes / N_EVAL_EPISODES * 100

print("=" * 50)
print("          EVALUATION RESULTS")
print("=" * 50)
print(f"  Episodes evaluated : {N_EVAL_EPISODES}")
print(f"  Mean reward        : {mean_reward:.2f}")
print(f"  Std  reward        : {std_reward:.2f}")
print(f"  Mean steps         : {mean_steps:.1f}")
print(f"  Success rate       : {success_rate:.1f}%")
print("=" * 50)

# Interpretation hint
if success_rate >= 95:
    print("[✓] Excellent policy – the agent solves Taxi-v3 reliably.")
elif success_rate >= 75:
    print("[~] Decent policy – consider more training episodes.")
else:
    print("[✗] Weak policy – the agent needs more training.")

# =============================================================================
# Optional visual demonstration
# =============================================================================
if RENDER_DEMO:
    print(f"\n[INFO] Running {N_DEMO_EPISODES} rendered demo episodes...\n")

    # Create a separate environment with human rendering
    demo_env = gym.make("Taxi-v3", render_mode="human")

    for demo_ep in range(N_DEMO_EPISODES):
        obs, info = demo_env.reset()
        total_reward = 0.0
        step = 0
        done = False

        print(f"--- Demo episode {demo_ep + 1} ---")

        while not done and step < MAX_STEPS:
            action = int(np.argmax(q_table[obs]))
            obs, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

        print(f"    Total reward: {total_reward:.0f}  |  Steps: {step}\n")

    demo_env.close()