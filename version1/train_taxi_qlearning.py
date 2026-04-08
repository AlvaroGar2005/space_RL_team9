"""
train_taxi_qlearning.py
=======================
SpaceRL – Version 1: Standard Gymnasium Environment
----------------------------------------------------
Trains a tabular Q-learning agent on the Taxi-v3 environment.

The agent learns to navigate a 5×5 grid, pick up a passenger from one of
four designated locations and drop them off at the correct destination.

Learning algorithm: Q-learning (off-policy TD control)
Q-table update (Bellman equation):
    Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

Outputs:
    qtable.npy   – trained Q-table
    metrics.npz  – per-episode training metrics
"""

import numpy as np
import gymnasium as gym

# =============================================================================
# Hyperparameters
# =============================================================================
N_EPISODES     = 10_000   # Total number of training episodes
MAX_STEPS      = 200      # Maximum steps per episode (Taxi-v3 default truncation)
LEARNING_RATE  = 0.1      # α: how strongly new information overrides old
DISCOUNT_FACTOR = 0.99    # γ: importance of future rewards (close to 1 = far-sighted)
EPSILON_START  = 1.0      # ε initial: fully exploratory at the start
EPSILON_MIN    = 0.01     # ε floor: always keep a tiny amount of exploration
EPSILON_DECAY  = 0.9995   # multiplicative decay applied after every episode
LOG_EVERY      = 500      # Print a summary to stdout every N episodes

# =============================================================================
# Environment setup
# =============================================================================
# render_mode=None during training for maximum speed
env = gym.make("Taxi-v3", render_mode=None)

n_states  = env.observation_space.n   # 500 discrete states
n_actions = env.action_space.n        # 6 discrete actions

print(f"[INFO] Environment  : Taxi-v3")
print(f"[INFO] States       : {n_states}")
print(f"[INFO] Actions      : {n_actions}")
print(f"[INFO] Episodes     : {N_EPISODES}")
print()

# =============================================================================
# Q-table initialisation
# =============================================================================
# All zeros: the agent starts with no prior knowledge
q_table = np.zeros((n_states, n_actions))

# =============================================================================
# Metric storage
# =============================================================================
rewards_per_episode   = np.zeros(N_EPISODES)  # total reward each episode
steps_per_episode     = np.zeros(N_EPISODES)  # number of steps each episode
td_errors_per_episode = np.zeros(N_EPISODES)  # mean |TD error| each episode

# =============================================================================
# Training loop
# =============================================================================
epsilon = EPSILON_START

for episode in range(N_EPISODES):

    # --- Reset environment ---
    # Modern Gymnasium API: reset() returns (obs, info)
    obs, info = env.reset()

    total_reward = 0.0
    total_td_error = 0.0
    step = 0
    done = False

    while not done and step < MAX_STEPS:

        # --- Action selection: epsilon-greedy policy ---
        if np.random.uniform(0, 1) < epsilon:
            # Explore: pick a random action
            action = env.action_space.sample()
        else:
            # Exploit: pick the action with the highest Q-value
            action = int(np.argmax(q_table[obs]))

        # --- Step the environment ---
        # Modern Gymnasium API: step() returns 5 values
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Episode ends if the agent reaches a terminal state (dropped off
        # passenger correctly) or if MAX_STEPS is exceeded (truncated)
        done = terminated or truncated

        # --- Q-learning update (Bellman equation) ---
        best_next_q   = np.max(q_table[next_obs])          # max_a' Q(s', a')
        current_q     = q_table[obs, action]               # Q(s, a)
        td_target     = reward + DISCOUNT_FACTOR * best_next_q * (not done)
        td_error      = td_target - current_q
        q_table[obs, action] += LEARNING_RATE * td_error

        # --- Accumulate metrics ---
        total_reward   += reward
        total_td_error += abs(td_error)
        step           += 1
        obs             = next_obs

    # --- Store episode metrics ---
    rewards_per_episode[episode]   = total_reward
    steps_per_episode[episode]     = step
    td_errors_per_episode[episode] = total_td_error / step if step > 0 else 0.0

    # --- Epsilon decay ---
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # --- Periodic logging ---
    if (episode + 1) % LOG_EVERY == 0:
        recent_slice   = slice(max(0, episode - LOG_EVERY + 1), episode + 1)
        mean_reward    = np.mean(rewards_per_episode[recent_slice])
        mean_steps     = np.mean(steps_per_episode[recent_slice])
        mean_td        = np.mean(td_errors_per_episode[recent_slice])
        print(
            f"Episode {episode + 1:>6}/{N_EPISODES} | "
            f"ε={epsilon:.4f} | "
            f"Avg reward={mean_reward:>7.2f} | "
            f"Avg steps={mean_steps:>6.1f} | "
            f"Avg TD error={mean_td:.4f}"
        )

env.close()

# =============================================================================
# Save artefacts
# =============================================================================
np.save("qtable.npy", q_table)
print("\n[INFO] Q-table saved  → qtable.npy")

np.savez(
    "metrics.npz",
    rewards=rewards_per_episode,
    steps=steps_per_episode,
    td_errors=td_errors_per_episode,
)
print("[INFO] Metrics saved  → metrics.npz")
print("\nTraining complete. Run evaluate_taxi_qlearning.py to assess the agent.")