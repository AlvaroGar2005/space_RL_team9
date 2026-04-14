"""
train_taxi_qlearning.py
-----------------------
Trains a Q-learning agent on the Taxi-v3 environment from Gymnasium.
Uses tabular Q-learning with an epsilon-greedy exploration policy.

Outputs:
    - qtable.npy     : learned Q-table
    - metrics.npz    : per-episode training metrics (rewards, steps, TD errors)
"""

import numpy as np
import gymnasium as gym

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
N_EPISODES     = 10_000   # total number of training episodes
MAX_STEPS      = 200      # maximum steps allowed per episode
LEARNING_RATE  = 0.1      # alpha: controls how fast Q-values are updated
DISCOUNT_FACTOR = 0.99    # gamma: importance of future rewards
EPSILON_START  = 1.0      # initial exploration rate
EPSILON_MIN    = 0.01     # minimum exploration rate (never fully greedy)
EPSILON_DECAY  = 0.9995   # multiplicative decay applied after each episode
LOG_EVERY      = 500      # print progress to console every N episodes


def make_env():
    """Create and return the Taxi-v3 environment (no rendering during training)."""
    return gym.make("Taxi-v3")


def initialize_qtable(env: gym.Env) -> np.ndarray:
    """
    Initialize the Q-table with zeros.

    Shape: (n_states, n_actions)
        - Taxi-v3 has 500 discrete states and 6 discrete actions.
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    print(f"[INFO] Q-table shape: ({n_states}, {n_actions})")
    return np.zeros((n_states, n_actions))


def choose_action(qtable: np.ndarray, state: int, epsilon: float, env: gym.Env) -> int:
    """
    Epsilon-greedy policy:
        - With probability epsilon  → explore (random action)
        - With probability 1-epsilon → exploit (greedy action from Q-table)
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()          # explore
    return int(np.argmax(qtable[state]))          # exploit


def bellman_update(
    qtable: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    done: bool,
) -> float:
    """
    Apply the Q-learning (off-policy TD) update rule.

    Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]

    Returns:
        td_error (float): absolute TD error for this transition.
    """
    best_next = 0.0 if done else float(np.max(qtable[next_state]))
    td_target = reward + DISCOUNT_FACTOR * best_next
    td_error  = td_target - qtable[state, action]
    qtable[state, action] += LEARNING_RATE * td_error
    return abs(td_error)


def train(env: gym.Env, qtable: np.ndarray):
    """
    Main training loop.

    Returns:
        episode_rewards (np.ndarray): total reward per episode
        episode_steps   (np.ndarray): number of steps per episode
        episode_td_errors (np.ndarray): mean TD error per episode
    """
    epsilon = EPSILON_START

    episode_rewards   = np.zeros(N_EPISODES)
    episode_steps     = np.zeros(N_EPISODES, dtype=int)
    episode_td_errors = np.zeros(N_EPISODES)

    for ep in range(N_EPISODES):
        obs, _info = env.reset()          # modern Gymnasium API
        state      = int(obs)

        total_reward = 0.0
        total_td     = 0.0
        steps        = 0

        for _ in range(MAX_STEPS):
            action = choose_action(qtable, state, epsilon, env)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = int(next_obs)
            done       = terminated or truncated   # episode ended for any reason

            td_err = bellman_update(qtable, state, action, reward, next_state, done)

            total_reward += reward
            total_td     += td_err
            steps        += 1
            state         = next_state

            if done:
                break

        # Decay epsilon after each episode
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        episode_rewards[ep]   = total_reward
        episode_steps[ep]     = steps
        episode_td_errors[ep] = total_td / steps if steps > 0 else 0.0

        # Periodic progress log
        if (ep + 1) % LOG_EVERY == 0:
            avg_rew = np.mean(episode_rewards[max(0, ep - LOG_EVERY + 1): ep + 1])
            print(
                f"  Episode {ep + 1:>6}/{N_EPISODES} | "
                f"avg reward (last {LOG_EVERY}): {avg_rew:>7.2f} | "
                f"epsilon: {epsilon:.4f}"
            )

    return episode_rewards, episode_steps, episode_td_errors


def save_results(qtable: np.ndarray, rewards, steps, td_errors):
    """Save Q-table and training metrics to disk."""
    np.save("qtable.npy", qtable)
    np.savez("metrics.npz", rewards=rewards, steps=steps, td_errors=td_errors)
    print("[INFO] Saved qtable.npy and metrics.npz")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  SpaceRL – Version 1: Taxi-v3 Q-learning Training")
    print("=" * 55)

    env = make_env()
    qtable = initialize_qtable(env)

    print(f"\n[INFO] Starting training for {N_EPISODES} episodes...")
    rewards, steps, td_errors = train(env, qtable)

    env.close()

    # Summary statistics
    last_n = 500
    print(f"\n[RESULTS] Last {last_n} episodes summary:")
    print(f"  Mean reward : {np.mean(rewards[-last_n:]):.2f}")
    print(f"  Mean steps  : {np.mean(steps[-last_n:]):.1f}")
    print(f"  Mean TD err : {np.mean(td_errors[-last_n:]):.4f}")

    save_results(qtable, rewards, steps, td_errors)
    print("\n[DONE] Training complete.")