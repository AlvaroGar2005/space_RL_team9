"""
training/train_loop.py
----------------------
Core training loop for Q-learning on Taxi-v3.

Separated from the entry-point script so it can be imported and called
from both train_taxi_qlearning.py and the Jupyter notebook without code
duplication.
"""

import numpy as np
import gymnasium as gym

from config import N_EPISODES, MAX_STEPS, LOG_EVERY
from agent.qlearning_agent import QLearningAgent


def run_training(
    env: gym.Env,
    agent: QLearningAgent,
    n_episodes: int = N_EPISODES,
    max_steps: int = MAX_STEPS,
    log_every: int = LOG_EVERY,
) -> tuple:
    """
    Run the full Q-learning training loop.

    Args:
        env        : Gymnasium environment (already created)
        agent      : QLearningAgent instance (Q-table initialized to zeros)
        n_episodes : number of episodes to train for
        max_steps  : maximum steps per episode
        log_every  : console log frequency

    Returns:
        rewards         (np.ndarray, shape n_episodes): total reward per episode
        steps           (np.ndarray, shape n_episodes): steps per episode
        td_errors       (np.ndarray, shape n_episodes): mean |TD error| per episode
        epsilon_history (np.ndarray, shape n_episodes): epsilon value at episode end
    """
    rewards         = np.zeros(n_episodes)
    steps_arr       = np.zeros(n_episodes, dtype=int)
    td_errors       = np.zeros(n_episodes)
    epsilon_history = np.zeros(n_episodes)

    for ep in range(n_episodes):
        obs, _info = env.reset()          # modern Gymnasium API
        state      = int(obs)

        total_reward = 0.0
        total_td     = 0.0
        n_steps      = 0

        for _ in range(max_steps):
            # 1. Select action (epsilon-greedy)
            action = agent.choose_action(state, env)

            # 2. Step environment
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = int(next_obs)
            done       = terminated or truncated

            # 3. Bellman update
            td_err       = agent.update(state, action, reward, next_state, done)
            total_reward += reward
            total_td     += td_err
            n_steps      += 1
            state         = next_state

            if done:
                break

        # 4. Decay exploration rate
        agent.decay_epsilon()

        # 5. Record metrics
        rewards[ep]         = total_reward
        steps_arr[ep]       = n_steps
        td_errors[ep]       = total_td / n_steps if n_steps > 0 else 0.0
        epsilon_history[ep] = agent.epsilon

        # 6. Periodic console log
        if (ep + 1) % log_every == 0:
            window_start = max(0, ep - log_every + 1)
            avg_rew      = np.mean(rewards[window_start : ep + 1])
            print(
                f"  Ep {ep + 1:>7,}/{n_episodes:,} | "
                f"avg_reward (last {log_every}): {avg_rew:>8.2f} | "
                f"epsilon: {agent.epsilon:.4f}"
            )

    return rewards, steps_arr, td_errors, epsilon_history