"""
training/train_loop_v3.py
--------------------------
Training loop for SpaceRL V3 — SpaceMissionEnv.

Records all required metrics:
    - reward, steps, td_error, epsilon (standard, same as V1/V2)
    - success, termination_reason      (mission-specific)
    - survival_rate                    (fraction of non-failure episodes)
"""

import numpy as np
import gymnasium as gym

from config import N_EPISODES, MAX_STEPS, LOG_EVERY
from agent.qlearning_agent import QLearningAgent


def run_training_v3(
    env: gym.Env,
    agent: QLearningAgent,
    n_episodes: int = N_EPISODES,
    max_steps: int  = MAX_STEPS,
    log_every: int  = LOG_EVERY,
) -> dict:
    """
    Run Q-learning training on SpaceMissionEnv.

    Returns a dict of numpy arrays (one entry per episode).
    """
    rewards          = np.zeros(n_episodes)
    steps_arr        = np.zeros(n_episodes, dtype=int)
    td_errors        = np.zeros(n_episodes)
    epsilon_history  = np.zeros(n_episodes)
    success_arr      = np.zeros(n_episodes, dtype=bool)
    term_reasons     = []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        state     = int(obs)

        total_reward = 0.0
        total_td     = 0.0
        n_steps      = 0
        success      = False
        term_reason  = "truncated"

        for _ in range(max_steps):
            action = agent.choose_action(state, env)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_obs)
            done       = terminated or truncated

            td_err       = agent.update(state, action, reward, next_state, done)
            total_reward += reward
            total_td     += td_err
            n_steps      += 1
            state         = next_state

            if terminated:
                term_reason = info.get("mission_success") and "success" or "failure"
                success     = info.get("mission_success", False)

            if done:
                break

        agent.decay_epsilon()

        rewards[ep]         = total_reward
        steps_arr[ep]       = n_steps
        td_errors[ep]       = total_td / n_steps if n_steps > 0 else 0.0
        epsilon_history[ep] = agent.epsilon
        success_arr[ep]     = success
        term_reasons.append(term_reason)

        if (ep + 1) % log_every == 0:
            w  = max(0, ep - log_every + 1)
            ar = np.mean(rewards[w:ep+1])
            sr = np.mean(success_arr[w:ep+1]) * 100
            print(
                f"  Ep {ep+1:>7,}/{n_episodes:,} | "
                f"avg_reward: {ar:>8.2f} | "
                f"success: {sr:>5.1f}% | "
                f"epsilon: {agent.epsilon:.4f}"
            )

    return {
        "rewards"         : rewards,
        "steps"           : steps_arr,
        "td_errors"       : td_errors,
        "epsilon"         : epsilon_history,
        "success"         : success_arr,
        "term_reasons"    : np.array(term_reasons),
    }