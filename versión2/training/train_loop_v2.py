"""
training/train_loop_v2.py
-------------------------
Extended training loop for SpaceRL V2.

Identical Q-learning logic to V1, but records additional per-episode metrics
from the three SpaceRL wrappers:
    - invalid_actions     (from RewardWrapper)
    - wrapper_penalty     (from RewardWrapper)
    - wrapper_bonus       (from RewardWrapper)
    - blocked_actions     (from ActionWrapper)
    - transformed_actions (from ActionWrapper)
    - risky_action_ratio  (from ActionWrapper)
    - useful_actions_ratio(from ActionWrapper)

These metrics are the main academic contribution of V2: they make the impact
of the wrappers quantifiable and comparable against V1.
"""

import numpy as np
import gymnasium as gym

from config import N_EPISODES, MAX_STEPS, LOG_EVERY
from agent.qlearning_agent import QLearningAgent


def run_training_v2(
    env: gym.Env,
    agent: QLearningAgent,
    n_episodes: int = N_EPISODES,
    max_steps: int  = MAX_STEPS,
    log_every: int  = LOG_EVERY,
) -> dict:
    """
    Run the Q-learning training loop on the wrapped V2 environment.

    Returns a dict of numpy arrays (one value per episode) containing
    both the standard V1 metrics and the V2 wrapper-specific metrics.
    """
    # Standard metrics (same as V1)
    rewards         = np.zeros(n_episodes)
    steps_arr       = np.zeros(n_episodes, dtype=int)
    td_errors       = np.zeros(n_episodes)
    epsilon_history = np.zeros(n_episodes)
    success_arr     = np.zeros(n_episodes, dtype=bool)

    # V2 wrapper-specific metrics
    invalid_actions_arr      = np.zeros(n_episodes, dtype=int)
    wrapper_penalty_arr      = np.zeros(n_episodes)
    wrapper_bonus_arr        = np.zeros(n_episodes)
    blocked_actions_arr      = np.zeros(n_episodes, dtype=int)
    transformed_actions_arr  = np.zeros(n_episodes, dtype=int)
    risky_ratio_arr          = np.zeros(n_episodes)
    useful_ratio_arr         = np.zeros(n_episodes)

    for ep in range(n_episodes):
        obs, _info = env.reset()
        state      = int(obs)

        total_reward = 0.0
        total_td     = 0.0
        n_steps      = 0
        success      = False

        for _ in range(max_steps):
            action = agent.choose_action(state, env)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = int(next_obs)
            done       = terminated or truncated

            td_err       = agent.update(state, action, reward, next_state, done)
            total_reward += reward
            total_td     += td_err
            n_steps      += 1
            state         = next_state

            # Successful delivery: native reward before wrappers would be +20,
            # after RewardWrapper it is ≥ 30 (delivery bonus added)
            if terminated and reward >= 20:
                success = True

            if done:
                break

        agent.decay_epsilon()

        # ── Collect wrapper stats ──────────────────
        # Unwrap to reach each individual wrapper layer
        reward_w = _get_wrapper(env, "SpaceRLRewardWrapper")
        action_w = _get_wrapper(env, "SpaceRLActionWrapper")

        rew_stats = reward_w.get_episode_wrapper_stats() if reward_w else {}
        act_stats = action_w.get_episode_action_stats()  if action_w else {}

        # ── Store metrics ──────────────────────────
        rewards[ep]                = total_reward
        steps_arr[ep]              = n_steps
        td_errors[ep]              = total_td / n_steps if n_steps > 0 else 0.0
        epsilon_history[ep]        = agent.epsilon
        success_arr[ep]            = success

        invalid_actions_arr[ep]    = rew_stats.get("invalid_actions", 0)
        wrapper_penalty_arr[ep]    = rew_stats.get("wrapper_penalty", 0.0)
        wrapper_bonus_arr[ep]      = rew_stats.get("wrapper_bonus", 0.0)
        blocked_actions_arr[ep]    = act_stats.get("blocked_actions", 0)
        transformed_actions_arr[ep]= act_stats.get("transformed_actions", 0)
        risky_ratio_arr[ep]        = act_stats.get("risky_action_ratio", 0.0)
        useful_ratio_arr[ep]       = act_stats.get("useful_actions_ratio", 1.0)

        # ── Console log ────────────────────────────
        if (ep + 1) % log_every == 0:
            w_start  = max(0, ep - log_every + 1)
            avg_rew  = np.mean(rewards[w_start : ep + 1])
            avg_inv  = np.mean(invalid_actions_arr[w_start : ep + 1])
            print(
                f"  Ep {ep + 1:>7,}/{n_episodes:,} | "
                f"avg_reward: {avg_rew:>8.2f} | "
                f"epsilon: {agent.epsilon:.4f} | "
                f"avg_invalid: {avg_inv:.1f}"
            )

    return {
        "rewards"             : rewards,
        "steps"               : steps_arr,
        "td_errors"           : td_errors,
        "epsilon"             : epsilon_history,
        "success"             : success_arr,
        "invalid_actions"     : invalid_actions_arr,
        "wrapper_penalty"     : wrapper_penalty_arr,
        "wrapper_bonus"       : wrapper_bonus_arr,
        "blocked_actions"     : blocked_actions_arr,
        "transformed_actions" : transformed_actions_arr,
        "risky_action_ratio"  : risky_ratio_arr,
        "useful_actions_ratio": useful_ratio_arr,
    }


def _get_wrapper(env: gym.Env, class_name: str):
    """
    Walk the wrapper chain to find an instance by class name.
    Returns None if not found.
    """
    current = env
    while current is not None:
        if type(current).__name__ == class_name:
            return current
        current = getattr(current, "env", None)
    return None
