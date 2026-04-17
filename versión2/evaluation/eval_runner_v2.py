"""
evaluation/eval_runner_v2.py
----------------------------
Greedy evaluation logic for the trained V2 agent on the wrapped environment.
"""

import numpy as np
import gymnasium as gym

from config import N_EVAL_EPS, MAX_STEPS
from agent.qlearning_agent import QLearningAgent
from environment import make_env_v2
from training.train_loop_v2 import _get_wrapper


def run_greedy_episode_v2(env, agent, max_steps=MAX_STEPS):
    obs, _ = env.reset()
    state  = int(obs)

    total_reward = 0.0
    success      = False

    for step in range(max_steps):
        action = agent.greedy_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        state        = int(next_obs)
        total_reward += reward
        done         = terminated or truncated

        if terminated and reward >= 20:
            success = True
        if done:
            break

    return total_reward, step + 1, success


def evaluate_agent_v2(agent: QLearningAgent,
                       n_episodes: int = N_EVAL_EPS) -> dict:
    """
    Run n_episodes greedy evaluation episodes on the V2 wrapped environment.
    Returns a dict of numpy arrays.
    """
    env = make_env_v2(render_mode=None)

    rewards   = np.zeros(n_episodes)
    steps     = np.zeros(n_episodes, dtype=int)
    successes = np.zeros(n_episodes, dtype=bool)
    inv_arr   = np.zeros(n_episodes, dtype=int)
    pen_arr   = np.zeros(n_episodes)
    bon_arr   = np.zeros(n_episodes)
    blk_arr   = np.zeros(n_episodes, dtype=int)
    use_arr   = np.zeros(n_episodes)

    print(f"\n[EVAL] Running {n_episodes} greedy episodes on wrapped env...")

    for ep in range(n_episodes):
        r, s, ok = run_greedy_episode_v2(env, agent)
        rewards[ep]   = r
        steps[ep]     = s
        successes[ep] = ok

        rw = _get_wrapper(env, "SpaceRLRewardWrapper")
        aw = _get_wrapper(env, "SpaceRLActionWrapper")
        rs = rw.get_episode_wrapper_stats() if rw else {}
        as_ = aw.get_episode_action_stats()  if aw else {}

        inv_arr[ep] = rs.get("invalid_actions", 0)
        pen_arr[ep] = rs.get("wrapper_penalty", 0.0)
        bon_arr[ep] = rs.get("wrapper_bonus",   0.0)
        blk_arr[ep] = as_.get("blocked_actions", 0)
        use_arr[ep] = as_.get("useful_actions_ratio", 1.0)

    env.close()
    return {
        "rewards": rewards, "steps": steps, "successes": successes,
        "invalid_actions": inv_arr, "wrapper_penalty": pen_arr,
        "wrapper_bonus": bon_arr, "blocked_actions": blk_arr,
        "useful_actions_ratio": use_arr,
    }
