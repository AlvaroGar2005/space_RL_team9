"""
evaluation/eval_runner.py
-------------------------
Greedy evaluation logic for the trained Q-learning agent.

Functions here are used by both:
    evaluate_taxi_qlearning.py  (CLI entry point)
    notebooks/SpaceRL_v1.ipynb  (Jupyter notebook)
"""

import time
import numpy as np
import gymnasium as gym

from config import N_EVAL_EPS, DEMO_EPS, MAX_STEPS
from agent.qlearning_agent import QLearningAgent
from environment.taxi_env import make_env


#Single episode

def run_greedy_episode(
    env: gym.Env,
    agent: QLearningAgent,
    max_steps: int = MAX_STEPS,
    render: bool = False,
    delay: float = 0.3,
) -> tuple:
    """
    Run one fully greedy evaluation episode (epsilon = 0).

    Args:
        env       : Gymnasium environment
        agent     : trained QLearningAgent
        max_steps : step cap per episode
        render    : whether to call env.render() at each step
        delay     : seconds between render frames

    Returns:
        total_reward (float): cumulative reward
        steps        (int)  : steps taken
        success      (bool) : True when taxi delivers passenger (reward == 20)
    """
    obs, _info = env.reset()
    state      = int(obs)

    total_reward = 0.0
    success      = False

    if render:
        env.render()
        time.sleep(delay)

    for step in range(max_steps):
        action = agent.greedy_action(state)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        state        = int(next_obs)
        total_reward += reward
        done         = terminated or truncated

        if render:
            env.render()
            time.sleep(delay)

        # Successful delivery in Taxi-v3 yields reward = +20
        if terminated and reward == 20:
            success = True

        if done:
            break

    return total_reward, step + 1, success


#Batch evaluation

def evaluate_agent(
    agent: QLearningAgent,
    n_episodes: int = N_EVAL_EPS,
) -> tuple:
    """
    Run n_episodes greedy evaluation episodes without rendering.

    Returns:
        rewards   (np.ndarray): total reward per episode
        steps     (np.ndarray): steps per episode
        successes (np.ndarray): bool array, True = successful delivery
    """
    env       = make_env(render_mode=None)
    rewards   = np.zeros(n_episodes)
    steps     = np.zeros(n_episodes, dtype=int)
    successes = np.zeros(n_episodes, dtype=bool)

    print(f"\n[EVAL] Running {n_episodes} greedy episodes...")

    for ep in range(n_episodes):
        r, s, ok      = run_greedy_episode(env, agent)
        rewards[ep]   = r
        steps[ep]     = s
        successes[ep] = ok

    env.close()
    return rewards, steps, successes


# Visual demo

def run_demo(agent: QLearningAgent, n_demo: int = DEMO_EPS):
    """
    Render n_demo episodes with render_mode='human' in the terminal.
    Requires a terminal that supports ANSI escape codes.
    """
    print(f"\n[DEMO] Rendering {n_demo} episode(s) in human mode...")
    env = make_env(render_mode="human")

    for ep in range(n_demo):
        r, s, ok = run_greedy_episode(env, agent, render=True, delay=0.35)
        status   = "SUCCESS ✓" if ok else "FAILED  ✗"
        print(f"  Demo ep {ep + 1}/{n_demo} | reward: {r:.0f} | steps: {s} | {status}")

    env.close()
    print("[DEMO] Done.")