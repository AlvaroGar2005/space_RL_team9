"""
evaluation/eval_runner_v3.py
-----------------------------
Greedy evaluation for SpaceRL V3.
"""

import numpy as np
from config import N_EVAL_EPS, MAX_STEPS, DEMO_EPS, ACTION_NAMES
from agent.qlearning_agent import QLearningAgent
from environment.space_mission_env import SpaceMissionEnv


def run_greedy_episode(env, agent, max_steps=MAX_STEPS, render=False):
    obs, _ = env.reset()
    state  = int(obs)

    total_reward = 0.0
    success      = False

    for step in range(max_steps):
        action = agent.greedy_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        state        = int(next_obs)
        total_reward += reward
        done         = terminated or truncated

        if terminated and info.get("mission_success", False):
            success = True
        if done:
            break

    return total_reward, step + 1, success


def evaluate_agent_v3(agent: QLearningAgent,
                       n_episodes: int = N_EVAL_EPS) -> dict:
    env       = SpaceMissionEnv()
    rewards   = np.zeros(n_episodes)
    steps     = np.zeros(n_episodes, dtype=int)
    successes = np.zeros(n_episodes, dtype=bool)

    print(f"\n[EVAL] Running {n_episodes} greedy episodes on SpaceMissionEnv...")
    for ep in range(n_episodes):
        r, s, ok      = run_greedy_episode(env, agent)
        rewards[ep]   = r
        steps[ep]     = s
        successes[ep] = ok

    env.close()
    return {"rewards": rewards, "steps": steps, "successes": successes}


def run_demo(agent: QLearningAgent, n_demo: int = DEMO_EPS):
    """Run visual demo episodes with ansi render."""
    print(f"\n[DEMO] Running {n_demo} demo episode(s)...")
    # Change render_mode to "ansi" for terminal display or "human" for pygame visual window
    env = SpaceMissionEnv(render_mode="human")

    for ep in range(n_demo):
        r, s, ok = run_greedy_episode(env, agent, render=True)
        status   = "SUCCESS ✓" if ok else "FAILED  ✗"
        print(f"\n  Demo ep {ep+1}/{n_demo} | reward: {r:.1f} | steps: {s} | {status}")

    env.close()
    print("[DEMO] Done.")