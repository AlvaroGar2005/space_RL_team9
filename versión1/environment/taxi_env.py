"""
environment/taxi_env.py
-----------------------
Factory and helper functions for the Taxi-v3 Gymnasium environment.

Keeping environment creation centralized makes it trivial to swap the
environment for Version 2 (wrappers) or Version 3 (SpaceMissionEnv)
without touching training or evaluation logic.
"""

import gymnasium as gym
from config import ENV_NAME


def make_env(render_mode: str = None) -> gym.Env:
    """
    Create and return a Taxi-v3 environment.

    Args:
        render_mode: None for training/evaluation without visuals,
                     "human" for terminal rendering (demo mode),
                     "ansi" for text-based rendering (e.g. inside notebooks).

    Returns:
        gym.Env: configured environment instance.
    """
    return gym.make(ENV_NAME, render_mode=render_mode)


def env_info(env: gym.Env) -> dict:
    """
    Return a summary dict of the environment's key properties.
    Useful for logging and for the Jupyter notebook.
    """
    return {
        "env_name"  : ENV_NAME,
        "n_states"  : env.observation_space.n,
        "n_actions" : env.action_space.n,
        "max_steps" : env.spec.max_episode_steps if env.spec else "unknown",
    }