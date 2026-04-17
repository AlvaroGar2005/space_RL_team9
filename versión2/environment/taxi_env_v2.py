"""
environment/taxi_env_v2.py
--------------------------
Factory functions for the V2 wrapped Taxi-v3 environment.
"""

import gymnasium as gym

from config import ENV_NAME
from wrappers import (
    SpaceRLObservationWrapper,
    SpaceRLRewardWrapper,
    SpaceRLActionWrapper,
)


def make_env_v2(render_mode: str = None) -> gym.Env:
    """
    Create the fully wrapped SpaceRL V2 environment.

    Stacking order (inner to outer):
        Taxi-v3 -> ObservationWrapper -> RewardWrapper -> ActionWrapper
    """
    env = gym.make(ENV_NAME, render_mode=render_mode)
    env = SpaceRLObservationWrapper(env)
    env = SpaceRLRewardWrapper(env)
    env = SpaceRLActionWrapper(env)
    return env


def make_env_base(render_mode: str = None) -> gym.Env:
    """Plain unwrapped Taxi-v3, used for baseline comparisons."""
    return gym.make(ENV_NAME, render_mode=render_mode)


def env_info_v2(env: gym.Env) -> dict:
    """Return a summary of the wrapped environment's key properties."""
    return {
        "env_name"  : ENV_NAME,
        "wrappers"  : ["SpaceRLObservationWrapper", "SpaceRLRewardWrapper", "SpaceRLActionWrapper"],
        "n_states"  : env.observation_space.n,
        "n_actions" : env.action_space.n,
    }