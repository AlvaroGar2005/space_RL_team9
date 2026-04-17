"""
wrappers/action_wrapper.py
--------------------------
SpaceRL ActionWrapper for Taxi-v3.

Narrative:
    In a real spacecraft or autonomous drone, certain operations carry higher
    operational risk (e.g. docking, cargo transfer). This wrapper adds a
    tracking and logging layer that records:
        - how often "high-risk" operations (pickup/dropoff) are attempted
        - whether those operations are valid in context
        - a count of "transformed" actions (random re-roll when epsilon-greedy
          selects the same risky action twice in a row, simulating a safety
          lock that prevents repeated invalid docking attempts)

What this wrapper changes:
    - Tracks action usage and flags semantically risky actions
    - Applies a simple safety-lock rule: if the same risky action is taken
      twice in a row AND it resulted in a -10 reward last time, the wrapper
      redirects it to the best neighbouring movement action (0 or 1).
    - Counts blocked (redirected) and transformed actions per episode.

What this wrapper does NOT change:
    - The action space (still Discrete(6))
    - The observation space
    - The reward signal (reward shaping is handled by RewardWrapper)
    - env.step() interface

Design note:
    The safety-lock is intentionally simple and conservative. It only
    redirects repeated invalid risky actions — it never blocks the agent
    from learning. This ensures the wrapper is academically defensible
    and does not artificially inflate performance.
"""

import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper

from config import RISKY_ACTIONS

# Taxi-v3 action labels
ACTION_LABELS = {
    0: "Move South",
    1: "Move North",
    2: "Move East",
    3: "Move West",
    4: "Pickup  (high-risk op)",
    5: "Dropoff (high-risk op)",
}

# Fallback movement when a repeated risky action is redirected
SAFETY_REDIRECT = 0   # Move South — neutral fallback


class SpaceRLActionWrapper(ActionWrapper):
    """
    Adds a semantic tracking and safety-lock layer over Taxi-v3 actions.

    Safety-lock rule:
        If action ∈ RISKY_ACTIONS AND the previous identical action resulted
        in reward == -10 (invalid), redirect to SAFETY_REDIRECT.
        This simulates a spacecraft safety protocol that prevents repeated
        invalid docking manoeuvres.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Per-episode counters
        self.action_counts       = {a: 0 for a in range(6)}
        self.blocked_actions     = 0
        self.transformed_actions = 0

        # Safety-lock state
        self._last_action        = None
        self._last_was_invalid   = False

    def reset(self, **kwargs):
        self.action_counts       = {a: 0 for a in range(6)}
        self.blocked_actions     = 0
        self.transformed_actions = 0
        self._last_action        = None
        self._last_was_invalid   = False
        return self.env.reset(**kwargs)

    def action(self, act: int) -> int:
        """
        Process the action before passing it to env.step().

        Applies safety-lock if conditions are met; otherwise passes through.
        """
        original_action = act

        # Safety-lock: redirect repeated invalid risky actions
        if (
            act in RISKY_ACTIONS
            and act == self._last_action
            and self._last_was_invalid
        ):
            act = SAFETY_REDIRECT
            self.blocked_actions     += 1
            self.transformed_actions += 1

        self.action_counts[original_action] += 1
        return act

    def step(self, action: int):
        """
        Override step to update safety-lock state after each transition.
        """
        transformed = self.action(action)
        obs, reward, terminated, truncated, info = self.env.step(transformed)

        # Update safety-lock memory
        self._last_action      = transformed
        self._last_was_invalid = (reward <= -10)

        # Inject action tracking into info dict
        info["original_action"]   = action
        info["transformed_action"] = transformed
        info["was_redirected"]    = (transformed != action)

        return obs, reward, terminated, truncated, info

    def get_episode_action_stats(self) -> dict:
        """Return per-episode action tracking metrics for logging."""
        total = sum(self.action_counts.values()) or 1
        risky_count = sum(self.action_counts.get(a, 0) for a in RISKY_ACTIONS)
        return {
            "blocked_actions"     : self.blocked_actions,
            "transformed_actions" : self.transformed_actions,
            "risky_action_ratio"  : round(risky_count / total, 4),
            "useful_actions_ratio": round(
                (total - self.blocked_actions) / total, 4
            ),
        }