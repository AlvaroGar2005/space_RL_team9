"""
wrappers/reward_wrapper.py
--------------------------
SpaceRL RewardWrapper for Taxi-v3.

Narrative:
    The taxi becomes an autonomous spacecraft or delivery drone operating in a
    resource-constrained environment. Every unnecessary movement costs fuel.
    Invalid manoeuvres represent critical operational errors. A fast, successful
    delivery earns an efficiency bonus, representing mission success under
    optimal resource usage.

What this wrapper changes:
    - Adds an extra step penalty (fuel consumption: -0.5 per step)
    - Adds an extra invalid-action penalty (-5.0 per illegal move)
    - Adds a delivery bonus (+10.0) on top of the native +20
    - Adds an efficiency bonus (+5.0) if delivery is completed in few steps

What this wrapper does NOT change:
    - The observation space
    - The action space
    - The internal dynamics of Taxi-v3
"""

import gymnasium as gym
from gymnasium import RewardWrapper

from config import (
    REWARD_INVALID_ACTION_PENALTY,
    REWARD_STEP_PENALTY,
    REWARD_DELIVERY_BONUS,
    REWARD_EFFICIENCY_THRESHOLD,
    REWARD_EFFICIENCY_BONUS,
)


class SpaceRLRewardWrapper(RewardWrapper):
    """
    Modifies the reward signal of Taxi-v3 to reflect a SpaceRL operational context.

    Reward mapping (on top of native Taxi-v3 rewards):
        Native -1  (step)         → -1 + STEP_PENALTY       = -1.5
        Native -10 (invalid move) → -10 + INVALID_PENALTY   = -15
        Native +20 (delivery)     → +20 + DELIVERY_BONUS [+ EFFICIENCY_BONUS]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._step_count   = 0
        self._last_reward  = 0.0
        # Tracked metrics for analysis
        self.invalid_action_count = 0
        self.wrapper_penalty_total = 0.0
        self.wrapper_bonus_total   = 0.0

    def reset(self, **kwargs):
        self._step_count          = 0
        self.invalid_action_count = 0
        self.wrapper_penalty_total = 0.0
        self.wrapper_bonus_total   = 0.0
        return self.env.reset(**kwargs)

    def reward(self, reward: float) -> float:
        """
        Transform the native reward.
        Called automatically by gymnasium after each env.step().

        Args:
            reward: native reward from Taxi-v3

        Returns:
            modified reward reflecting SpaceRL operational costs
        """
        self._step_count += 1
        modified = reward

        if reward == -1:
            # Normal step: add fuel consumption penalty
            extra = REWARD_STEP_PENALTY
            modified += extra
            self.wrapper_penalty_total += abs(extra)

        elif reward == -10:
            # Invalid action: add operational error penalty
            extra = REWARD_INVALID_ACTION_PENALTY
            modified += extra
            self.invalid_action_count  += 1
            self.wrapper_penalty_total += abs(extra)

        elif reward == 20:
            # Successful delivery: add mission success bonus
            modified += REWARD_DELIVERY_BONUS
            self.wrapper_bonus_total += REWARD_DELIVERY_BONUS

            # Efficiency bonus for fast deliveries
            if self._step_count <= REWARD_EFFICIENCY_THRESHOLD:
                modified += REWARD_EFFICIENCY_BONUS
                self.wrapper_bonus_total += REWARD_EFFICIENCY_BONUS

        self._last_reward = modified
        return modified

    def get_episode_wrapper_stats(self) -> dict:
        """Return per-episode wrapper-specific metrics for logging."""
        return {
            "invalid_actions"    : self.invalid_action_count,
            "wrapper_penalty"    : self.wrapper_penalty_total,
            "wrapper_bonus"      : self.wrapper_bonus_total,
        }
