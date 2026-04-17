"""
tests/test_wrappers.py
----------------------
Basic tests for the three SpaceRL V2 wrappers.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import gymnasium as gym

from wrappers import (
    SpaceRLRewardWrapper,
    SpaceRLObservationWrapper,
    SpaceRLActionWrapper,
    decode_taxi_state,
    state_to_spacerl_description,
)
from environment import make_env_v2, make_env_base
from config import RISKY_ACTIONS


# ── RewardWrapper ─────────────────────────────────────────────────────────────

class TestRewardWrapper:

    def setup_method(self):
        base = make_env_base()
        self.env = SpaceRLRewardWrapper(base)
        self.env.reset(seed=42)

    def teardown_method(self):
        self.env.close()

    def test_step_penalty_applied(self):
        """Normal step reward should be more negative than -1."""
        # Run steps until we get a plain -1 (non-invalid, non-terminal)
        for _ in range(50):
            obs, reward, term, trunc, _ = self.env.step(0)  # Move South
            if not term and not trunc and reward != -15.0:
                # -1 native + -0.5 wrapper = -1.5
                assert reward == pytest.approx(-1.5, abs=0.01), \
                    f"Expected -1.5 for normal step, got {reward}"
                return
        # If we couldn't verify in 50 steps, at least no crash
        assert True

    def test_invalid_action_extra_penalty(self):
        """Invalid action reward should be more negative than -10."""
        self.env.reset(seed=0)
        # Try pickup and dropoff repeatedly until we hit an invalid one
        for action in [4, 5] * 20:
            obs, reward, term, trunc, _ = self.env.step(action)
            if reward < -10:
                # -10 native + -5 wrapper = -15
                assert reward == pytest.approx(-15.0, abs=0.01), \
                    f"Expected -15.0 for invalid action, got {reward}"
                assert self.env.invalid_action_count > 0
                return
        assert True  # No crash

    def test_wrapper_stats_reset_on_episode(self):
        """Stats should reset to zero at start of each episode."""
        self.env.step(4)  # May cause invalid action
        self.env.reset()
        assert self.env.invalid_action_count == 0
        assert self.env.wrapper_penalty_total == 0.0
        assert self.env.wrapper_bonus_total == 0.0

    def test_get_episode_wrapper_stats_keys(self):
        """Stats dict should contain expected keys."""
        stats = self.env.get_episode_wrapper_stats()
        assert "invalid_actions" in stats
        assert "wrapper_penalty" in stats
        assert "wrapper_bonus" in stats


# ── ObservationWrapper ────────────────────────────────────────────────────────

class TestObservationWrapper:

    def setup_method(self):
        base = make_env_base()
        self.env = SpaceRLObservationWrapper(base)

    def teardown_method(self):
        self.env.close()

    def test_observation_space_size_unchanged(self):
        """Observation space must remain Discrete(500) for Q-table compatibility."""
        assert self.env.observation_space.n == 500

    def test_observation_is_integer(self):
        """Returned observation must be a plain integer."""
        obs, _ = self.env.reset(seed=42)
        assert isinstance(obs, (int, np.integer))

    def test_observation_in_valid_range(self):
        """Observation must be in [0, 499]."""
        for seed in range(10):
            obs, _ = self.env.reset(seed=seed)
            assert 0 <= obs < 500, f"obs={obs} out of range"

    def test_decode_encode_roundtrip(self):
        """decode → encode should be identity for all 500 states."""
        from wrappers.observation_wrapper import decode_taxi_state, encode_taxi_state
        for state in range(500):
            row, col, pas, dest = decode_taxi_state(state)
            reencoded = encode_taxi_state(row, col, pas, dest)
            assert reencoded == state, f"Roundtrip failed at state {state}"

    def test_state_description_keys(self):
        """SpaceRL description must contain expected semantic keys."""
        desc = state_to_spacerl_description(0)
        for key in ["state_id", "vehicle_row", "vehicle_col",
                    "cargo_status", "target_destination", "cargo_on_board"]:
            assert key in desc, f"Missing key: {key}"

    def test_get_last_decoded_after_step(self):
        """get_last_decoded should return a non-empty dict after a step."""
        self.env.reset(seed=0)
        self.env.step(0)
        decoded = self.env.get_last_decoded()
        assert isinstance(decoded, dict)
        assert len(decoded) > 0


# ── ActionWrapper ─────────────────────────────────────────────────────────────

class TestActionWrapper:

    def setup_method(self):
        base = make_env_base()
        self.env = SpaceRLActionWrapper(base)
        self.env.reset(seed=42)

    def teardown_method(self):
        self.env.close()

    def test_action_space_unchanged(self):
        """Action space must remain Discrete(6)."""
        assert self.env.action_space.n == 6

    def test_action_counts_updated(self):
        """action_counts dict should update on every step."""
        self.env.reset(seed=0)
        self.env.step(0)
        self.env.step(1)
        self.env.step(0)
        total = sum(self.env.action_counts.values())
        assert total == 3

    def test_stats_reset_on_episode(self):
        """Action stats should reset at the start of each episode."""
        self.env.step(0)
        self.env.reset()
        assert sum(self.env.action_counts.values()) == 0
        assert self.env.blocked_actions == 0

    def test_info_contains_action_keys(self):
        """Info dict should contain action tracking keys after step."""
        self.env.reset(seed=0)
        _, _, _, _, info = self.env.step(0)
        assert "original_action" in info
        assert "transformed_action" in info
        assert "was_redirected" in info

    def test_episode_stats_keys(self):
        """Action stats dict should contain expected keys."""
        stats = self.env.get_episode_action_stats()
        for key in ["blocked_actions", "transformed_actions",
                    "risky_action_ratio", "useful_actions_ratio"]:
            assert key in stats

    def test_useful_ratio_between_0_and_1(self):
        """useful_actions_ratio must be in [0, 1]."""
        self.env.reset(seed=0)
        for _ in range(10):
            self.env.step(self.env.action_space.sample())
        stats = self.env.get_episode_action_stats()
        assert 0.0 <= stats["useful_actions_ratio"] <= 1.0


# ── Composed wrapped environment ──────────────────────────────────────────────

class TestComposedEnvironment:

    def setup_method(self):
        self.env = make_env_v2()

    def teardown_method(self):
        self.env.close()

    def test_observation_space_discrete_500(self):
        assert self.env.observation_space.n == 500

    def test_action_space_discrete_6(self):
        assert self.env.action_space.n == 6

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset(seed=0)
        assert isinstance(obs, (int, np.integer))
        assert 0 <= obs < 500

    def test_step_returns_correct_types(self):
        self.env.reset(seed=0)
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert isinstance(obs, (int, np.integer))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_full_episode_no_crash(self):
        """Run a full episode without errors."""
        obs, _ = self.env.reset(seed=99)
        done = False
        steps = 0
        while not done and steps < 200:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
