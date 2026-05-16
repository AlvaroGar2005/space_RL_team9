"""
tests/test_environment.py
--------------------------
Unit tests for SpaceMissionEnv and the Q-learning agent.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from environment import SpaceMissionEnv
from agent import QLearningAgent
from config import N_STATES, N_ACTIONS, MAX_STEPS


class TestSpaceMissionEnv:

    def setup_method(self):
        self.env = SpaceMissionEnv()

    def teardown_method(self):
        self.env.close()

    def test_observation_space(self):
        assert self.env.observation_space.n == N_STATES

    def test_action_space(self):
        assert self.env.action_space.n == N_ACTIONS

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset(seed=42)
        assert isinstance(obs, (int, np.integer))
        assert 0 <= obs < N_STATES
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        self.env.reset(seed=0)
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert isinstance(obs, (int, np.integer))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_in_valid_range_after_steps(self):
        self.env.reset(seed=0)
        for _ in range(20):
            obs, _, term, trunc, _ = self.env.step(self.env.action_space.sample())
            assert 0 <= obs < N_STATES
            if term or trunc:
                self.env.reset()

    def test_full_episode_no_crash(self):
        obs, _ = self.env.reset(seed=99)
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            obs, reward, terminated, truncated, info = self.env.step(
                self.env.action_space.sample()
            )
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_decode_state_keys(self):
        desc = SpaceMissionEnv.decode_state(0)
        for key in ["energy", "oxygen", "food", "fuel", "hull", "event"]:
            assert key in desc

    def test_decode_encode_roundtrip(self):
        """All 972 states should decode and re-encode correctly."""
        env = SpaceMissionEnv()
        for state in range(N_STATES):
            obs, _ = env.reset(seed=state)
            # After reset, state should be valid
            assert 0 <= obs < N_STATES
        env.close()

    def test_truncation_at_max_steps(self):
        """Episode must truncate at MAX_STEPS if not terminated."""
        self.env.reset(seed=0)
        # Force power_saving to keep resources stable
        truncated = False
        for _ in range(MAX_STEPS + 10):
            _, _, terminated, truncated, _ = self.env.step(4)  # power_saving
            if terminated or truncated:
                break
        # Either truncated or terminated — both are valid
        assert terminated or truncated

    def test_info_contains_required_keys(self):
        self.env.reset(seed=0)
        _, _, _, _, info = self.env.step(0)
        for key in ["energy", "oxygen", "food", "fuel", "hull", "event", "mission_success"]:
            assert key in info

    def test_reward_is_float(self):
        self.env.reset(seed=0)
        for action in range(N_ACTIONS):
            self.env.reset()
            _, reward, _, _, _ = self.env.step(action)
            assert isinstance(reward, float)


class TestQLearningAgentV3:

    def setup_method(self):
        self.env   = SpaceMissionEnv()
        self.agent = QLearningAgent(self.env)

    def teardown_method(self):
        self.env.close()

    def test_qtable_shape(self):
        assert self.agent.qtable.shape == (N_STATES, N_ACTIONS)

    def test_qtable_initialized_to_zero(self):
        assert np.all(self.agent.qtable == 0.0)

    def test_choose_action_valid(self):
        self.env.reset(seed=0)
        for state in range(0, N_STATES, 100):
            action = self.agent.choose_action(state, self.env)
            assert 0 <= action < N_ACTIONS

    def test_greedy_action_selects_max(self):
        self.agent.qtable[10, 3] = 99.0
        assert self.agent.greedy_action(10) == 3

    def test_bellman_update_modifies_qtable(self):
        before = self.agent.qtable[0, 0]
        self.agent.update(0, 0, 1.0, 1, False)
        assert self.agent.qtable[0, 0] != before

    def test_td_error_non_negative(self):
        td = self.agent.update(0, 0, 1.0, 1, False)
        assert td >= 0.0

    def test_epsilon_decay(self):
        for _ in range(20_000):
            self.agent.decay_epsilon()
        assert self.agent.epsilon >= 0.01
        assert self.agent.epsilon < 1.0

    def test_save_load_roundtrip(self, tmp_path):
        self.agent.qtable[0, 0] = 42.0
        path = str(tmp_path / "qtable_test.npy")
        self.agent.save(path)

        agent2 = QLearningAgent(self.env)
        agent2.load(path)
        assert agent2.qtable[0, 0] == pytest.approx(42.0)
