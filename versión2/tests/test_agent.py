"""
tests/test_agent.py
--------------------
Basic tests for the Q-learning agent and training loop.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from agent import QLearningAgent
from environment import make_env_v2
from config import EPSILON_START, EPSILON_MIN


class TestQLearningAgent:

    def setup_method(self):
        self.env   = make_env_v2()
        self.agent = QLearningAgent(self.env)

    def teardown_method(self):
        self.env.close()

    def test_qtable_shape(self):
        """Q-table must be (500, 6)."""
        assert self.agent.qtable.shape == (500, 6)

    def test_qtable_initialized_to_zero(self):
        """Q-table must start at zero."""
        assert np.all(self.agent.qtable == 0.0)

    def test_epsilon_starts_at_one(self):
        assert self.agent.epsilon == pytest.approx(EPSILON_START)

    def test_choose_action_valid(self):
        """All chosen actions must be in [0, 5]."""
        self.env.reset(seed=0)
        for state in range(0, 500, 50):
            action = self.agent.choose_action(state, self.env)
            assert 0 <= action < 6

    def test_greedy_action_valid(self):
        for state in range(0, 500, 50):
            action = self.agent.greedy_action(state)
            assert 0 <= action < 6

    def test_greedy_action_selects_max_qvalue(self):
        """Greedy action must correspond to argmax of Q-values."""
        self.agent.qtable[10, 3] = 99.0
        assert self.agent.greedy_action(10) == 3

    def test_bellman_update_changes_qtable(self):
        """A single update must modify the Q-table."""
        before = self.agent.qtable[0, 0]
        self.agent.update(state=0, action=0, reward=1.0,
                          next_state=1, done=False)
        after = self.agent.qtable[0, 0]
        assert before != after

    def test_bellman_update_returns_positive_td_error(self):
        """TD error must be non-negative."""
        td = self.agent.update(0, 0, 1.0, 1, False)
        assert td >= 0.0

    def test_terminal_update_ignores_next_state(self):
        """When done=True, next state Q-values should not be used."""
        self.agent.qtable[99, :] = 999.0   # next state has huge Q-values
        self.agent.update(0, 0, 1.0, 99, done=True)
        # If done is handled correctly, td_target = reward + 0 (not 999)
        expected_target = 1.0
        expected_q      = 0.0 + 0.1 * (expected_target - 0.0)
        assert self.agent.qtable[0, 0] == pytest.approx(expected_q, abs=1e-6)

    def test_epsilon_decay(self):
        """Epsilon must decrease and never go below EPSILON_MIN."""
        for _ in range(10_000):
            self.agent.decay_epsilon()
        assert self.agent.epsilon >= EPSILON_MIN
        assert self.agent.epsilon < EPSILON_START

    def test_save_load_roundtrip(self, tmp_path):
        """Save and reload Q-table should produce identical values."""
        self.agent.qtable[0, 0] = 42.0
        path = str(tmp_path / "qtable_test.npy")
        self.agent.save(path)

        env2   = make_env_v2()
        agent2 = QLearningAgent(env2)
        agent2.load(path)
        env2.close()

        assert agent2.qtable[0, 0] == pytest.approx(42.0)
        assert np.allclose(self.agent.qtable, agent2.qtable)
