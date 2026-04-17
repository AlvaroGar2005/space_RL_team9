"""
agent/qlearning_agent.py
------------------------
Tabular Q-learning agent — identical logic to V1, reused without modification.

The agent is environment-agnostic: it only needs a discrete observation space
and a discrete action space. The V2 wrapped environment satisfies both.

This file is intentionally kept identical to V1's agent to demonstrate that
the wrappers, not the algorithm, are what changes in Version 2.
"""

import sys
import numpy as np
import gymnasium as gym

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    QTABLE_PATH,
)


class QLearningAgent:
    """
    Tabular Q-learning agent compatible with any Gymnasium environment
    that has discrete observation and action spaces.
    """

    def __init__(self, env: gym.Env):
        self.n_states  = env.observation_space.n
        self.n_actions = env.action_space.n
        self.qtable    = np.zeros((self.n_states, self.n_actions))
        self.epsilon   = EPSILON_START
        print(f"[Agent] Q-table initialized  shape: ({self.n_states}, {self.n_actions})")

    def choose_action(self, state: int, env: gym.Env) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        return int(np.argmax(self.qtable[state]))

    def greedy_action(self, state: int) -> int:
        """Pure greedy action (evaluation only)."""
        return int(np.argmax(self.qtable[state]))

    def update(self, state, action, reward, next_state, done) -> float:
        """
        Q-learning Bellman update.
        Returns absolute TD error.
        """
        best_next  = 0.0 if done else float(np.max(self.qtable[next_state]))
        td_target  = reward + DISCOUNT_FACTOR * best_next
        td_error   = td_target - self.qtable[state, action]
        self.qtable[state, action] += LEARNING_RATE * td_error
        return abs(td_error)

    def decay_epsilon(self):
        """Multiplicative epsilon decay after each episode."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path: str = QTABLE_PATH):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.qtable)
        print(f"[Agent] Q-table saved → {path}")

    def load(self, path: str = QTABLE_PATH):
        try:
            self.qtable = np.load(path)
            print(f"[Agent] Q-table loaded ← {path}  shape: {self.qtable.shape}")
        except FileNotFoundError:
            print(f"\n[ERROR] Q-table not found: '{path}'\n  → Train first: python train_taxi_v2.py\n")
            sys.exit(1)
