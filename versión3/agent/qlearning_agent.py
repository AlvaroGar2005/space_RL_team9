"""
agent/qlearning_agent.py
------------------------
Tabular Q-learning agent for SpaceRL V3.

Identical algorithm to V1 and V2 — demonstrates that the same agent
works on a fully custom environment without any modification.
Everything is implemented from scratch: Q-table, epsilon-greedy policy,
Bellman update. No external RL libraries used.
"""

import sys
import numpy as np
import gymnasium as gym

from config import (
    LEARNING_RATE, DISCOUNT_FACTOR,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    QTABLE_PATH, N_STATES, N_ACTIONS,
)


class QLearningAgent:
    """
    Tabular Q-learning agent.
    Compatible with any Gymnasium environment with discrete spaces.
    """

    def __init__(self, env: gym.Env = None):
        if env is not None:
            n_states  = env.observation_space.n
            n_actions = env.action_space.n
        else:
            n_states  = N_STATES
            n_actions = N_ACTIONS

        self.qtable  = np.zeros((n_states, n_actions))
        self.epsilon = EPSILON_START
        print(f"[Agent] Q-table initialized  shape: ({n_states}, {n_actions})")

    def choose_action(self, state: int, env: gym.Env) -> int:
        """Epsilon-greedy: explore with prob ε, exploit otherwise."""
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        return int(np.argmax(self.qtable[state]))

    def greedy_action(self, state: int) -> int:
        """Pure greedy action for evaluation (ε = 0)."""
        return int(np.argmax(self.qtable[state]))

    def update(self, state, action, reward, next_state, done) -> float:
        """
        Q-learning Bellman update (implemented from scratch):
            Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',·) − Q(s,a)]
        Returns absolute TD error.
        """
        best_next = 0.0 if done else float(np.max(self.qtable[next_state]))
        td_target = reward + DISCOUNT_FACTOR * best_next
        td_error  = td_target - self.qtable[state, action]
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
            print(f"\n[ERROR] Q-table not found: '{path}'\n  → Train first: python train_v3.py\n")
            sys.exit(1)