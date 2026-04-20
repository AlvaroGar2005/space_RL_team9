"""
agent/qlearning_agent.py
------------------------
Tabular Q-learning agent for discrete environments.

Responsibilities:
    - Initialize and hold the Q-table.
    - Select actions via epsilon-greedy policy.
    - Apply the Bellman update rule.
    - Decay epsilon after each episode.
    - Save / load the Q-table from disk.
"""

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

    #Policy

    def choose_action(self, state: int, env: gym.Env) -> int:
        """
        Epsilon-greedy action selection.
            - prob ε   → random (explore)
            - prob 1-ε → greedy (exploit)
        """
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        return int(np.argmax(self.qtable[state]))

    def greedy_action(self, state: int) -> int:
        """Pure greedy action (used during evaluation, ε = 0)."""
        return int(np.argmax(self.qtable[state]))

    #Learning

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Q-learning Bellman update:
            Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

        Returns:
            td_error (float): absolute TD error for this transition.
        """
        best_next  = 0.0 if done else float(np.max(self.qtable[next_state]))
        td_target  = reward + DISCOUNT_FACTOR * best_next
        td_error   = td_target - self.qtable[state, action]
        self.qtable[state, action] += LEARNING_RATE * td_error
        return abs(td_error)

    def decay_epsilon(self):
        """Apply multiplicative epsilon decay after each episode."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # Persistence

    def save(self, path: str = QTABLE_PATH):
        """Save the Q-table as a .npy file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.qtable)
        print(f"[Agent] Q-table saved → {path}")

    def load(self, path: str = QTABLE_PATH):
        """
        Load a Q-table from disk.
        Raises FileNotFoundError with a descriptive message if missing.
        """
        import sys
        try:
            self.qtable = np.load(path)
            print(f"[Agent] Q-table loaded ← {path}  shape: {self.qtable.shape}")
        except FileNotFoundError:
            print(
                f"\n[ERROR] Q-table file not found: '{path}'\n"
                "  → Train first:  python train_taxi_qlearning.py\n"
            )
            sys.exit(1)