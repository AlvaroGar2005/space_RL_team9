"""
wrappers/observation_wrapper.py
--------------------------------
SpaceRL ObservationWrapper for Taxi-v3.

Narrative:
    Instead of receiving an opaque integer state (0-499), the autonomous system
    now receives a semantically decoded observation that maps directly to
    operational parameters: vehicle position (row, col), cargo status
    (where the passenger is or if already aboard), and target destination.

What this wrapper changes:
    - Decodes the native integer state into 4 semantic components:
        (taxi_row, taxi_col, passenger_location, destination)
    - Re-encodes these 4 components as a NEW discrete integer (0-499)
      so that the Q-table remains exactly the same shape and the agent
      stays fully tabular.

What this wrapper does NOT change:
    - The action space
    - The reward signal
    - The internal dynamics of Taxi-v3
    - The Q-table shape (observation space is still Discrete(500))

Design note:
    The decode → re-encode step produces a bijective mapping, so NO information
    is lost. The key academic contribution is that the agent (and the notebook
    analysis) can now inspect observations in human-readable form, enabling
    SpaceRL semantic analysis (e.g. "vehicle at row 2, col 3, cargo on board,
    heading to docking bay 1").
"""

import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Discrete

from config import OBS_GRID_SIZE, OBS_N_PASSENGER, OBS_N_DEST


# ── Decode / encode helpers (module-level so tests can import them) ──────────

def decode_taxi_state(state: int) -> tuple:
    """
    Decode a Taxi-v3 integer state into semantic components.

    Taxi-v3 encodes state as:
        state = dest + 4 * (passenger + 5 * (col + 5 * row))

    Returns:
        (taxi_row, taxi_col, passenger_location, destination)
        - taxi_row          : 0-4
        - taxi_col          : 0-4
        - passenger_location: 0-4  (0-3 = locations, 4 = in taxi)
        - destination       : 0-3
    """
    dest      = state % 4;       state //= 4
    passenger = state % 5;       state //= 5
    col       = state % 5;       state //= 5
    row       = state
    return int(row), int(col), int(passenger), int(dest)


def encode_taxi_state(row: int, col: int, passenger: int, dest: int) -> int:
    """
    Re-encode semantic components back into a Taxi-v3 integer state.
    Inverse of decode_taxi_state.
    """
    return int(dest + 4 * (passenger + 5 * (col + 5 * row)))


# SpaceRL semantic labels for notebook analysis
PASSENGER_LABELS = {
    0: "Location R (Red)",
    1: "Location G (Green)",
    2: "Location Y (Yellow)",
    3: "Location B (Blue)",
    4: "On board",
}

DESTINATION_LABELS = {
    0: "Docking Bay R",
    1: "Docking Bay G",
    2: "Docking Bay Y",
    3: "Docking Bay B",
}


def state_to_spacerl_description(state: int) -> dict:
    """
    Return a human-readable SpaceRL description of a Taxi-v3 state.
    Used in the notebook for semantic analysis.
    """
    row, col, passenger, dest = decode_taxi_state(state)
    return {
        "state_id"           : state,
        "vehicle_row"        : row,
        "vehicle_col"        : col,
        "cargo_status"       : PASSENGER_LABELS.get(passenger, str(passenger)),
        "target_destination" : DESTINATION_LABELS.get(dest, str(dest)),
        "cargo_on_board"     : passenger == 4,
    }


# ── Wrapper ──────────────────────────────────────────────────────────────────

class SpaceRLObservationWrapper(ObservationWrapper):
    """
    Decodes the Taxi-v3 integer state into semantic components and
    re-encodes it as a new integer.

    The observation space size remains Discrete(500), so the Q-table
    shape is unchanged and the agent is still fully tabular.

    The main benefit is interpretability: we can call decode_taxi_state()
    on any observation to get a human-readable SpaceRL state description.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Observation space unchanged in size — still discrete 500
        n = OBS_GRID_SIZE * OBS_GRID_SIZE * OBS_N_PASSENGER * OBS_N_DEST
        self.observation_space = Discrete(n)

        # Track decoded observations for episode analysis
        self._decoded_obs_history = []

    def reset(self, **kwargs):
        self._decoded_obs_history = []
        return super().reset(**kwargs)

    def observation(self, obs: int) -> int:
        """
        Decode native state → semantic tuple → re-encode as integer.
        The bijective mapping preserves all information.
        """
        row, col, passenger, dest = decode_taxi_state(obs)
        new_obs = encode_taxi_state(row, col, passenger, dest)

        # Store decoded form for analysis
        self._decoded_obs_history.append((row, col, passenger, dest))
        return new_obs

    def get_last_decoded(self) -> dict:
        """Return the last observation in SpaceRL semantic form."""
        if not self._decoded_obs_history:
            return {}
        row, col, passenger, dest = self._decoded_obs_history[-1]
        return state_to_spacerl_description(
            encode_taxi_state(row, col, passenger, dest)
        )
