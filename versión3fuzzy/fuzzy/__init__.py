"""
fuzzy/
------
Fuzzy Logic module for SpaceRL V3.

Implements a Mamdani fuzzy inference system that evaluates the health
of the spacecraft mission based on the five resource variables:
    energy, oxygen, food, fuel, hull

Main exports:
    evaluate_mission_state()  — full fuzzy evaluation of a state
    decode_all_states()       — fuzzy MHI for all 972 states
    qtable_fuzzy_analysis()   — cross-reference Q-table with fuzzy MHI
    plot_membership_functions() — visualise fuzzy membership functions
    plot_mhi_distribution()     — MHI distribution across all states
    plot_mhi_vs_qvalue()        — fuzzy MHI vs Q-learning Q-values
    plot_fuzzy_rules_example()  — visual rule evaluation examples
"""

from .fuzzy_mission import (
    evaluate_mission_state,
    get_memberships,
    membership_low,
    membership_medium,
    membership_high,
    evaluate_rules,
    defuzzify_centroid,
    mission_health_index,
    evaluate_episode_states,
)

from .fuzzy_analysis import (
    decode_all_states,
    qtable_fuzzy_analysis,
    save_fuzzy_analysis,
    plot_membership_functions,
    plot_mhi_distribution,
    plot_mhi_vs_qvalue,
    plot_fuzzy_rules_example,
)

__all__ = [
    "evaluate_mission_state",
    "get_memberships",
    "membership_low", "membership_medium", "membership_high",
    "evaluate_rules",
    "defuzzify_centroid",
    "mission_health_index",
    "evaluate_episode_states",
    "decode_all_states",
    "qtable_fuzzy_analysis",
    "save_fuzzy_analysis",
    "plot_membership_functions",
    "plot_mhi_distribution",
    "plot_mhi_vs_qvalue",
    "plot_fuzzy_rules_example",
]
