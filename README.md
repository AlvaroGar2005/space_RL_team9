# SpaceRL — Reinforcement Learning for a Space Mission Environment

Marta Carrascosa, Paula Martín y Álvaro García   GitHub: https://github.com/AlvaroGar2005/space_RL_team9

---

## Overview

**SpaceRL** is a reinforcement learning project that studies how a tabular Q-learning agent behaves as the environment becomes progressively more complex.

The project starts with a standard `Taxi-v3` baseline, then introduces space-mission-inspired wrappers, and finally evolves into a fully custom environment where the agent must manage spacecraft resources, react to random events, and keep the mission alive.

The main goal is not only to train an agent, but to compare how environment design, reward shaping, state representation, and uncertainty reasoning affect learning performance.

---

## Project Evolution

The project is divided into three versions:

| Version | Main idea | Environment | Agent |
| **V1** | Baseline experiment | Standard `Taxi-v3` | Tabular Q-learning |
| **V2** | Space-inspired adaptation | `Taxi-v3` with custom Gymnasium wrappers | Same Q-learning agent |
| **V3** | Final custom mission | `SpaceMissionEnv` built from scratch | Same Q-learning agent + fuzzy mission health evaluation |

Keeping the Q-learning algorithm consistent across versions makes it possible to focus the comparison on the environment and reward design rather than on changes in the learning method.

---

## Version Summary

### V1 — Baseline Q-learning on Taxi-v3

V1 establishes the reference point.  
The agent learns with standard tabular Q-learning in the original `Taxi-v3` environment, without modifications to rewards, observations, or actions.

This version provides the baseline metrics used later to evaluate whether the following versions improve learning behavior or add useful complexity.

---

### V2 — Taxi-v3 with SpaceRL Wrappers

V2 keeps the same Q-learning agent but modifies the environment through three Gymnasium wrappers:

- **Reward wrapper**: adapts rewards to better represent mission costs and bonuses.
- **Observation wrapper**: decodes the original Taxi state into more interpretable semantic information.
- **Action wrapper**: tracks risky actions and adds a safety-lock mechanism.

This version transforms the original Taxi task into a more space-mission-like scenario without changing the underlying Gymnasium environment.

---

### V3 — Custom SpaceMissionEnv

V3 replaces Taxi-v3 completely with a custom environment built from scratch.

The agent must manage a spacecraft through six possible actions while maintaining key mission resources:

- energy
- oxygen
- food
- fuel
- hull integrity
- external event state

The environment includes random events such as solar storms, asteroid impacts, safe zones, and normal mission decay. Rewards are designed natively around survival, balance, damage, failure, and mission success.

V3 also includes a visual simulation mode using Pygame and a terminal-based render mode.

---

## Fuzzy Logic Extension

The final version adds a fuzzy logic system to estimate the health of the mission in real time.

The fuzzy module uses resource levels such as energy, oxygen, food, fuel, and hull condition to compute a **Mission Health Index (MHI)** between 0 and 1.

This adds an uncertainty-reasoning layer to the project, connecting the reinforcement learning environment with fuzzy inference concepts such as:

- membership functions
- IF-THEN rules
- risk aggregation
- centroid defuzzification

The MHI does not replace the Q-learning agent, but provides an interpretable indicator of how safe or risky the current mission state is.

---

## Learning Algorithm

All versions use **tabular Q-learning**.

The agent learns a Q-table that estimates the expected future reward of taking each action in each discrete state. Exploration is controlled through an epsilon-greedy policy, where epsilon decays over training so that the agent gradually shifts from exploration to exploitation.

The use of the same algorithm across the three versions allows a clearer comparison between:

- a standard Gymnasium environment
- a wrapped environment with shaped behavior
- a fully custom environment with semantic states and mission events

---

## Repository Structure

```text
space_RL_team9/
├── v1/
│   ├── train_taxi_qlearning.py
│   ├── evaluate_taxi_qlearning.py
│   ├── agent/
│   ├── environment/
│   ├── training/
│   ├── evaluation/
│   └── results/
│
├── v2/
│   ├── train_taxi_v2.py
│   ├── evaluate_taxi_v2.py
│   ├── wrappers/
│   ├── agent/
│   ├── environment/
│   ├── training/
│   ├── evaluation/
│   ├── tests/
│   └── results/
│
├── v3/
│   ├── train_v3.py
│   ├── evaluate_v3.py
│   ├── environment/
│   ├── agent/
│   ├── training/
│   ├── evaluation/
│   ├── tests/
│   └── results/
│
└── README.md