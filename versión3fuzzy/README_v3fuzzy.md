# SpaceRL — V3: SpaceMissionEnv + Fuzzy Logic
Marta Carrascosa, Paula Martín y Álvaro García. GitHub-> __https://github.com/AlvaroGar2005/space_RL_team9__

## What This Is

Same Q-learning agent as V1 and V2, now running on a **fully custom environment built from scratch** with an added **Fuzzy Logic inference system** that evaluates the health of the mission in real time.

`SpaceMissionEnv` inherits from `gym.Env` and implements the complete space mission narrative natively. The fuzzy module applies concepts from **Tema 5: Razonamiento con Incertidumbre** — membership functions, IF-THEN rules and centroid defuzzification — to produce a **Mission Health Index (MHI)** for every state.

Requires V1 and V2 trained first for the three-way comparison.

---

## What Changes vs V2

| Element | V2 | V3 |
|---|---|---|
| Algorithm | Tabular Q-learning | Identical |
| Base environment | Taxi-v3 + wrappers | SpaceMissionEnv (from scratch) |
| Q-table shape | (500, 6) | (972, 6) |
| State space | 500 re-encoded integers | 972 semantic discrete states |
| Reward signal | Shaped via RewardWrapper | Designed natively |
| Events | None | Solar storm, asteroid, safe zone |
| Render | None | Pygame visual window + terminal |
| **Fuzzy Logic** | **No** | **Yes — Mission Health Index** |
| Tests | Wrappers + agent | Environment + agent + fuzzy system |

---

## The Environment — SpaceMissionEnv

### State (972 discrete states)

| Variable | Levels |
|---|---|
| `energy` | 0=critical, 1=stable, 2=optimal |
| `oxygen` | 0=critical, 1=stable, 2=optimal |
| `food` | 0=critical, 1=stable, 2=optimal |
| `fuel` | 0=critical, 1=stable, 2=optimal |
| `hull` | 0=damaged, 1=stable, 2=good |
| `event` | 0=normal, 1=solar_storm, 2=asteroid, 3=safe_zone |

### Actions (6)

| ID | Action | Effect |
|---|---|---|
| 0 | `life_support` | +oxygen, +food, −energy |
| 1 | `propulsion` | −fuel |
| 2 | `gather_resources` | +food, +fuel |
| 3 | `repair_ship` | +hull, −energy |
| 4 | `power_saving` | +energy |
| 5 | `move_to_safe_zone` | +energy, +oxygen, +food, −fuel |

### Reward shaping

| Event | Reward |
|---|---|
| Survive a step | +2 |
| All resources balanced | +5 |
| Safe zone well used | +5 |
| Critical resource at 0 | −2 |
| Severe hull damage | −5 |
| Mission failure | −20 |
| Mission success | +50 |

---

## Fuzzy Logic — Mission Health Index

Based on **Tema 5: Razonamiento con Incertidumbre**.

### Membership functions (triangular, universe [0, 2])

| Set | Function | Description |
|---|---|---|
| LOW | trimf(0, 0, 1) | Critical resource |
| MEDIUM | trimf(0, 1, 2) | Stable resource |
| HIGH | trimf(1, 2, 2) | Optimal resource |

### IF-THEN Rules (Mamdani)

| Rule | Antecedent | Consequent |
|---|---|---|
| R1 | IF energy is LOW | RISK is CRITICAL |
| R2 | IF oxygen is LOW | RISK is CRITICAL |
| R3 | IF hull is LOW | RISK is CRITICAL |
| R4 | IF food is LOW OR fuel is LOW | RISK is HIGH |
| R5 | IF all resources are MEDIUM | RISK is MODERATE |
| R6 | IF all resources are HIGH | RISK is SAFE |
| R7 | IF energy AND oxygen AND hull are HIGH | RISK is SAFE |

**Aggregation:** max operator
**Defuzzification:** centroid method
**Mission Health Index (MHI)** = 1 − (crisp_risk / 3) ∈ [0, 1]

---
