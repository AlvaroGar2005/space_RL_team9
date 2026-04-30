# SpaceRL — V3: Custom SpaceMissionEnv
Marta Carrascosa, Paula Martín y Álvaro García. GitHub-> __https://github.com/AlvaroGar2005/space_RL_team9__

## What This Is

Same Q-learning agent as V1 and V2, but now running on a **fully custom environment built from scratch**. `SpaceMissionEnv` inherits from `gym.Env` and implements the complete space mission narrative natively — no wrappers, no standard environments underneath.

Requires V1 and V2 trained first: `results/metrics/metrics.csv` and `results/metrics/metrics_v2.csv` must exist for the three-way comparison.

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
| Render | Terminal text | Pygame visual window + terminal |
| Tests | Wrappers + agent | Environment + agent |

---

## The Environment — SpaceMissionEnv

### State (972 discrete states)

6 variables × 3 levels, plus 4 event types:

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

### Random events

| Event | Effect |
|---|---|
| Solar storm (15%) | Drains energy, may damage hull |
| Asteroid (5%) | Damages hull, may reduce oxygen |
| Safe zone (10%) | Passively restores energy and food |
| Normal (70%) | Slow natural resource decay |

---

## Project Structure

```
v3/
├── config.py
├── train_v3.py
├── evaluate_v3.py
├── environment/
│   └── space_mission_env.py         ← SpaceMissionEnv (gym.Env from scratch)
├── agent/
│   └── qlearning_agent.py           ← Same as V1 and V2
├── training/
│   └── train_loop_v3.py
├── evaluation/
│   ├── eval_runner_v3.py
│   └── plot_metrics_v3.py
├── utils/
│   ├── metrics_io_v3.py
│   └── comparison_v3.py             ← V1 vs V2 vs V3 comparison
├── tests/
│   └── test_environment.py
├── notebooks/
│   └── SpaceRL_v3.ipynb
└── results/                         ← Generated outputs (not committed)
    ├── models/qtable_v3.npy
    ├── metrics/metrics_v3.csv
    ├── metrics/evaluation_v3.csv
    ├── figures/training_metrics_v3.png
    ├── figures/v1_v2_v3_comparison.png
    └── comparisons/v1_v2_v3_comparison.csv
```

---

## Run

```bash
# 1 — Train
python train_v3.py

# 2 — Evaluate
python evaluate_v3.py

# 3 — Tests
python -m pytest tests/ -v
```

Or open `notebooks/SpaceRL_v3.ipynb` in JupyterLab and run all cells.

For the visual demo, see below.

---

## Hyperparameters (config.py)

| Parameter | Default | Description |
|---|---|---|
| `N_STATES` | 972 | Discrete state space size |
| `N_ACTIONS` | 6 | Number of strategic actions |
| `MAX_STEPS` | 200 | Max steps per episode |
| `MISSION_SUCCESS_STEPS` | 50 | Balanced steps needed to win |
| `N_EPISODES` | 20,000 | Training episodes |
| `LEARNING_RATE` | 0.1 | Alpha (α) |
| `DISCOUNT_FACTOR` | 0.99 | Gamma (γ) |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_DECAY` | 0.9999 | Decay per episode |
| `RENDER_DEMO` | False | Set True for visual simulation |

---

## Visual Demo

Watch the trained agent manage the spacecraft in real time.

1. Open `config.py` and set `RENDER_DEMO = True`
2. Run:

```bash
cd v3
python evaluate_v3.py
```

Two render modes are available in `eval_runner_v3.py`:

```python
# Change render_mode to "ansi" for terminal display or "human" for pygame visual window
env = SpaceMissionEnv(render_mode="human")
```

- `"human"` — pygame window with resource bars, event panel, action panel and mission progress
- `"ansi"` — text-based terminal display with bars and status indicators

You must train first. Without a Q-table there is nothing to simulate.

The agent plays `DEMO_EPS` episodes (default: 3). Change that value in `config.py` to see more.

---

## Outputs

| File | Description |
|---|---|
| `results/models/qtable_v3.npy` | Learned Q-table (972 × 6) |
| `results/metrics/metrics_v3.csv` | Per-episode training data |
| `results/metrics/evaluation_v3.csv` | Greedy evaluation results |
| `results/figures/training_metrics_v3.png` | 6-panel learning curves |
| `results/figures/v1_v2_v3_comparison.png` | Three-way comparison figure |
| `results/comparisons/v1_v2_v3_comparison.csv` | Smoothed metrics for all versions |

---

## Metrics

| Metric | Description |
|---|---|
| `reward` | Total cumulative reward per episode |
| `steps` | Steps taken per episode |
| `td_error` | Mean absolute TD error per episode |
| `epsilon` | Exploration rate at end of episode |
| `success` | 1 if mission completed successfully |
| `term_reason` | Why the episode ended (success / failure type) |