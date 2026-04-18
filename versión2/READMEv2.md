# SpaceRL — V2: Q-learning with Gymnasium Wrappers on Taxi-v3

Marta Carrascosa, Paula Martín y Álvaro García.

---

## What This Is

Same Q-learning agent as V1, but the environment is modified through three stacked Gymnasium wrappers. The wrappers add reward shaping, semantic observation decoding, and action tracking — without touching Taxi-v3's source code.

> Requires V1 trained first: `v1/results/metrics/metrics.csv` must exist.

---

## What Changes vs V1

| Element           | V1                 | V2                                     |
| Algorithm         | Tabular Q-learning | Identical                              |
| Q-table shape     | (500, 6)           | (500, 6) — unchanged                   |
| Reward signal     | Native             | Shaped (+fuel cost, +efficiency bonus) |
| Observation       | Opaque integer     | Semantically decoded and re-encoded    |
| Actions           | Direct             | Tracked with safety-lock layer         |
| Training metrics  | 5 columns          | 12 columns                             |
| Tests             | None               | Unit tests for all wrappers and agent  |

---

## The Three Wrappers

**SpaceRLRewardWrapper** — adjusts reward values to reflect space mission costs:

| Native reward  | V2 reward  | Reason                             |
| −1 (step)      | −1.5       | Fuel consumption                   |
| −10 (invalid)  | −15        | Critical operational error         |
| +20 (delivery) | +30 or +35 | Mission success + efficiency bonus |

**SpaceRLObservationWrapper** — decodes the opaque state integer (0–499) into 4 semantic fields (`vehicle_row`, `vehicle_col`, `cargo_status`, `target_destination`) and re-encodes it. Q-table shape stays (500, 6).

**SpaceRLActionWrapper** — tracks risky operations (Pickup, Dropoff), redirects repeated invalid actions (safety-lock), and records per-episode stats.

**Stacking order:**
```
Taxi-v3 → SpaceRLObservationWrapper → SpaceRLRewardWrapper → SpaceRLActionWrapper
```

---

## Project Structure

```
v2/
├── config.py
├── train_taxi_v2.py
├── evaluate_taxi_v2.py
├── wrappers/
│   ├── reward_wrapper.py
│   ├── observation_wrapper.py
│   └── action_wrapper.py
├── agent/
│   └── qlearning_agent.py           ← Same as V1
├── environment/
│   └── taxi_env_v2.py               ← make_env_v2() stacks all wrappers
├── training/
│   └── train_loop_v2.py
├── evaluation/
│   ├── eval_runner_v2.py
│   └── plot_metrics_v2.py
├── utils/
│   ├── metrics_io_v2.py
│   └── comparison.py
├── tests/
│   ├── test_wrappers.py
│   └── test_agent.py
├── notebooks/
│   └── SpaceRL_v2.ipynb
└── results/                         ← Generated outputs (not committed)
    ├── models/qtable_v2.npy
    ├── metrics/metrics_v2.csv
    ├── metrics/evaluation_v2.csv
    ├── figures/training_metrics_v2.png
    ├── figures/v1_vs_v2_comparison.png
    └── comparisons/v1_vs_v2_comparison.csv
```

---

## Run

```bash
# 1 — Train
python train_taxi_v2.py

# 2 — Evaluate
python evaluate_taxi_v2.py

# 3 — Tests
python -m pytest tests/ -v
```

Or open `notebooks/SpaceRL_v2.ipynb` in JupyterLab and run all cells.

For the visual demo, see the root `README.md`.

---

## Hyperparameters (`config.py`)

| Parameter                       | Default | Description                           |
| `N_EPISODES`                    | 10,000  | Training episodes                     |
| `LEARNING_RATE`                 | 0.1     | Alpha (α)                             |
| `DISCOUNT_FACTOR`               | 0.99    | Gamma (γ)                             |
| `EPSILON_START`                 | 1.0     | Initial exploration rate              |
| `EPSILON_DECAY`                 | 0.9995  | Decay per episode                     |
| `REWARD_STEP_PENALTY`           | −0.5    | Extra penalty per step                |
| `REWARD_INVALID_ACTION_PENALTY` | −5.0    | Extra penalty for invalid moves       |
| `REWARD_DELIVERY_BONUS`         | +10.0   | Extra delivery bonus                  |    
| `REWARD_EFFICIENCY_BONUS`       | +5.0    | Bonus for fast deliveries             |
| `REWARD_EFFICIENCY_THRESHOLD`   | 15      | Steps threshold for efficiency bonus  |
| `RENDER_DEMO`                   | False   | Set True for visual simulation        |

---

## Outputs

| File                                          | Description                            |
| `results/models/qtable_v2.npy`                | Learned Q-table (500 × 6)              | 
| `results/metrics/metrics_v2.csv`              | Per-episode training data (12 columns) |
| `results/metrics/evaluation_v2.csv`           | Greedy evaluation with wrapper stats   |
| `results/figures/training_metrics_v2.png`     | 8-panel learning curves                |
| `results/figures/v1_vs_v2_comparison.png`     | V1 vs V2 comparison figure             |
| `results/comparisons/v1_vs_v2_comparison.csv` | Smoothed metrics aligned by episode    |

---

## Wrapper-Specific Metrics

| Metric                 | Wrapper       | Description                       |
| `invalid_actions`      | RewardWrapper | Invalid moves per episode         |
| `wrapper_penalty`      | RewardWrapper | Total extra penalty applied       |
| `wrapper_bonus`        | RewardWrapper | Total extra bonus applied         |
| `blocked_actions`      | ActionWrapper | Actions redirected by safety-lock |
| `risky_action_ratio`   | ActionWrapper | Proportion of risky operations    |
| `useful_actions_ratio` | ActionWrapper | Proportion of non-blocked actions |