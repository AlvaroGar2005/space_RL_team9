# SpaceRL — V1: Tabular Q-learning on Taxi-v3

Marta Carrascosa, Paula Martín y Álvaro García.

---

## What This Is

Baseline pipeline: standard `Taxi-v3` environment with tabular Q-learning. No modifications to the environment. This is the reference point for V2 and V3 comparisons.

---

## Environment

| Property            | Value                                   |
| State space         | 500 discrete states                     |
| Action space        | 6 actions (N, S, E, W, Pickup, Dropoff) |
| Step reward         | −1                                      |
| Invalid action      | −10                                     |
| Successful delivery | +20                                     |

---

## Project Structure

```
v1/
├── config.py                        ← All hyperparameters and file paths
├── train_taxi_qlearning.py          ← Training entry point
├── evaluate_taxi_qlearning.py       ← Evaluation + optional visual demo
├── plot_metrics.py                  ← Generate training plots
├── agent/
│   └── qlearning_agent.py
├── environment/
│   └── taxi_env.py
├── training/
│   └── train_loop.py
├── evaluation/
│   ├── eval_runner.py
│   └── plot_metrics.py
├── utils/
│   └── metrics_io.py
├── notebooks/
│   └── SpaceRL_v1.ipynb
└── results/                         ← Generated outputs (not committed)
    ├── models/qtable.npy
    ├── metrics/metrics.csv
    ├── metrics/evaluation.csv
    └── figures/training_metrics.png
```

---

## Run

```bash
# 1 — Train
python train_taxi_qlearning.py

# 2 — Evaluate
python evaluate_taxi_qlearning.py

# 3 — Plot
python plot_metrics.py
```

Or open `notebooks/SpaceRL_v1.ipynb` in JupyterLab and run all cells.

For the visual demo, see the root `README.md`.

---

## Hyperparameters (`config.py`)

| Parameter         | Default | Description                    |
| `N_EPISODES`      | 10,000  | Training episodes              |
| `MAX_STEPS`       | 200     | Max steps per episode          | 
| `LEARNING_RATE`   | 0.1     | Alpha (α)                      |
| `DISCOUNT_FACTOR` | 0.99    | Gamma (γ)                      |
| `EPSILON_START`   | 1.0     | Initial exploration rate       |
| `EPSILON_MIN`     | 0.01    | Minimum exploration rate       |
| `EPSILON_DECAY`   | 0.9995  | Decay per episode              |
| `N_EVAL_EPS`      | 100     | Evaluation episodes            |
| `DEMO_EPS`        | 3       | Episodes shown in visual demo  |
| `RENDER_DEMO`     | False   | Set True for visual simulation |

---

## Outputs

| File                                    | Description                           |
| `results/models/qtable.npy`             | Learned Q-table (500 × 6)             |
| `results/metrics/metrics.csv`           | Per-episode training data (5 columns) |
| `results/metrics/evaluation.csv`        | Greedy evaluation results             |
| `results/figures/training_metrics.png`  | 4-panel learning curves               |