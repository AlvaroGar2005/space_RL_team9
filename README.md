# SpaceRL — Reinforcement Learning Project

Marta Carrascosa, Paula Martín y Álvaro García.

---

## What This Project Is

SpaceRL applies tabular Q-learning to a space mission narrative across three progressive versions of the same environment. Each version adds complexity on top of the previous one. The Q-learning agent is identical in all three.

| Version | Environment                      | What changes                                           |
| **V1**  | `Taxi-v3` unmodified             | Baseline pipeline                                      |
| **V2**  | `Taxi-v3` + 3 Gymnasium wrappers | Reward shaping, semantic observations, action tracking |
| **V3**  | Custom `SpaceMissionEnv`         | Full custom environment with space mission narrative   |

---

## Repo Structure

```
space_RL_team9/
├── v1/          ← Baseline Q-learning
├── v2/          ← Q-learning + wrappers
├── v3/          ← Custom environment (SpaceMissionEnv)
└── README.md    ← This file
```

Each version is self-contained with its own `config.py`, scripts, and `results/` folder. See the README inside each version for details.

---

## Setup (once, shared across all versions)

```bash
# Create virtual environment at the repo root
python -m venv venv

# Activate it
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies (each version has its own requirements.txt)
pip install -r v1/requirements.txt
pip install -r v2/requirements.txt
```

> `venv/` is excluded from Git via `.gitignore`. Never commit it.  
> If accidentally staged: `git rm -r --cached venv/`

---

## Running Each Version

### V1 — Baseline

```bash
cd v1
python train_taxi_qlearning.py
python evaluate_taxi_qlearning.py
python plot_metrics.py
```

### V2 — Wrappers

> Requires V1 to have been trained first (`v1/results/metrics/metrics.csv` must exist for comparison).

```bash
cd v2
python train_taxi_v2.py
python evaluate_taxi_v2.py
python -m pytest tests/ -v
```

---

## Visual Demo

Watch the trained agent play in the terminal. Works in both V1 and V2.

**V1**
1. Open `v1/config.py` and set `RENDER_DEMO = True`
2. Run:
```bash
cd v1
python evaluate_taxi_qlearning.py
```

**V2**
1. Open `v2/config.py` and set `RENDER_DEMO = True`
2. Run:
```bash
cd v2
python evaluate_taxi_v2.py
```

The agent plays `DEMO_EPS` episodes (default: 3). Change that value in `config.py` to see more.

> You must train first. Without a Q-table there is nothing to simulate.

---
