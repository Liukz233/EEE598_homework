# HW2 — Episodic optimal path: policy iteration vs. Q-learning

Solves the 8-state stochastic shortest-path MDP (Fig. 1 of the assignment) two ways:
model-based **policy iteration** and model-free **tabular Q-learning** (seeds 0–4).

## Software

Tested with Python 3.13, NumPy 2.5, Matplotlib 3.11 (exact versions are written to
`results/run_info.txt`). Should also run on Python ≥ 3.9 with recent NumPy/Matplotlib.

```bash
pip install -r requirements.txt
python run_all.py              # everything, ~40 s on a laptop CPU
python run_all.py --no-sweep   # skip the hyperparameter sweep, ~5 s
```

All randomness comes from `np.random.default_rng(seed)`, one generator per seed, so
the results in `results/` are reproduced exactly.

## Files

| File | What it does |
|---|---|
| `mdp.py` | States, actions, edge costs, transition model `P(s'|s,a)` |
| `policy_iteration.py` | Exact policy evaluation (backward sweep on the DAG), greedy improvement, stability test |
| `q_learning.py` | Environment (hides the model behind `step()`), ε-greedy tabular Q-learning, greedy rollouts |
| `run_all.py` | Runs every part of the assignment and writes `results/` |
| `utils.py` | Formatting, CSV output, plotting (boilerplate) |

## Conventions

- Cost of a step = weight of the edge **actually traversed** (a slip pays the slipped edge's cost).
- Minimize undiscounted total cost, γ = 1, V(h) = 0. d, e, f, g have one action, `go`.
- Q-learning target: `c + min_a' Q(s', a')`, with bootstrap 0 when `s' = h`.

## Q-learning settings (main results)

| Setting | Value |
|---|---|
| Episodes per seed | 5000, seeds 0–4, Q₀ = 0 reset for each seed |
| Learning rate | α = 1 / N(s,a)^0.6 (per state–action visit count) |
| Exploration | ε linear 1.0 → 0.05 over the first 3000 episodes, then fixed |
| Ties | broken uniformly at random |
| Evaluation | ε = 0 greedy policy, evaluated exactly with the known model, plus 20 000 greedy rollouts |

## Outputs (`results/`)

| File | Content |
|---|---|
| `policy_iteration.txt` | Every PI round: V^π for all states, lookahead Q, argmin, stability test |
| `hand_check_seed0.txt` | First five episodes of seed 0: target, TD error, Q update per step |
| `qlearning_seeds.csv` | Greedy policy per seed, exact V^π(a), gap to V*(a), optimal? |
| `q_table_vs_qstar.csv` | Learned Q of every seed vs. Q* from policy iteration |
| `seed4_learning_curve.png/.pdf` | Seed 4 episode cost and 100-episode rolling mean |
| `sweep.csv` | Sensitivity to α, final ε and training length |
| `episode_costs.csv` | Raw training episode costs, all seeds |

## Main results

- Policy iteration converges after 2 evaluations: π\* = (a: A_b, b: A_e, c: A_f),
  V\* = (a 8.44, b 6.2, c 7.4, d 4, e 4, f 3, g 1, h 0).
- All five Q-learning seeds recover π\*; exact gap to V\*(a) is 0.

## Disclosure

Generative AI (Claude, Anthropic) assisted with interpreting the assignment, drafting
and debugging the code (in particular the logging, table and plotting helpers in
`utils.py`), and cross-checking hand calculations. The team reviewed the code and results.
