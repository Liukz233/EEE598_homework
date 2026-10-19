# HW2: policy iteration vs. Q-learning on the episodic shortest-path MDP

Kangzheng Liu, Runyang Li (EEE 598, Arizona State University)

Model-based policy iteration and model-free tabular Q-learning (seeds 0-4) on the
8-state slippery graph from Fig. 1 of the assignment.

## How to run

Tested with Python 3.13.16, NumPy 2.5.3, Matplotlib 3.11.2.

```
pip install -r requirements.txt
python run_all.py              # everything, ~40 s
python run_all.py --no-sweep   # skip the hyperparameter sweep, ~5 s
```

Each seed uses its own `np.random.default_rng(seed)`, so the numbers in `results/` come
out exactly the same on a rerun.

## Code

- `mdp.py` - states, actions, edge costs, slip model P(s'|s,a)
- `policy_iteration.py` - exact policy evaluation (one backward sweep, the graph is a DAG),
  greedy improvement, stability check
- `q_learning.py` - environment (model hidden behind `step()`), eps-greedy Q-learning,
  greedy rollouts
- `run_all.py` - runs all parts and writes `results/`
- `utils.py` - printing, CSV, plotting

Conventions: a step costs the weight of the edge actually taken (a slip pays the
slipped edge), gamma = 1, V(h) = 0, terminal bootstrap = 0. d, e, f, g have a single
action `go`.

Q-learning settings: 5000 episodes per seed, Q0 = 0 reset per seed,
alpha = 1/N(s,a)^0.6, eps linear 1.0 -> 0.05 over the first 3000 episodes, random
tie-breaking. Evaluation uses eps = 0.

## Results

- `policy_iteration.txt` - every PI round (values, lookahead Q, argmin, stability)
- `hand_check_seed0.txt` - first five episodes of seed 0, update by update
- `qlearning_seeds.csv` - greedy policy of each seed, exact V(a), gap to V*(a)
- `q_table_vs_qstar.csv` - learned Q vs. Q* from PI
- `seed4_learning_curve.png/.pdf` - episode cost + 100-episode rolling mean, seed 4
- `sweep.csv` - sensitivity to alpha, final eps and number of episodes
- `episode_costs.csv` - raw training costs, all seeds

PI stops after two evaluations with pi* = (a: A_b, b: A_e, c: A_f) and V*(a) = 8.44.
All five Q-learning seeds give the same greedy policy (gap 0).

## AI disclosure

Generative AI (Claude, Anthropic) was used to help draft and debug the code, including the
policy-iteration and Q-learning implementations and the logging, table and plotting
utilities, and to help draft and edit the report. The team checked the policy-iteration
values by hand, reviewed the code, reproduced all results, and is responsible for everything
reported.
