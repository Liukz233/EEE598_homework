# HW3: neural Q-function (DQN) on the episodic shortest-path MDP

Kangzheng Liu, Runyang Li (EEE 598, Arizona State University)

Same 8-state slippery graph as HW2, with the Q table replaced by a small neural
network trained by DQN (experience replay + target network). The environment is
imported unchanged from `../HW2_PI_QLearning`.

## How to run

Tested with Python 3.13.16, NumPy 2.5.3, Matplotlib 3.11.2, PyTorch 2.14.1,
on CPU only (Intel Xeon @ 2.10 GHz, Linux, 1 thread, no GPU).

```
git clone https://github.com/Liukz233/EEE598_homework.git
cd EEE598_homework/HW3_Neural_Q
pip install -r requirements.txt
python run_all.py              # 5 seeds + evaluation + plots + sensitivity sweep, ~25 min
python run_all.py --no-sweep   # 5 seeds + evaluation + plots only, ~3 min
```

The HW2 folder must sit next to this one (it provides the environment). Each seed
resets the networks, optimizer, replay buffer, exploration schedule and all RNGs
(`np.random.default_rng(seed)`, `torch.manual_seed(seed)`); evaluation uses a separate
`default_rng(10000 + seed)`. PyTorch runs single-threaded with deterministic algorithms,
so a rerun reproduces `results/` exactly.

## Code

- `dqn.py` - Q-network, action masking, replay buffer, TD target, training loop,
  evaluation. Only sees `env.step()`; never reads the transition model or HW2 values.
- `run_all.py` - runs all seeds, evaluation, plots and the sweep. Uses the HW2 model
  only after training, to check results (exact value of the learned policy, Q*).
- `utils.py` - printing, CSV, plotting

## Settings (main results)

| Item | Value |
|---|---|
| Network | one-hot(s) 8 -> 32 ReLU -> 32 ReLU -> 6 (one Q per action), 1542 parameters |
| Masking | infeasible outputs set to +inf before argmin / min (cost minimization) |
| Loss, optimizer | MSE, Adam lr 1e-3 |
| Replay | capacity 10,000, warm-up 200 transitions, batch 32, one update per env step |
| Target network | hard copy every 100 learning steps |
| Exploration | eps linear 1.00 -> 0.05 over episodes 1-4000, then 0.05 |
| Training | 5000 episodes per seed, seeds 0-4 |
| Evaluation | 1000 greedy episodes (eps = 0) per seed |

## Results

- `checks.txt` - parameter count, action masks, what `env.step()` returns
- `first_batch_seed0.txt` - first mini-batch of seed 0 with its TD targets (terminal target = c)
- `seeds_eval.csv` - per seed: greedy action at every state, nominal path, 1000-episode
  evaluation (mean, std, min, max, % reaching h), exact value of the learned policy, final loss
- `q_vs_qstar.csv` - learned Q of every seed vs. Q* from HW2, and tabular Q-learning (HW2)
- `tabular_same_protocol.csv` - HW2 tabular Q-learning evaluated with the same protocol
- `seed4_curves.png/.pdf` - seed 4 episode cost (+100-episode mean) and training loss (+500-step mean)
- `sweep.csv`, `sweep_detail.csv` - one-change-at-a-time sensitivity (mask, loss, M, lr, batch)
- `huber_check.csv` - Huber fixed point vs. learned Q
- `noise_floor.txt` - Var(y|s,a) at Q* and the MSE floor along pi* (0.1216)
- `episode_costs.csv`, `losses.csv` - raw training curves, all seeds

All five seeds learn pi* = (a: A_b, b: A_e, c: A_f, d-g: go), nominal path a-b-e-g-h.
1000-episode evaluation means 8.43-8.52 (V*(a) = 8.44); exact value of every learned policy
is 8.44. Training loss settles at 0.124-0.136, the slip-noise floor of the final buffer is
0.127-0.128.

## AI disclosure

Generative AI (Claude, Anthropic) helped draft and debug the code (the DQN implementation and
the logging, table and plotting utilities) and helped draft and edit the report. The authors
derived the formulation, parameter count, regression fixed point, loss floor, Huber bias and
evaluation expectations by hand, checked them against the program output, and are responsible
for all reported results.
