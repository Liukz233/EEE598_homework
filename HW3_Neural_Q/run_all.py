"""
Reproduces every number and figure in the HW3 report.

    python run_all.py              # 5 seeds, evaluation, plots, sensitivity sweep (~20 min)
    python run_all.py --no-sweep   # 5 seeds, evaluation, plots only (~3 min)

Outputs go to results/. Training lives in dqn.py and never sees the model;
this script uses the HW2 model and policy iteration only AFTER training, to
check and compare (exact value of the learned policy, Q*, noise floors).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import dqn
from dqn import ACTIONS, BASE_CFG, NONTERMINAL, STATES, evaluate, greedy, q_values, train
from mdp import BACKWARD_ORDER, BRANCHES, NEXT, START, TERMINAL, actions, transitions
from policy_iteration import PI0_STOCHASTIC, deterministic, lookahead, policy_iteration
from policy_iteration import evaluate as exact_value
import q_learning as tabular
import utils

SEEDS = [0, 1, 2, 3, 4]
GAP_TOL = 1e-6
DECISIONS = [(s, a) for s in 'abc' for a in actions(s)]

torch.set_num_threads(1)                  # CPU, single thread -> bit-identical reruns
torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------- analysis (uses the model)
def fixed_point(delta=None):
    """Fixed point of the regression the target network asks for, solved backward on
    the DAG with the model. delta=None: MSE -> mean of y = Q*. delta: Huber -> root of
    E[psi(y - q)] = 0. For checking only, never used in training."""
    Q, V = {}, {TERMINAL: 0.0}
    for s in BACKWARD_ORDER:
        Q[s] = {}
        for a in actions(s):
            ys = [(p, c + V[s2]) for p, s2, c in transitions(s, a)]
            if delta is None:
                q = sum(p * y for p, y in ys)
            else:
                lo, hi = min(y for _, y in ys), max(y for _, y in ys)
                for _ in range(100):      # E[psi] is decreasing in q -> bisection
                    q = 0.5 * (lo + hi)
                    g = sum(p * float(np.clip(y - q, -delta, delta)) for p, y in ys)
                    lo, hi = (q, hi) if g > 0 else (lo, q)
            Q[s][a] = q
        V[s] = min(Q[s].values())
    return Q


def target_variance(Q_star):
    """Var(y | s, a) at the fixed point: the part of the MSE no network can remove."""
    V = {s: min(Q_star[s].values()) for s in Q_star}
    V[TERMINAL] = 0.0
    out = {}
    for s in Q_star:
        for a in Q_star[s]:
            ys = [(p, c + V[s2]) for p, s2, c in transitions(s, a)]
            m = sum(p * y for p, y in ys)
            out[s, a] = sum(p * (y - m) ** 2 for p, y in ys)
    return out


def nominal_path(policy):
    """Greedy path from a when every intended transition succeeds."""
    s, path = START, [START]
    while s != TERMINAL:
        a = policy[s]
        s = a[2:] if s in BRANCHES else NEXT[s]
        path.append(s)
    return '-'.join(path)


# ---------------------------------------------------------------- experiments
def checks(out):
    net = dqn.QNet(BASE_CFG['hidden'])
    n_par = sum(p.numel() for p in net.parameters())
    assert n_par == 1542
    assert not hasattr(dqn, 'transitions') and not hasattr(dqn, 'policy_iteration')
    env = dqn.PathEnv(np.random.default_rng(0))
    env.reset()
    sample = env.step('A_b')
    rows = [[s, ', '.join(actions(s)) or '(terminal)',
             ''.join('1' if v else '0' for v in dqn.VALID[dqn.S_IDX[s]].tolist())] for s in STATES]
    text = (f"Trainable parameters: {n_par}  (8*32+32 + 32*32+32 + 32*6+6)\n"
            f"Output slots: {ACTIONS}\n\n"
            + utils.text_table(["state", "feasible actions", "mask over slots"], rows)
            + f"\n\nenv.step('A_b') from a returns only (s', cost, done) = {sample}\n"
              "dqn.py does not import transitions() or policy_iteration.\n")
    utils.write_text(out / "checks.txt", text)
    print(text)


def main_runs(out, Q_star, V_star, var_y):
    seeds_rows, q_runs, costs_all, loss_all, tab_rows, tab_q = [], [], [], [], [], []
    floor_pi = (var_y['a', 'A_b'] + 0.8 * var_y['b', 'A_e'] + 0.2 * var_y['c', 'A_f']) / 4

    for seed in SEEDS:
        net, costs, losses, info = train(seed, BASE_CFG, log_first_batch=(seed == 0))
        if seed == 0:
            utils.write_text(out / "first_batch_seed0.txt",
                             "First mini-batch of seed 0 (32 samples; distinct transitions "
                             "shown). Target net = initial online net.\n\n"
                             + utils.first_batch_report(info['first_batch'], STATES, ACTIONS))
        pol_rng = np.random.default_rng(20_000 + seed)
        policy = {s: greedy(net, s, pol_rng) for s in NONTERMINAL}
        ev, reach = evaluate(lambda s, r: greedy(net, s, r), seed)
        V_pi = exact_value(deterministic({s: policy[s] for s in BRANCHES}))[START]
        gap = V_pi - V_star[START]
        buf = info['buffer']
        floor_buf = np.mean([var_y[STATES[s], ACTIONS[a]] for s, a in
                             zip(buf.s[:buf.n], buf.a[:buf.n])])
        seeds_rows.append([seed, *[policy[s] for s in NONTERMINAL], nominal_path(policy),
                           ev.mean(), ev.std(ddof=1), ev.min(), ev.max(), 100 * reach,
                           V_pi, gap, 'yes' if abs(gap) < GAP_TOL else 'no',
                           losses[-1000:].mean(), floor_buf, info['steps'], info['copies'],
                           info['seconds']])
        q_runs.append(q_values(net))
        costs_all.append(costs)
        loss_all.append(losses)
        if seed == 4:
            utils.plot_seed_curves(costs, losses, V_star[START], floor_pi,
                                   out / "seed4_curves", eps_end_episode=BASE_CFG['eps_decay_end'],
                                   copy_every=BASE_CFG['target_every'])

        # tabular Q-learning from HW2, same seed, same evaluation protocol
        t0 = time.perf_counter()
        Qt, *_ = tabular.train(seed, n_episodes=5000, eps_start=1.0, eps_end=0.05,
                               decay_episodes=3000, alpha='visit', omega=0.6)
        t_tab = time.perf_counter() - t0
        evt, _ = evaluate(lambda s, r: tabular.argmin_random(Qt[s], r), seed)
        tpol = tabular.greedy_policy(Qt, np.random.default_rng(20_000 + seed))
        tab_rows.append([seed, tpol['a'], tpol['b'], tpol['c'], evt.mean(), evt.std(ddof=1),
                         exact_value(deterministic(tpol))[START], t_tab])
        tab_q.append(Qt)
        print(f"seed {seed}: DQN policy {[policy[s] for s in 'abc']}, eval {ev.mean():.3f}, "
              f"loss {losses[-1000:].mean():.4f}, {info['seconds']:.1f}s | tabular eval {evt.mean():.3f}")

    headers = (["seed"] + [f"pi({s})" for s in NONTERMINAL] +
               ["nominal path", "eval mean", "eval std", "eval min", "eval max", "reach h %",
                "V^pi(a) exact", "gap", "optimal", "loss (last 1000 steps)",
                "predicted floor (final buffer)", "learning steps", "target copies",
                "train seconds"])
    utils.write_csv(out / "seeds_eval.csv", headers, seeds_rows)
    print("\n=== DQN, five seeds ===\n" + utils.text_table(headers, seeds_rows, ".3f"))

    th = ["seed", "pi(a)", "pi(b)", "pi(c)", "eval mean", "eval std", "V^pi(a) exact",
          "train seconds"]
    utils.write_csv(out / "tabular_same_protocol.csv", th, tab_rows)
    print("\n=== Tabular Q-learning (HW2), same evaluation protocol ===\n"
          + utils.text_table(th, tab_rows, ".3f"))

    rows = []
    for s in NONTERMINAL:
        for a in actions(s):
            v = np.array([q[s][a] for q in q_runs])
            t = np.array([q[s][a] for q in tab_q])
            rows.append([s, a, Q_star[s][a], *v, v.mean(), v.std(ddof=1),
                         np.abs(v - Q_star[s][a]).max(), t.mean(), t.std(ddof=1)])
    qh = (["s", "a", "Q*"] + [f"DQN seed{k}" for k in SEEDS] +
          ["DQN mean", "DQN std", "DQN max|err|", "tabular mean", "tabular std"])
    utils.write_csv(out / "q_vs_qstar.csv", qh, rows)
    print("\n=== Learned Q vs Q* ===\n" + utils.text_table(qh, rows, ".3f"))

    utils.write_csv(out / "episode_costs.csv", [f"seed{k}" for k in SEEDS],
                    np.column_stack(costs_all).tolist())
    utils.write_csv(out / "losses.csv", [f"seed{k}" for k in SEEDS],
                    np.column_stack(loss_all).tolist())
    utils.write_text(out / "noise_floor.txt",
                     f"Var(y|s,a) at Q*: " + ", ".join(f"{k}: {v:.4f}" for k, v in var_y.items())
                     + f"\nMSE floor along pi* (per sample) = {floor_pi:.4f}\n")


def sweep(out, Q_star, V_star):
    huber1 = fixed_point(delta=1.0)
    configs = [
        ('baseline', {}),
        ('Huber delta=1', dict(loss='huber', huber_delta=1.0)),
        ('Huber delta=2', dict(loss='huber', huber_delta=2.0)),
        ('M = 1', dict(target_every=1)),
        ('M = 1000', dict(target_every=1000)),
        ('lr = 1e-2', dict(lr=1e-2)),
        ('lr = 1e-4', dict(lr=1e-4)),
        ('B = 64', dict(batch=64)),
    ]
    rows, detail, huber_q = [], [], {}
    for name, change in configs:
        cfg = dict(BASE_CFG, **change)
        n_opt, means, errs, lfin, qs = 0, [], [], [], []
        for seed in SEEDS:
            net, costs, losses, _ = train(seed, cfg)
            pol = {s: greedy(net, s, np.random.default_rng(20_000 + seed)) for s in BRANCHES}
            gap = exact_value(deterministic(pol))[START] - V_star[START]
            ev, _ = evaluate(lambda s, r: greedy(net, s, r), seed)
            q = q_values(net)
            err = max(abs(q[s][a] - Q_star[s][a]) for s, a in DECISIONS)
            n_opt += abs(gap) < GAP_TOL
            means.append(ev.mean())
            errs.append(err)
            lfin.append(losses[-1000:].mean())
            qs.append(q)
            detail.append([name, seed, pol['a'], pol['b'], pol['c'], gap, ev.mean(), err,
                           losses[-1000:].mean()])
        rows.append([name, f"{n_opt}/5", np.mean(means), np.mean(errs), np.mean(lfin)])
        if name.startswith('Huber') or name == 'baseline':
            huber_q[name] = {k: np.mean([q[k[0]][k[1]] for q in qs]) for k in DECISIONS}
        print(f"{name}: {n_opt}/5 optimal, eval {np.mean(means):.3f}, max|Q-Q*| {np.mean(errs):.3f}")

    h = ["config", "optimal seeds", "eval mean cost", "mean max|Q-Q*| (a,b,c)", "final loss"]
    utils.write_csv(out / "sweep.csv", h, rows)
    utils.write_csv(out / "sweep_detail.csv",
                    ["config", "seed", "pi(a)", "pi(b)", "pi(c)", "gap", "eval mean",
                     "max|Q-Q*|", "final loss"], detail)
    print("\n=== Sensitivity ===\n" + utils.text_table(h, rows, ".3f"))

    hr = [[s, a, Q_star[s][a], huber_q['baseline'][s, a], huber1[s][a],
           huber_q['Huber delta=1'][s, a], huber_q['Huber delta=2'][s, a]] for s, a in DECISIONS]
    hh = ["s", "a", "Q* (MSE fixed point)", "learned MSE", "Huber d=1 fixed point",
          "learned Huber d=1", "learned Huber d=2"]
    utils.write_csv(out / "huber_check.csv", hh, hr)
    print("\n=== Huber bias check (mean over seeds) ===\n" + utils.text_table(hh, hr, ".3f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-sweep', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)

    print(utils.env_info())
    _, V_star, _ = policy_iteration(PI0_STOCHASTIC)          # HW2 answer, for comparison
    Q_star = {s: {a: lookahead(V_star, s, a) for a in actions(s)} for s in NONTERMINAL}
    assert all(abs(fixed_point()[s][a] - Q_star[s][a]) < 1e-12 for s in Q_star for a in Q_star[s])
    var_y = target_variance(Q_star)

    checks(out)
    main_runs(out, Q_star, V_star, var_y)
    if not args.no_sweep:
        sweep(out, Q_star, V_star)
    utils.write_text(out / "run_info.txt", utils.env_info() + f"\nBase config: {BASE_CFG}\n")


if __name__ == '__main__':
    main()
