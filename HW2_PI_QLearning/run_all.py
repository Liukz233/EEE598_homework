"""
Reproduces every number and figure in the report.

    python run_all.py               # PI + Q-learning (seeds 0-4) + sweep
    python run_all.py --no-sweep    # skip the hyperparameter sweep (~1 min)

Outputs go to results/.
"""
import argparse
from pathlib import Path

import numpy as np

from mdp import BRANCHES, START, actions
from policy_iteration import (PI0_PATH, PI0_STOCHASTIC, deterministic, evaluate,
                              lookahead, policy_iteration)
from q_learning import greedy_policy, rollout_greedy, train
import utils

SEEDS = [0, 1, 2, 3, 4]
GAP_TOL = 1e-6

# Q-learning settings used for the main results
CFG = dict(n_episodes=5000, eps_start=1.0, eps_end=0.05, decay_episodes=3000,
           alpha='visit', omega=0.6)


def part1_policy_iteration(out):
    pi_star, V_star, hist = policy_iteration(PI0_STOCHASTIC)
    _, V_star2, hist2 = policy_iteration(PI0_PATH)
    assert all(abs(V_star[s] - V_star2[s]) < 1e-12 for s in V_star)  # same optimum

    # Q* = one-step lookahead on V*, used later to check what Q-learning learned
    Q_star = {s: {a: lookahead(V_star, s, a) for a in actions(s)} for s in V_star if s != 'h'}

    text = (utils.pi_history_report(hist, "Policy iteration from the stochastic pi_0")
            + "\n\n"
            + utils.pi_history_report(hist2, "Supplement: start from path a->c->f->g->h "
                                             "(b keeps the stochastic pi_0)")
            + "\n\n=== Optimal values V* ===\n"
            + utils.text_table(["state"] + utils.STATE_ORDER,
                               [["V*"] + [V_star[s] for s in utils.STATE_ORDER]])
            + f"\n\nOptimal policy: {utils.fmt_policy(pi_star)}\n")
    utils.write_text(out / "policy_iteration.txt", text)
    print(text)
    return pi_star, V_star, Q_star


def part2_q_learning(out, V_star, Q_star):
    rows, q_runs, all_costs = [], [], {}
    for seed in SEEDS:
        Q, costs, trace = train(seed, trace_episodes=5 if seed == 0 else 0, **CFG)
        all_costs[seed] = costs
        q_runs.append(Q)
        if seed == 0:
            utils.write_text(out / "hand_check_seed0.txt",
                             "First five training episodes of seed 0 (Q_0 = 0, "
                             "alpha = 1/N(s,a)^0.6)\n\n" + utils.hand_check_report(trace))

        eval_rng = np.random.default_rng(1000 + seed)     # eps = 0 from here on
        choice = greedy_policy(Q, eval_rng)
        V_pi = evaluate(deterministic(choice))            # exact, uses the model
        gap = V_pi[START] - V_star[START]
        mc = rollout_greedy(Q, 20000, eval_rng)           # sanity check by sampling
        rows.append([seed, choice['a'], choice['b'], choice['c'], V_pi[START],
                     V_star[START], gap, 'yes' if abs(gap) < GAP_TOL else 'no',
                     mc.mean(), mc.std(ddof=1) / np.sqrt(len(mc)), costs[-500:].mean()])

    headers = ["seed", "pi(a)", "pi(b)", "pi(c)", "V^pi(a) exact", "V*(a)", "gap",
               "optimal", "greedy MC mean", "MC s.e.", "train cost (last 500 ep)"]
    utils.write_csv(out / "qlearning_seeds.csv", headers, rows)
    table = utils.text_table(headers, rows)
    print("\n=== Q-learning, greedy policy of each seed ===\n" + table)

    # learned Q vs Q* (mean/std across seeds)
    qrows = []
    for s in Q_star:
        for a in Q_star[s]:
            vals = np.array([Q[s][a] for Q in q_runs])
            qrows.append([s, a, Q_star[s][a], *vals, vals.mean(), vals.std(ddof=1),
                          np.abs(vals - Q_star[s][a]).max()])
    qheaders = ["s", "a", "Q*"] + [f"seed{k}" for k in SEEDS] + ["mean", "std", "max |err|"]
    utils.write_csv(out / "q_table_vs_qstar.csv", qheaders, qrows)
    print("\n=== Learned Q vs Q* ===\n" + utils.text_table(qheaders, qrows, ".3f"))

    utils.write_csv(out / "episode_costs.csv", [f"seed{k}" for k in SEEDS],
                    np.column_stack([all_costs[k] for k in SEEDS]).tolist())
    utils.plot_learning_curve(all_costs[4], V_star[START], out / "seed4_learning_curve",
                              seed=4, eps_end_episode=CFG['decay_episodes'])
    return table


def part3_sweep(out, V_star, Q_star):
    """Sensitivity: how often is the greedy policy optimal, and how close is Q to Q*?"""
    rows = []
    for n_ep in (200, 5000):
        for alpha in ('visit', 0.05, 0.2, 0.8):
            for eps_end in (0.0, 0.05, 0.3):
                n_opt, q_err = 0, []
                for seed in SEEDS:
                    Q, _, _ = train(seed, n_episodes=n_ep, eps_start=1.0, eps_end=eps_end,
                                    decay_episodes=int(0.6 * n_ep), alpha=alpha, omega=0.6)
                    choice = greedy_policy(Q, np.random.default_rng(1000 + seed))
                    gap = evaluate(deterministic(choice))[START] - V_star[START]
                    n_opt += abs(gap) < GAP_TOL
                    q_err.append(max(abs(Q[s][a] - Q_star[s][a]) for s in BRANCHES for a in Q[s]))
                a_name = '1/N^0.6' if alpha == 'visit' else f'{alpha}'
                rows.append([n_ep, a_name, eps_end, f"{n_opt}/5", np.mean(q_err)])
    headers = ["episodes", "alpha", "eps_end", "optimal seeds", "mean max|Q-Q*| (a,b,c)"]
    utils.write_csv(out / "sweep.csv", headers, rows)
    print("\n=== Hyperparameter sweep ===\n" + utils.text_table(headers, rows, ".3f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-sweep', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)

    print(utils.env_info())
    _, V_star, Q_star = part1_policy_iteration(out)
    part2_q_learning(out, V_star, Q_star)
    if not args.no_sweep:
        part3_sweep(out, V_star, Q_star)
    utils.write_text(out / "run_info.txt", utils.env_info() + f"\nQ-learning config: {CFG}\n")


if __name__ == '__main__':
    main()
