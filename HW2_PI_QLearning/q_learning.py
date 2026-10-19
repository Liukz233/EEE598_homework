"""
Tabular Q-learning (model-free).

The agent never touches P or the edge costs. It only sees what env.step()
returns: the sampled next state and the cost of that transition.
"""
import numpy as np

from mdp import STATES, START, TERMINAL, BRANCHES, actions, transitions


class PathEnv:
    """The model lives inside the environment, hidden from the agent."""

    def __init__(self, rng):
        self.rng = rng
        self.s = START

    def reset(self):
        self.s = START
        return self.s

    def step(self, a):
        outcomes = transitions(self.s, a)
        i = self.rng.choice(len(outcomes), p=[p for p, _, _ in outcomes])
        _, s2, cost = outcomes[i]
        self.s = s2
        return s2, cost, s2 == TERMINAL


def argmin_random(q, rng):
    # break ties uniformly at random (matters a lot at the start: all Q = 0)
    best = min(q.values())
    ties = [a for a, v in q.items() if v - best < 1e-12]
    return ties[rng.integers(len(ties))]


def eps_greedy(q, eps, rng):
    if rng.random() < eps:
        acts = list(q)
        return acts[rng.integers(len(acts))]
    return argmin_random(q, rng)


def epsilon(ep, eps_start, eps_end, decay_episodes):
    # linear decay, then flat
    frac = min(ep / decay_episodes, 1.0)
    return eps_start + frac * (eps_end - eps_start)


def train(seed, n_episodes=5000, eps_start=1.0, eps_end=0.05,
          decay_episodes=3000, alpha='visit', omega=0.6, trace_episodes=0):
    """One independent run. Q is re-initialized to 0 for every seed.

    alpha='visit' -> alpha = 1 / N(s,a)^omega  (Robbins-Monro for 0.5<omega<=1)
    alpha=<float> -> constant step size
    """
    rng = np.random.default_rng(seed)   # one generator per seed: transitions,
    env = PathEnv(rng)                  # exploration and tie-breaking all use it

    Q = {s: {a: 0.0 for a in actions(s)} for s in STATES if s != TERMINAL}
    N = {s: {a: 0 for a in Q[s]} for s in Q}
    ep_cost, trace = [], []

    for ep in range(n_episodes):
        eps = epsilon(ep, eps_start, eps_end, decay_episodes)
        s, done, total = env.reset(), False, 0.0
        while not done:
            a = eps_greedy(Q[s], eps, rng)
            s2, c, done = env.step(a)

            N[s][a] += 1
            lr = 1.0 / N[s][a] ** omega if alpha == 'visit' else alpha
            boot = 0.0 if done else min(Q[s2].values())   # terminal: no bootstrap
            target = c + boot
            td = target - Q[s][a]
            q_old = Q[s][a]
            Q[s][a] += lr * td

            if ep < trace_episodes:
                trace.append(dict(ep=ep, s=s, a=a, c=c, s2=s2, alpha=lr,
                                  q_old=q_old, boot=boot, target=target,
                                  td=td, q_new=Q[s][a]))
            total += c
            s = s2
        ep_cost.append(total)

    return Q, np.array(ep_cost), trace


def greedy_policy(Q, rng):
    """pi(s) in argmin_a Q(s,a) at the branching states (eps = 0)."""
    return {s: argmin_random(Q[s], rng) for s in BRANCHES}


def rollout_greedy(Q, n, rng):
    """Follow the greedy policy from a for n episodes, return episode costs."""
    env = PathEnv(rng)
    costs = np.empty(n)
    for i in range(n):
        s, done, total = env.reset(), False, 0.0
        while not done:
            s, c, done = env.step(argmin_random(Q[s], rng))
            total += c
        costs[i] = total
    return costs
