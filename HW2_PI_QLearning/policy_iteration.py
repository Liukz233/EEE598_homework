"""
Policy iteration (model-based): uses P(s'|s,a) and c(s,a,s') directly.

A policy is stored as pi[s] = {action: prob}, so the stochastic initial
policy and the deterministic improved policies go through the same code.
"""
from mdp import TERMINAL, BACKWARD_ORDER, BRANCHES, actions, transitions

TIE_TOL = 1e-9


def lookahead(V, s, a):
    # Q(s,a) = sum_s' P(s'|s,a) * [c(s,a,s') + V(s')]      (gamma = 1)
    return sum(p * (c + V[s2]) for p, s2, c in transitions(s, a))


def evaluate(pi):
    """Exact V^pi. DAG + backward order -> one sweep, no iteration needed."""
    V = {TERMINAL: 0.0}
    for s in BACKWARD_ORDER:
        V[s] = sum(prob * lookahead(V, s, a) for a, prob in pi[s].items())
    return V


def improve(V, pi_old):
    """Greedy policy w.r.t. V. Also returns the lookahead table for the report.

    Tie rule: if the old (most likely) action is among the argmins, keep it.
    Otherwise take the first argmin. This stops the policy from flipping
    between equally good actions, so the stability test can terminate.
    """
    pi_new, Q = {}, {}
    for s in BACKWARD_ORDER:
        Q[s] = {a: lookahead(V, s, a) for a in actions(s)}
        best = min(Q[s].values())
        ties = [a for a, q in Q[s].items() if q - best < TIE_TOL]
        old = max(pi_old[s], key=pi_old[s].get)
        a_star = old if old in ties else ties[0]
        pi_new[s] = {a_star: 1.0}
    return pi_new, Q


def policy_iteration(pi0, max_iter=50):
    """Returns the final policy, its V, and a history of
    (pi_k, V^pi_k, lookahead table, pi_{k+1}) for every round."""
    pi, history = pi0, []
    for _ in range(max_iter):
        V = evaluate(pi)
        pi_next, Q = improve(V, pi)
        history.append((pi, V, Q, pi_next))
        if pi_next == pi:            # stable -> optimal
            return pi, V, history
        pi = pi_next
    raise RuntimeError('policy iteration did not converge')


def deterministic(choice):
    """{'a': 'A_b', ...} -> full policy dict. Non-branching states get 'go'."""
    pi = {s: {a: 1.0} for s, a in choice.items()}
    for s in BACKWARD_ORDER:
        if s not in BRANCHES:
            pi[s] = {'go': 1.0}
    return pi


# initial policies from the assignment ---------------------------------------

PI0_STOCHASTIC = deterministic({})          # fills d, e, f, g with 'go'
PI0_STOCHASTIC.update({
    'a': {'A_b': 0.6, 'A_c': 0.4},
    'b': {'A_d': 0.7, 'A_e': 0.3},
    'c': {'A_e': 0.5, 'A_f': 0.5},
})

# "a -> c -> f -> g -> h" from page 1. It does not say what to do at b, but b
# is still reachable (0.2 slip at a), so we keep the stochastic pi0 there.
PI0_PATH = dict(PI0_STOCHASTIC)
PI0_PATH['a'] = {'A_c': 1.0}
PI0_PATH['c'] = {'A_f': 1.0}
