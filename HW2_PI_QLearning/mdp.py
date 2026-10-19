"""
Episodic shortest-path MDP (Fig. 1 of the assignment).

Conventions we use everywhere:
  - cost c(s, a, s') = weight of the edge that is ACTUALLY traversed (s -> s'),
    not the intended one. E.g. at c, choosing A_e but slipping to f costs 4.
  - minimize total undiscounted cost, gamma = 1, V(h) = 0.
  - only a, b, c have a choice. d, e, f, g have a single deterministic
    action, which we call 'go'.
"""

STATES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
START, TERMINAL = 'a', 'h'
GAMMA = 1.0
P_INTENDED = 0.8      # intended branch succeeds w.p. 0.8, other branch 0.2

EDGE_COST = {
    ('a', 'b'): 2, ('a', 'c'): 2,
    ('b', 'd'): 3, ('b', 'e'): 2,
    ('c', 'e'): 5, ('c', 'f'): 4,
    ('d', 'g'): 3,
    ('e', 'g'): 3,
    ('f', 'g'): 2,
    ('g', 'h'): 1,
}

# branching states -> the two possible successors
BRANCHES = {'a': ('b', 'c'), 'b': ('d', 'e'), 'c': ('e', 'f')}
# single-successor states
NEXT = {'d': 'g', 'e': 'g', 'f': 'g', 'g': 'h'}

# reverse topological order. Since the graph is a DAG, sweeping V in this
# order gives the exact value in one pass.
BACKWARD_ORDER = ['g', 'f', 'e', 'd', 'c', 'b', 'a']


def actions(s):
    if s in BRANCHES:
        return ['A_' + x for x in BRANCHES[s]]   # A_x = "try to go to x"
    if s in NEXT:
        return ['go']
    return []                                    # terminal


def transitions(s, a):
    """P(s'|s,a) together with the edge cost: list of (prob, s', cost)."""
    if s in BRANCHES:
        target = a[2:]                           # 'A_b' -> 'b'
        u, v = BRANCHES[s]
        other = v if target == u else u
        return [(P_INTENDED, target, EDGE_COST[s, target]),
                (1 - P_INTENDED, other, EDGE_COST[s, other])]
    s2 = NEXT[s]
    return [(1.0, s2, EDGE_COST[s, s2])]
