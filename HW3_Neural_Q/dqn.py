"""
Neural Q-learning (DQN) on the same shortest-path MDP as HW2.

The environment is imported unchanged from HW2. The learner only sees what
env.step() returns: (s', cost, done). Nothing in this file reads P(s'|s,a),
the policy-iteration values or the tabular Q*.

Conventions (same as HW2): cost = weight of the edge actually traversed,
minimize, gamma = 1, no bootstrap at h.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# append, not insert: HW3's own utils.py must win over HW2's
sys.path.append(str(Path(__file__).resolve().parent.parent / 'HW2_PI_QLearning'))
from mdp import STATES, TERMINAL, GAMMA, actions      # noqa: E402  same MDP as HW2
from q_learning import PathEnv                        # noqa: E402  same environment

ACTIONS = ['A_b', 'A_c', 'A_d', 'A_e', 'A_f', 'go']  # one output slot per action
S_IDX = {s: i for i, s in enumerate(STATES)}
A_IDX = {a: k for k, a in enumerate(ACTIONS)}
NONTERMINAL = [s for s in STATES if s != TERMINAL]

# VALID[s, k]: is output k a feasible action in state s? Row h is all False.
VALID = torch.zeros(len(STATES), len(ACTIONS), dtype=torch.bool)
for _s in NONTERMINAL:
    for _a in actions(_s):
        VALID[S_IDX[_s], A_IDX[_a]] = True
ONE_HOT = torch.eye(len(STATES))

BASE_CFG = dict(
    hidden=32,               # two hidden layers of this width, ReLU
    lr=1e-3,                 # Adam
    batch=32,
    capacity=10_000,         # replay buffer (= 2500 episodes, every episode has 4 steps)
    warmup=200,              # transitions in the buffer before the first update
    target_every=100,        # hard copy theta- <- theta every M learning steps
    loss='mse',              # 'mse' or 'huber'
    huber_delta=1.0,
    episodes=5000,
    eps_start=1.0, eps_end=0.05, eps_decay_end=4000,   # linear over episodes 1..4000
)


class QNet(nn.Module):
    """one-hot(s) [8] -> 32 ReLU -> 32 ReLU -> Q(s, .) [6]. Smaller = better."""

    def __init__(self, hidden=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(len(STATES), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, s_idx):
        return self.f(ONE_HOT[s_idx])


def masked(q, s_idx):
    # cost minimization: an infeasible output has to look infinitely expensive.
    # (DQN code that maximizes reward fills -inf here; that would be wrong for us.)
    return q.masked_fill(~VALID[s_idx], float('inf'))


def greedy(net, s, rng):
    """argmin over feasible actions; exact ties broken uniformly at random."""
    with torch.no_grad():
        q = masked(net(torch.tensor([S_IDX[s]])), torch.tensor([S_IDX[s]]))[0]
    best = q.min()
    ties = torch.nonzero(q == best).flatten().tolist()
    return ACTIONS[ties[rng.integers(len(ties))]]


def epsilon(ep, cfg):
    # ep = 1, 2, ...; eps(1) = 1.0, eps(4000) = 0.05, flat after that
    frac = min((ep - 1) / (cfg['eps_decay_end'] - 1), 1.0)
    return cfg['eps_start'] + frac * (cfg['eps_end'] - cfg['eps_start'])


class Replay:
    """Circular buffer of (s, a, c, s', done), stored as indices."""

    def __init__(self, capacity):
        self.s = np.zeros(capacity, np.int64)
        self.a = np.zeros(capacity, np.int64)
        self.c = np.zeros(capacity, np.float32)
        self.s2 = np.zeros(capacity, np.int64)
        self.done = np.zeros(capacity, bool)
        self.capacity, self.n, self.i = capacity, 0, 0

    def add(self, s, a, c, s2, done):
        k = self.i
        self.s[k], self.a[k], self.c[k], self.s2[k], self.done[k] = s, a, c, s2, done
        self.i = (k + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch, rng):
        idx = rng.choice(self.n, size=batch, replace=False)   # uniform, no repeats
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.c[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.done[idx]))


def td_target(target_net, c, s2, done, mask=True):
    """y = c                              if s' = h
           c + gamma * min_a' Q-(s', a')  otherwise   (feasible a' only)
    mask=False is only for the ablation run: min over all six outputs."""
    with torch.no_grad():
        q_next = target_net(s2)
        q_next = masked(q_next, s2) if mask else q_next
        # h has no feasible action, so its min is +inf. torch.where picks 0 there;
        # (1 - done) * min_q would give 0 * inf = NaN and poison the batch.
        boot = torch.where(done, torch.zeros_like(c), q_next.min(dim=1).values)
    return c + GAMMA * boot


def train(seed, cfg=BASE_CFG, log_first_batch=False):
    """One independent run: fresh nets, optimizer, buffer, eps schedule and RNGs."""
    rng = np.random.default_rng(seed)    # env transitions, exploration, replay, ties
    torch.manual_seed(seed)              # network initialization
    env = PathEnv(rng)

    online = QNet(cfg['hidden'])
    target = QNet(cfg['hidden'])
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=cfg['lr'])
    if cfg['loss'] == 'mse':
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.HuberLoss(delta=cfg['huber_delta'])
    buf = Replay(cfg['capacity'])

    ep_cost, losses, steps, copies, first = [], [], 0, 0, None
    t0 = time.perf_counter()
    for ep in range(1, cfg['episodes'] + 1):
        eps = epsilon(ep, cfg)
        s, done, total = env.reset(), False, 0.0
        while not done:
            if rng.random() < eps:                      # explore among feasible actions
                feas = actions(s)
                a = feas[rng.integers(len(feas))]
            else:
                a = greedy(online, s, rng)
            s2, c, done = env.step(a)
            buf.add(S_IDX[s], A_IDX[a], c, S_IDX[s2], done)
            total += c
            s = s2

            if buf.n < cfg['warmup']:
                continue
            bs, ba, bc, bs2, bdone = buf.sample(cfg['batch'], rng)
            y = td_target(target, bc, bs2, bdone, mask=cfg.get('mask_target', True))
            q = online(bs).gather(1, ba[:, None]).squeeze(1)
            loss = loss_fn(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            losses.append(loss.item())
            if steps % cfg['target_every'] == 0:
                target.load_state_dict(online.state_dict())
                copies += 1
            if log_first_batch and first is None:
                first = dict(s=bs, a=ba, c=bc, s2=bs2, done=bdone, y=y, q=q.detach())
        ep_cost.append(total)

    info = dict(steps=steps, copies=copies, seconds=time.perf_counter() - t0,
                buffer=buf, first_batch=first)
    return online, np.array(ep_cost), np.array(losses), info


def q_values(net):
    """{s: {a: Q(s,a)}} over feasible actions, for reporting."""
    with torch.no_grad():
        q = net(torch.arange(len(STATES)))
    return {s: {a: q[S_IDX[s], A_IDX[a]].item() for a in actions(s)} for s in NONTERMINAL}


def evaluate(act, seed, n=1000, max_steps=20):
    """n episodes with exploration off, using an RNG separate from training.
    act(s, rng) -> action, e.g. lambda s, r: greedy(net, s, r)."""
    rng = np.random.default_rng(10_000 + seed)
    env = PathEnv(rng)
    costs, reached = np.empty(n), 0
    for i in range(n):
        s, done, total, t = env.reset(), False, 0.0, 0
        while not done and t < max_steps:
            s, c, done = env.step(act(s, rng))
            total += c
            t += 1
        costs[i], reached = total, reached + done
    return costs, reached / n
