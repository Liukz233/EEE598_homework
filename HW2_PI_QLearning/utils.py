"""
Helpers for logging, tables and plots.

Boilerplate only (formatting, CSV output, plotting, version info); drafted
with AI assistance and reviewed by us. No algorithm logic lives here.
"""
from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend so the script runs anywhere
import matplotlib.pyplot as plt  # noqa: E402

STATE_ORDER = ["a", "b", "c", "d", "e", "f", "g", "h"]


# --------------------------------------------------------------------------
# text formatting
# --------------------------------------------------------------------------
def fmt_dist(dist: dict[str, float]) -> str:
    """{'A_b': 0.6, 'A_c': 0.4} -> 'A_b:0.60 / A_c:0.40'; deterministic -> 'A_b'."""
    if len(dist) == 1:
        return next(iter(dist))
    return " / ".join(f"{a}:{p:.2f}" for a, p in dist.items())


def fmt_policy(pi: dict[str, dict[str, float]], states: Iterable[str] = "abc") -> str:
    return ", ".join(f"{s}: {fmt_dist(pi[s])}" for s in states)


def text_table(headers: Sequence[str], rows: Sequence[Sequence], floatfmt: str = ".4f") -> str:
    """Plain-text table with aligned columns (for .txt logs and the console)."""
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return format(x, floatfmt)
        return str(x)

    body = [[cell(x) for x in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    out = [line, sep] + ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body]
    return "\n".join(out)


def pi_history_report(history, title: str) -> str:
    """Readable log of every policy-iteration round: policy, V^pi, lookahead Q, new policy."""
    lines = [f"=== {title} ===", ""]
    for k, (pi, V, Q, pi_next) in enumerate(history):
        lines.append(f"--- Round {k}: evaluate pi_{k} ---")
        lines.append(f"pi_{k}: {fmt_policy(pi)}")
        lines.append(text_table(["state"] + STATE_ORDER,
                                [[f"V^pi_{k}"] + [V[s] for s in STATE_ORDER]]))
        lines.append("")
        lines.append(f"Improvement: one-step lookahead Q(s,a) using V^pi_{k}")
        rows = []
        for s in "abc":
            q = Q[s]
            best = min(q.values())
            for a, v in q.items():
                rows.append([s, a, v, "<- argmin" if abs(v - best) < 1e-9 else ""])
        lines.append(text_table(["s", "a", "Q(s,a)", ""], rows))
        stable = pi_next == pi
        lines.append(f"pi_{k + 1}: {fmt_policy(pi_next)}"
                     f"   [{'STABLE: pi_{k+1} == pi_{k}, stop' if stable else 'changed, continue'}]")
        lines.append("")
    return "\n".join(lines)


def hand_check_report(trace: list[dict]) -> str:
    headers = ["ep", "s", "a", "c", "s'", "alpha", "Q_old", "min Q(s',.)",
               "target", "TD error", "Q_new"]
    rows = [[t["ep"], t["s"], t["a"], t["c"], t["s2"], t["alpha"], t["q_old"],
             "0 (terminal)" if t["s2"] == "h" else f"{t['boot']:.4f}",
             t["target"], t["td"], t["q_new"]] for t in trace]
    return text_table(headers, rows)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, (float, np.floating)) else x for x in r])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def env_info() -> str:
    return (f"Python {platform.python_version()}, NumPy {np.__version__}, "
            f"Matplotlib {matplotlib.__version__}")


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def rolling_mean(x: np.ndarray, window: int = 100) -> np.ndarray:
    """Trailing rolling mean over full windows only: element i averages episodes
    i-window+1 ... i, so the output has len(x) - window + 1 points."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def plot_learning_curve(costs: np.ndarray, v_star: float, path: Path, seed: int,
                        window: int = 100, eps_end_episode: int | None = None) -> None:
    """Episode total cost (dots) + rolling mean (line) + optimal expected cost (dashed).

    Sized for one IEEE column (3.5 in). Saved as both PNG and PDF.
    """
    ink, muted, grid = "#1f1f1f", "#6b6b6b", "#e3e3e3"
    raw_c, mean_c = "#9aa3ad", "#2b5c9e"

    plt.rcParams.update({"font.size": 8, "font.family": "serif",
                         "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ep = np.arange(1, len(costs) + 1)

    # small jitter only so overlapping dots on the 3 discrete cost levels stay visible
    jitter = np.random.default_rng(12345).uniform(-0.08, 0.08, len(costs))
    ax.scatter(ep, costs + jitter, s=1.5, color=raw_c, alpha=0.35, lw=0,
               label="Episode total cost", rasterized=True)
    ax.plot(ep[window - 1:], rolling_mean(costs, window), color=mean_c, lw=1.4,
            label=f"{window}-episode rolling mean")
    ax.axhline(v_star, color=ink, lw=0.9, ls="--",
               label=f"Optimal expected cost $V^*(a)$ = {v_star:.2f}")
    if eps_end_episode is not None:
        ax.axvline(eps_end_episode, color=muted, lw=0.7, ls=":")
        ax.text(eps_end_episode + 60, 11.25, r"$\varepsilon$ reaches floor",
                color=muted, fontsize=7, va="center")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total cost per episode")
    ax.set_xlim(0, len(costs))
    ax.set_ylim(7.6, 11.5)
    ax.set_yticks([8, 9, 10, 11])
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # legend sits in the empty band between the cost levels 9 and 11
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=6.5,
                    frameon=True, framealpha=0.9, edgecolor=grid, markerscale=3)
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
