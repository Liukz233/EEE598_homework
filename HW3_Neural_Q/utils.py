"""
Helpers for logging, tables and plots.

Boilerplate only (formatting, CSV output, plotting, version info); drafted
with AI assistance and reviewed by us. No learning logic lives here.
"""
from __future__ import annotations

import csv
import platform
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend so the script runs anywhere
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# text / files
# --------------------------------------------------------------------------
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
    import torch
    cpu = platform.processor() or platform.machine()
    try:  # Linux: nicer CPU name
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return (f"Python {platform.python_version()}, NumPy {np.__version__}, "
            f"Matplotlib {matplotlib.__version__}, PyTorch {torch.__version__} (CPU, 1 thread)\n"
            f"Hardware: {cpu}, {platform.system()} {platform.machine()}, no GPU used")


def first_batch_report(fb: dict, states: Sequence[str], acts: Sequence[str]) -> str:
    """Distinct transitions in the first sampled mini-batch with their TD targets."""
    seen, rows = set(), []
    for i in range(len(fb["y"])):
        key = (int(fb["s"][i]), int(fb["a"][i]), int(fb["s2"][i]))
        if key in seen:
            continue
        seen.add(key)
        done = bool(fb["done"][i])
        boot = 0.0 if done else float(fb["y"][i] - fb["c"][i])
        rows.append([states[key[0]], acts[key[1]], float(fb["c"][i]), states[key[2]],
                     done, "0 (terminal)" if done else f"{boot:.4f}",
                     float(fb["y"][i]), float(fb["q"][i])])
    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    return text_table(["s", "a", "c", "s'", "done", "boot = min Q-(s',.)", "target y",
                       "Q_theta(s,a)"], rows)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def rolling_mean(x: np.ndarray, window: int = 100) -> np.ndarray:
    """Trailing rolling mean over full windows only (len(x) - window + 1 points)."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def plot_seed_curves(costs: np.ndarray, losses: np.ndarray, v_star: float, loss_floor: float,
                     path: Path, eps_end_episode: int, copy_every: int,
                     cost_window: int = 100, loss_window: int = 500) -> None:
    """Two stacked panels for one IEEE column: (top) episode cost + 100-episode rolling
    mean, (bottom) training loss + rolling mean on a log axis. Saved as PNG and PDF."""
    ink, muted, grid = "#1f1f1f", "#6b6b6b", "#e3e3e3"
    raw_c, mean_c = "#9aa3ad", "#2b5c9e"
    plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral",  # close to IEEE Times
                         "mathtext.fontset": "stix",
                         "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.9))

    # --- episode cost
    ep = np.arange(1, len(costs) + 1)
    jitter = np.random.default_rng(12345).uniform(-0.08, 0.08, len(costs))
    ax1.scatter(ep, costs + jitter, s=1.5, color=raw_c, alpha=0.35, lw=0,
                label="Episode total cost", rasterized=True)
    ax1.plot(ep[cost_window - 1:], rolling_mean(costs, cost_window), color=mean_c, lw=1.3,
             label=f"{cost_window}-episode rolling mean")
    ax1.axhline(v_star, color=ink, lw=0.9, ls="--", label=f"$V^*(a)$ = {v_star:.2f}")
    ax1.axvline(eps_end_episode, color=muted, lw=0.7, ls=":")
    ax1.text(eps_end_episode - 60, 11.25, r"$\varepsilon$ = 0.05 from here", color=muted,
             fontsize=6.5, va="center", ha="right")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total cost per episode")
    ax1.set_xlim(0, len(costs))
    ax1.set_ylim(7.6, 11.5)
    ax1.set_yticks([8, 9, 10, 11])
    leg = ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=6.3,
                     frameon=True, framealpha=0.9, edgecolor=grid, markerscale=3)
    leg.get_frame().set_linewidth(0.6)

    # --- training loss
    step = np.arange(1, len(losses) + 1)
    ax2.plot(step, losses, color=raw_c, lw=0.3, alpha=0.5, label="Mini-batch loss",
             rasterized=True)
    ax2.plot(step[loss_window - 1:], rolling_mean(losses, loss_window), color=mean_c, lw=1.3,
             label=f"{loss_window}-step rolling mean")
    ax2.axhline(loss_floor, color=ink, lw=0.9, ls="--",
                label=f"Noise floor under $\\pi^*$ = {loss_floor:.2f}")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"Learning step (target copied every {copy_every} steps)")
    ax2.set_ylabel("Training loss (MSE)")
    ax2.set_xlim(0, len(losses))
    leg = ax2.legend(loc="upper right", fontsize=6.3, frameon=True, framealpha=0.9,
                     edgecolor=grid)
    leg.get_frame().set_linewidth(0.6)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
