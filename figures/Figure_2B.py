#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_2B.py

Generate Figure 2B from a representative trajectory sample.

Input:
    data/trajectories/PM_trajectory_N300_kper2p5_kop3.npz

Output:
    figures/output/Figure_2B.pdf
    figures/output/Figure_2B.png
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


try:
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None


EPS = 1e-12
DEFAULT_CMAP = "viridis"

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "trajectories"
    / "PM_trajectory_N300_kper2p5_kop3.npz"
)

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.dpi": 600,
        }
    )


def despine(ax):
    if sns is not None:
        sns.despine(ax=ax)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def load_positions_npz(path: Path, key: str = "pos"):
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")

    data = np.load(path)

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}")

    X = data[key]

    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T, 2, N), got {X.shape}")

    return X


def theta_from_positions(X):
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (T, 2, N). Got {X.shape}")

    dX = X[1:] - X[:-1]
    return np.arctan2(dX[:, 1, :], dX[:, 0, :])


def polarity_timeseries(X):
    theta = theta_from_positions(X)
    cx = np.cos(theta).mean(axis=1)
    sy = np.sin(theta).mean(axis=1)
    return np.sqrt(cx**2 + sy**2)


def milling_timeseries(X):
    theta = theta_from_positions(X)

    pos = X[:-1]
    x = pos[:, 0, :]
    y = pos[:, 1, :]

    cx = x.mean(axis=1, keepdims=True)
    cy = y.mean(axis=1, keepdims=True)

    rx = x - cx
    ry = y - cy

    norm = np.sqrt(rx**2 + ry**2) + EPS
    rx = rx / norm
    ry = ry / norm

    vx = np.cos(theta)
    vy = np.sin(theta)

    cross = rx * vy - ry * vx
    return np.abs(cross.mean(axis=1))


def polarity_milling_timeseries(X):
    return polarity_timeseries(X), milling_timeseries(X)


def plot_pm_trajectory(P, M, out_pdf, out_png, cmap=DEFAULT_CMAP):
    ok = np.isfinite(P) & np.isfinite(M)
    P = P[ok]
    M = M[ok]

    if P.size < 2:
        raise ValueError("Not enough valid PM points to plot.")

    t = np.linspace(0, 1, len(P))

    fig, ax = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)

    points = np.column_stack([P, M]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
    lc.set_array(t[:-1])
    lc.set_linewidth(0.35)
    lc.set_alpha(0.95)
    ax.add_collection(lc)

    sc = ax.scatter(
        P,
        M,
        c=t,
        cmap=cmap,
        s=2.2,
        linewidths=0,
        alpha=0.75,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(which="major", alpha=0.15, linewidth=0.3)
    ax.grid(which="minor", alpha=0.12, linewidth=0.2)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"Polarity $P$")
    ax.set_ylabel(r"Milling $M$")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.055, pad=0.03)
    cbar.set_label("Simulation step")
    cbar.ax.tick_params(labelsize=5)

    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(
        [
            "0",
            r"$2.5 \times 10^4$",
            r"$5 \times 10^4$",
            r"$7.5 \times 10^4$",
            r"$1 \times 10^5$",
        ]
    )

    despine(ax)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    set_paper_style()

    X = load_positions_npz(DATA_FILE, key="pos")
    P, M = polarity_milling_timeseries(X)

    out_pdf = OUT_DIR / "Figure_2B.pdf"
    out_png = OUT_DIR / "Figure_2B.png"

    plot_pm_trajectory(
        P=P,
        M=M,
        out_pdf=out_pdf,
        out_png=out_png,
    )

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()