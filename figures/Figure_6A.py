#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_6A.py

Generate Figure 6: representative COM trajectories for three conditions.

Input:
    data/trajectories/PM_trajectory_N300_kper0p5_kop1p5.npz
    data/trajectories/PM_trajectory_N300_kper1p5_kop3.npz
    data/trajectories/PM_trajectory_N300_kper2p5_kop3.npz

Output:
    figures/output/Figure_6A.pdf
    figures/output/Figure_6A.png
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


try:
    import seaborn as sns
    sns.set_theme(style="white", context="paper")
except Exception:
    sns = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "trajectories"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CONDITIONS = [
    (
        "PM_trajectory_N300_kper0p5_kop1p5.npz",
        r"$\kappa_{\rm per}=0.5,\ \kappa_{\rm op}=1.5$",
        "#34495E",
    ),
    (
        "PM_trajectory_N300_kper1p5_kop3.npz",
        r"$\kappa_{\rm per}=1.5,\ \kappa_{\rm op}=2.0$",
        "#C06C3E",
    ),
    (
        "PM_trajectory_N300_kper2p5_kop3.npz",
        r"$\kappa_{\rm per}=2.5,\ \kappa_{\rm op}=3.0$",
        "#5B8A72",
    ),
]


def set_paper_style():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_positions_npz(path: Path, key: str = "pos") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")

    data = np.load(path, allow_pickle=False)

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}")

    X = data[key]

    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T, 2, N), got {X.shape}")

    return X


def center_of_mass_trajectory(X: np.ndarray):
    cx = X[:, 0, :].mean(axis=1)
    cy = X[:, 1, :].mean(axis=1)
    return cx, cy


def plot_one_com_trajectory(ax, cx, cy, color, title):
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)

    points = np.column_stack([cx, cy])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        colors=color,
        linewidths=0.70,
        alpha=0.88,
    )
    ax.add_collection(lc)

    ax.scatter(cx[0], cy[0], s=12, color="0.12", zorder=3)
    ax.scatter(
        cx[-1],
        cy[-1],
        s=12,
        facecolor="white",
        edgecolor="0.12",
        linewidth=0.7,
        zorder=3,
    )

    pad_x = 0.05 * (np.nanmax(cx) - np.nanmin(cx) + 1e-12)
    pad_y = 0.05 * (np.nanmax(cy) - np.nanmin(cy) + 1e-12)

    ax.set_xlim(np.nanmin(cx) - pad_x, np.nanmax(cx) + pad_x)
    ax.set_ylim(np.nanmin(cy) - pad_y, np.nanmax(cy) + pad_y)

    ax.set_aspect("equal", adjustable="datalim")

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    ax.grid(which="major", color="0.90", linewidth=0.30)
    ax.tick_params(which="major", length=3.0, width=0.65, pad=2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("0.15")


def main():
    set_paper_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, 2.25),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    for i, (ax, (filename, label, color)) in enumerate(zip(axes, CONDITIONS)):
        X = load_positions_npz(DATA_DIR / filename, key="pos")
        cx, cy = center_of_mass_trajectory(X)

        title = f"({chr(97 + i)}) {label}"
        plot_one_com_trajectory(ax, cx, cy, color, title)

    out_pdf = OUT_DIR / "Figure_6A.pdf"
    out_png = OUT_DIR / "Figure_6A.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()