#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_pm_trajectory_single.py

Paper-style PM trajectory figure for one representative run.

Default target:
    kappa_per = 2.5
    kappa_op  = 3.0
    N         = 300

Assumed directory structure:
    position_root/
      N300/
        kappa_2p5/
          okappa_3/
            *.npz

Outputs:
    PM_trajectory_N300_kper2.5_kop3.pdf
    PM_trajectory_N300_kper2.5_kop3.png
    selected_npz_single.csv

PM definition:
    P(t) = |mean_i exp(j theta_i)|
    M(t) = |mean_i [ r_hat_i x v_hat_i ]|
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator

try:
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None


EPS = 1e-12
DEFAULT_CMAP = "viridis"


# =========================================================
# Paper style
# =========================================================
def set_paper_style():
    """Apply compact journal-friendly style, following make_figure.py style."""
    plt.rcParams.update({
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
    })


def despine(ax):
    if sns is not None:
        sns.despine(ax=ax)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt_float_for_dir(x: float, nd: int = 6) -> str:
    """Convert 2.5 -> 2p5, 3.0 -> 3, -0.5 -> m0p5."""
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def find_npz(position_root: Path, N: int, kappa_per: float, kappa_op: float):
    """Find candidate npz files for the target N, kappa_per, and kappa_op."""
    n_dir = position_root / f"N{N}"
    kappa_dir = n_dir / f"kappa_{fmt_float_for_dir(kappa_per)}"
    okappa_dir = kappa_dir / f"okappa_{fmt_float_for_dir(kappa_op)}"

    candidates = sorted(okappa_dir.glob("*.npz"))
    if candidates:
        return candidates

    # Fallback: tolerate small naming variations by searching under N directory.
    fallback = []
    if n_dir.exists():
        for p in sorted(n_dir.glob("kappa_*/okappa_*/*.npz")):
            parts = p.parts
            if f"kappa_{fmt_float_for_dir(kappa_per)}" in parts and f"okappa_{fmt_float_for_dir(kappa_op)}" in parts:
                fallback.append(p)
    return fallback


def choose_representative(npz_files):
    """Deterministic representative: first file in lexical order."""
    if not npz_files:
        return None
    return Path(sorted(npz_files)[0]).resolve()


# =========================================================
# Loading and PM definitions
# =========================================================
def load_positions_npz(path, key="pos"):
    data = np.load(path)
    X = data[key]
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T,2,N), got {X.shape}")
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


# =========================================================
# Plotting
# =========================================================
def plot_pm_trajectory(P, M, out_pdf, out_png, N, kappa_per, kappa_op, cmap=DEFAULT_CMAP):
    ok = np.isfinite(P) & np.isfinite(M)
    P = P[ok]
    M = M[ok]
    if P.size < 2:
        raise ValueError("Not enough valid PM points to plot.")

    t = np.linspace(0, 1, len(P))

    fig, ax = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)

    # Continuous trajectory colored by normalized time.
    points = np.column_stack([P, M]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
    lc.set_array(t[:-1])
    lc.set_linewidth(0.35)
    lc.set_alpha(0.95)
    ax.add_collection(lc)

    # Small points help reveal dense/stationary regions.
    sc = ax.scatter(P, M, c=t, cmap=cmap, s=2.2, linewidths=0, alpha=0.75)

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
    cbar.set_ticklabels([
        "0",
        "2.5×10⁴",
        "5×10⁴",
        "7.5×10⁴",
        "1×10⁵"
    ])

    despine(ax)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-root", required=True, help="root of original position npz files")
    parser.add_argument("--output-root", required=True, help="directory where the figure will be saved")
    parser.add_argument("--key", default="pos", help="npz key, default: pos")
    parser.add_argument("--N", type=int, default=300, help="target group size, default: 300")
    parser.add_argument("--kappa-per", type=float, default=2.5, help="target perceptual kappa")
    parser.add_argument("--kappa-op", type=float, default=3.0, help="target option kappa")
    parser.add_argument("--npz", default=None, help="optional explicit npz path; overrides automatic search")
    args = parser.parse_args()

    set_paper_style()

    position_root = Path(args.position_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not position_root.exists():
        raise FileNotFoundError(f"position root not found: {position_root}")
    ensure_dir(output_root)

    if args.npz is not None:
        npz_path = Path(args.npz).expanduser().resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"npz not found: {npz_path}")
        npz_files = [npz_path]
    else:
        npz_files = find_npz(position_root, args.N, args.kappa_per, args.kappa_op)
        npz_path = choose_representative(npz_files)
        if npz_path is None:
            raise RuntimeError(
                "No npz file found for "
                f"N={args.N}, kappa_per={args.kappa_per}, kappa_op={args.kappa_op} "
                f"under {position_root}"
            )

    pd.DataFrame({
        "N": [args.N],
        "kappa_per": [args.kappa_per],
        "kappa_op": [args.kappa_op],
        "npz_path": [str(npz_path)],
        "n_candidates": [len(npz_files)],
    }).to_csv(output_root / "selected_npz_single.csv", index=False)

    X = load_positions_npz(npz_path, key=args.key)
    P, M = polarity_milling_timeseries(X)

    out_base = f"PM_trajectory_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}"
    out_pdf = output_root / f"{out_base}.pdf"
    out_png = output_root / f"{out_base}.png"

    plot_pm_trajectory(
        P=P,
        M=M,
        out_pdf=out_pdf,
        out_png=out_png,
        N=args.N,
        kappa_per=args.kappa_per,
        kappa_op=args.kappa_op,
    )

    print("[done]")
    print(f"source npz: {npz_path}")
    print(f"figure pdf: {out_pdf}")
    print(f"figure png: {out_png}")
    print(f"catalog: {output_root / 'selected_npz_single.csv'}")


if __name__ == "__main__":
    main()
