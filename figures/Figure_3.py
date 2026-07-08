#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_3.py

Generate Figure 3 from precomputed flip-interval survival data.

Input:
    data/processed/figure3/Figure_flip_interval_survival_data.csv

Output:
    figures/output/Figure_3.pdf
    figures/output/Figure_3.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


try:
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "figure3"
    / "Figure_flip_interval_survival_data.csv"
)

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CONDITIONS = [
    ("I   : Milling phase", 2.5, 3.0),
    ("III : MS phase", 1.5, 2.0),
    ("IV : MS phase", 2.5, 1.8),
    ("V  : SMS phase", 0.5, 1.5),
]


ROMAN_COLORS = {
    "I   : Milling phase": "#2E8B57",
    "III : MS phase": "#8C2D2D",
    "IV : MS phase": "#D95F5F",
    "V  : SMS phase": "#2B2B2B",
}


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
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


def survival_xy(durations):
    d = np.asarray(durations, float)
    d = d[np.isfinite(d) & (d > 0)]

    if len(d) == 0:
        return np.array([]), np.array([])

    vals = np.sort(np.unique(d))
    surv = np.array([(d >= v).mean() for v in vals], dtype=float)

    return vals, surv


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = {
        "condition",
        "kappa_per",
        "kappa_op",
        "flip_interval",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    return df


def plot_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
):
    if df.empty:
        raise RuntimeError("No flip interval data to plot.")

    fig, ax = plt.subplots(figsize=(3.1, 2.45), constrained_layout=True)

    for condition_name, kper, kop in CONDITIONS:
        sub = df[
            (np.isclose(df["kappa_per"], kper))
            & (np.isclose(df["kappa_op"], kop))
        ]

        if sub.empty:
            continue

        x, y = survival_xy(sub["flip_interval"].to_numpy())

        if len(x) == 0:
            continue

        ax.plot(
            x,
            y,
            color=ROMAN_COLORS[condition_name],
            linewidth=0.8,
            alpha=0.96,
            label=condition_name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Flip interval")
    ax.set_ylabel(r"$P(T \geq t)$")

    ax.grid(which="major", alpha=0.200, linewidth=0.25)
    ax.grid(which="minor", alpha=0.200, linewidth=0.15)

    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=5,
        handlelength=1.2,
        labelspacing=0.2,
        borderaxespad=0.15,
    )

    despine(ax)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")

    plt.close(fig)


def main():
    set_paper_style()

    df = load_data(DATA_FILE)

    out_pdf = OUT_DIR / "Figure_3.pdf"
    out_png = OUT_DIR / "Figure_3.png"

    plot_figure(
        df=df,
        out_pdf=out_pdf,
        out_png=out_png,
    )

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()