#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_S5.py

Generate Figure S5: COM alpha vs N for three conditions.

Input:
    data/processed/figureS5/*_com_all_runs_metrics.csv

Output:
    figures/output/Figure_S5.pdf
    figures/output/Figure_S5.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


try:
    import seaborn as sns
    sns.set_theme(style="white", context="paper")
except Exception:
    sns = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "figureS5"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CONDITIONS = [
    (
        "kappa_0p5__okappa_1p5_com_all_runs_metrics.csv",
        r"$\kappa_{\rm per}=0.5,\ \kappa_{\rm op}=1.5$",
        "#34495E",
    ),
    (
        "kappa_1p5__okappa_2_com_all_runs_metrics.csv",
        r"$\kappa_{\rm per}=1.5,\ \kappa_{\rm op}=2.0$",
        "#C06C3E",
    ),
    (
        "kappa_2p5__okappa_3_com_all_runs_metrics.csv",
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


def despine(ax):
    if sns is not None:
        sns.despine(ax=ax)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def summarize_com_alpha(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"N", "truncated_alpha"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["truncated_alpha"] = pd.to_numeric(df["truncated_alpha"], errors="coerce")

    df = df[np.isfinite(df["N"]) & np.isfinite(df["truncated_alpha"])]

    if df.empty:
        return pd.DataFrame(columns=["N", "mean", "std", "count"])

    return (
        df.groupby("N")["truncated_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("N")
    )


def main():
    set_paper_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, 2.25),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    for i, (ax, (filename, title, color)) in enumerate(zip(axes, CONDITIONS)):
        tmp = summarize_com_alpha(DATA_DIR / filename)

        if tmp.empty:
            raise RuntimeError(f"No valid data after filtering: {DATA_DIR / filename}")

        ax.errorbar(
            tmp["N"],
            tmp["mean"],
            yerr=tmp["std"],
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=1.05,
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            markersize=3.2,
            markeredgewidth=0.0,
        )

        ax.set_title(f"({chr(97 + i)}) {title}", fontsize=8, pad=3)
        ax.set_xlabel(r"$N$")

        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(which="major", color="0.88", linewidth=0.35)
        ax.grid(which="minor", color="0.94", linewidth=0.22)

        ax.tick_params(which="major", length=3.0, width=0.65, pad=2)
        ax.tick_params(which="minor", length=1.7, width=0.45)

        despine(ax)

    axes[0].set_ylabel(r"COM $\alpha$")

    out_pdf = OUT_DIR / "Figure_S5.pdf"
    out_png = OUT_DIR / "Figure_S5.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()