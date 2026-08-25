#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_S4.py

Generate Figure S4 from precomputed criticality summary data.

Input:
    data/processed/figureS4/*_criticality_summary.csv

Output:
    figures/output/Figur_S4.pdf
    figures/output/Figure_S4.png
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
DATA_DIR = ROOT / "data" / "processed" / "figureS4"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CONDITIONS = [
    (
        "kappa_0p5__okappa_1p5_criticality_summary.csv",
        r"$\kappa_{\mathrm{per}}=0.5,\ \kappa_{\mathrm{op}}=1.5$",
        "#34495E",
    ),
    (
        "kappa_1p5__okappa_2_criticality_summary.csv",
        r"$\kappa_{\mathrm{per}}=1.5,\ \kappa_{\mathrm{op}}=2.0$",
        "#D95F5F",
    ),
    (
        "kappa_2p5__okappa_3_criticality_summary.csv",
        r"$\kappa_{\mathrm{per}}=2.5,\ \kappa_{\mathrm{op}}=3.0$",
        "#2E8B57",
    ),
    (
        "kappa_1p8__okappa_2p5_criticality_summary.csv",
        r"$\kappa_{\mathrm{per}}=1.8,\ \kappa_{\mathrm{op}}=2.5$",
        "#7A5195",
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


def summarize_alpha_hi(
    csv_path: Path,
    min_hi_fit: int = 4,
    min_flip_pos: int = 4,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "status",
        "n_hi_fit",
        "n_flip_pos",
        "N",
        "alpha_hi",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    ok = df[
        (df["status"] == "ok")
        & (pd.to_numeric(df["n_hi_fit"], errors="coerce") >= min_hi_fit)
        & (pd.to_numeric(df["n_flip_pos"], errors="coerce") >= min_flip_pos)
    ].copy()

    ok["N"] = pd.to_numeric(ok["N"], errors="coerce")
    ok["alpha_hi"] = pd.to_numeric(ok["alpha_hi"], errors="coerce")

    ok = ok[np.isfinite(ok["N"]) & np.isfinite(ok["alpha_hi"])]

    if ok.empty:
        return pd.DataFrame(columns=["N", "mean", "std", "count"])

    return (
        ok.groupby("N")["alpha_hi"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("N")
    )


def plot_figure(out_pdf: Path, out_png: Path):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(6.8, 2.25),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    for i, (ax, (filename, title, color)) in enumerate(zip(axes, CONDITIONS)):
        csv_path = DATA_DIR / filename
        tmp = summarize_alpha_hi(csv_path)

        if tmp.empty:
            raise RuntimeError(f"No valid alpha_hi data after filtering: {csv_path}")

        ax.errorbar(
            tmp["N"],
            tmp["mean"],
            yerr=tmp["std"],
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=1.0,
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

    axes[0].set_ylabel(r"$\alpha_{\mathrm{hi}}$")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")

    plt.close(fig)


def main():
    set_paper_style()

    out_pdf = OUT_DIR / "Figure_S4.pdf"
    out_png = OUT_DIR / "Figure_S4.png"

    plot_figure(out_pdf, out_png)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()