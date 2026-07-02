#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns


BASE = Path(
    "/home/hpc/Desktop/criticality_analysis_by_param_085"
)

CONDITIONS = [
    ("kappa_0p5__okappa_1p5", r"$\kappa_{\mathrm{per}}=0.5,\ \kappa_{\mathrm{op}}=1.5$"),
    ("kappa_1p5__okappa_2",   r"$\kappa_{\mathrm{per}}=1.5,\ \kappa_{\mathrm{op}}=2.0$"),
    ("kappa_2p5__okappa_3",   r"$\kappa_{\mathrm{per}}=2.5,\ \kappa_{\mathrm{op}}=3.0$"),
]


def summarize_alpha_hi(csv_path: Path, min_hi_fit=4, min_flip_pos=4):
    df = pd.read_csv(csv_path)

    ok = df[
        (df["status"] == "ok") &
        (pd.to_numeric(df["n_hi_fit"], errors="coerce") >= min_hi_fit) &
        (pd.to_numeric(df["n_flip_pos"], errors="coerce") >= min_flip_pos)
    ].copy()

    ok["N"] = pd.to_numeric(ok["N"], errors="coerce")
    ok["alpha_hi"] = pd.to_numeric(ok["alpha_hi"], errors="coerce")
    ok = ok[np.isfinite(ok["N"]) & np.isfinite(ok["alpha_hi"])]

    return (
        ok.groupby("N")["alpha_hi"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("N")
    )


def main():
    sns.set_theme(style="white", context="paper")

    plt.rcParams.update({
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
    })

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.8, 2.25),
        sharex=True,
        sharey=False,
        constrained_layout=True
    )

    colors = [
        "#34495E",  # charcoal blue
        "#D95F5F",  # muted terracotta
        "#2E8B57",  # sage green
    ]

    for i, (ax, (folder, title)) in enumerate(zip(axes, CONDITIONS)):
        csv_path = BASE / folder / "criticality_summary.csv"
        tmp = summarize_alpha_hi(csv_path)

        ax.errorbar(
            tmp["N"], tmp["mean"], yerr=tmp["std"],
            fmt="o-",
            color=colors[i],
            ecolor=colors[i],
            linewidth=1.0,
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            markersize=3.2,
            markeredgewidth=0.0,
        )

        ax.set_title(f"({chr(97+i)}) {title}", fontsize=8, pad=3)
        ax.set_xlabel(r"$N$")

        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(which="major", color="0.88", linewidth=0.35)
        ax.grid(which="minor", color="0.94", linewidth=0.22)

        ax.tick_params(which="major", length=3.0, width=0.65, pad=2)
        ax.tick_params(which="minor", length=1.7, width=0.45)

        sns.despine(ax=ax)

    axes[0].set_ylabel(r"$\alpha_{\mathrm{hi}}$")

    out_pdf = BASE / "alpha_hi_vs_N_three_conditions.pdf"
    out_png = BASE / "alpha_hi_vs_N_three_conditions.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()