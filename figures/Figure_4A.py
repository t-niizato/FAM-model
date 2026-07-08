#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_4A.py

Generate Figure 4A from precomputed chi-tau data.

Input:
    data/processed/figure4/Figure_chi_tau_N300_kper1.5_kop2_data.csv

Output:
    figures/output/Figure_4A.pdf
    figures/output/Figure_4A.png
"""

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
    / "figure4"
    / "Figure_chi_tau_N300_kper1.5_kop2_data.csv"
)

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
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


def fit_powerlaw_tau_vs_chi(chi, tau):
    chi = np.asarray(chi, float)
    tau = np.asarray(tau, float)

    m = np.isfinite(chi) & np.isfinite(tau) & (chi > 0) & (tau > 0)

    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())

    x = np.log10(chi[m])
    y = np.log10(tau[m])

    alpha, a = np.polyfit(x, y, 1)
    yhat = a + alpha * x

    denom = ((y - y.mean()) ** 2).sum()
    r2 = np.nan if denom <= 0 else 1.0 - ((y - yhat) ** 2).sum() / denom
    pref = 10 ** a

    return float(alpha), float(pref), float(r2), int(m.sum())


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = {"chi", "tau_int"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    return df


def plot_figure(df: pd.DataFrame, out_pdf: Path, out_png: Path):
    chi = df["chi"].to_numpy(float)
    tau = df["tau_int"].to_numpy(float)

    m = np.isfinite(chi) & np.isfinite(tau) & (chi > 0) & (tau > 0)

    if m.sum() < 3:
        raise RuntimeError("Not enough valid chi-tau points.")

    x = chi[m]
    y = tau[m]

    alpha, pref, r2, _ = fit_powerlaw_tau_vs_chi(chi, tau)

    fig, ax = plt.subplots(figsize=(2.65, 2.25), constrained_layout=True)

    ax.scatter(
        x,
        y,
        s=12,
        c="#4E79A7",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.88,
        zorder=2,
    )

    if np.isfinite(alpha) and np.isfinite(pref):
        xs = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)

        ax.plot(
            xs,
            pref * xs ** alpha,
            color="0.15",
            linewidth=0.8,
            zorder=3,
        )

        ax.text(
            0.04,
            0.96,
            rf"$\alpha={alpha:.2f}$" + "\n" + rf"$R^2={r2:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\chi_\Omega$")
    ax.set_ylabel(r"$\tau_{\mathrm{int},\Omega}$")

    ax.grid(which="major", alpha=0.15, linewidth=0.25)
    ax.grid(which="minor", alpha=0.15, linewidth=0.15)

    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    despine(ax)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")

    plt.close(fig)


def main():
    set_paper_style()

    df = load_data(DATA_FILE)

    out_pdf = OUT_DIR / "Figure_4A.pdf"
    out_png = OUT_DIR / "Figure_4A.png"

    plot_figure(df, out_pdf, out_png)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()