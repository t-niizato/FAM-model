#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_6B.py

Generate Figure 6B: COM flight-length CCDF with fitted curves.

Inputs:
    data/processed/figure7/*_com_all_runs_metrics.csv
    data/trajectories/PM_trajectory_N300_kper0p5_kop1p5.npz
    data/trajectories/PM_trajectory_N300_kper1p5_kop3.npz
    data/trajectories/PM_trajectory_N300_kper2p5_kop3.npz

Output:
    figures/output/Figure_6B.pdf
    figures/output/Figure_6B.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


try:
    import seaborn as sns
    sns.set_theme(style="white", context="paper")
except Exception:
    sns = None


ROOT = Path(__file__).resolve().parents[1]

TRAJ_DIR = ROOT / "data" / "trajectories"
METRIC_DIR = ROOT / "data" / "processed" / "figure7"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CONDITIONS = [
    (
        "PM_trajectory_N300_kper0p5_kop1p5.npz",
        "kappa_0p5__okappa_1p5_com_all_runs_metrics.csv",
        r"$\kappa_{\rm per}=0.5,\ \kappa_{\rm op}=1.5$",
        "#34495E",
    ),
    (
        "PM_trajectory_N300_kper1p5_kop2.npz",
        "kappa_1p5__okappa_2_com_all_runs_metrics.csv",
        r"$\kappa_{\rm per}=1.5,\ \kappa_{\rm op}=2.0$",
        "#C06C3E",
    ),
    (
        "PM_trajectory_N300_kper2p5_kop3.npz",
        "kappa_2p5__okappa_3_com_all_runs_metrics.csv",
        r"$\kappa_{\rm per}=2.5,\ \kappa_{\rm op}=3.0$",
        "#5B8A72",
    ),
]


EPS = 1e-12


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


def finite_positive(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x) & (x > 0)]


def empirical_ccdf(x):
    x = np.sort(finite_positive(x))
    if x.size == 0:
        return None, None

    y = 1.0 - np.arange(1, x.size + 1) / x.size
    y = np.maximum(y, 1.0 / x.size)

    return x, y


def load_positions_npz(path: Path, key: str = "pos"):
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")

    data = np.load(path, allow_pickle=False)

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}")

    X = data[key]

    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T, 2, N), got {X.shape}")

    return X


def center_of_mass_trajectory(X):
    cx = X[:, 0, :].mean(axis=1)
    cy = X[:, 1, :].mean(axis=1)
    return cx, cy


def center_step_lengths(cx, cy):
    return np.hypot(np.diff(cx), np.diff(cy))


def flight_lengths_from_series(series, threshold):
    flights = []
    acc = 0.0

    for v in series:
        if np.isfinite(v) and v >= threshold:
            acc += float(v)
        else:
            if acc > 0:
                flights.append(acc)
                acc = 0.0

    if acc > 0:
        flights.append(acc)

    return np.asarray(flights, float)


def ccdf_truncated_powerlaw(grid, xmin, xmax, alpha):
    grid = np.asarray(grid, float)

    if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(alpha)):
        return None
    if xmin <= 0 or xmax <= xmin:
        return None

    a1 = 1.0 - alpha
    out = np.zeros_like(grid)

    m1 = grid < xmin
    m2 = (grid >= xmin) & (grid <= xmax)
    m3 = grid > xmax

    out[m1] = 1.0
    out[m3] = 0.0

    if abs(a1) > EPS:
        out[m2] = (xmax ** a1 - grid[m2] ** a1) / (xmax ** a1 - xmin ** a1)
    else:
        out[m2] = np.log(xmax / grid[m2]) / np.log(xmax / xmin)

    return out


def ccdf_shifted_exp(grid, xmin, lam):
    grid = np.asarray(grid, float)

    if not (np.isfinite(xmin) and np.isfinite(lam)):
        return None
    if xmin <= 0 or lam <= 0:
        return None

    out = np.ones_like(grid)
    m = grid >= xmin
    out[m] = np.exp(-lam * (grid[m] - xmin))

    return out


def choose_metric_row(metrics: pd.DataFrame, prefer_N: int = 300, rep: int = 0, seed: int = 30000000):
    df = metrics.copy()

    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    sub = df[
        (df["N"] == prefer_N)
        & (df["rep"] == rep)
        & (df["seed"] == seed)
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No matching metric row found for N={prefer_N}, rep={rep}, seed={seed}"
        )

    return sub.iloc[0]


def get_param(row, prefix, name):
    col = f"{prefix}_{name}"
    if col not in row:
        return np.nan
    return float(row[col])


def plot_one_ccdf(ax, flights, row, color, title):
    x_all, y_all = empirical_ccdf(flights)

    if x_all is None:
        raise RuntimeError("No positive flight lengths to plot.")

    ax.loglog(
        x_all,
        y_all,
        ".",
        ms=2.4,
        color=color,
        alpha=0.50,
        label="Empirical",
    )

    xmin = get_param(row, "truncated", "xmin")
    alpha = get_param(row, "truncated", "alpha")

    if not np.isfinite(xmin):
        xmin = get_param(row, "shifted_exp", "xmin")

    lam = get_param(row, "shifted_exp", "alpha")

    if np.isfinite(xmin) and xmin > 0:
        idx0 = np.searchsorted(x_all, xmin, side="left")
        S0 = float(y_all[idx0] if idx0 < len(y_all) else y_all[-1])

        tail = np.sort(finite_positive(flights))
        tail = tail[tail >= xmin]

        if tail.size > 0:
            xmax_data = float(tail.max())
            xmax_model = xmax_data * 0.98

            if xmax_model > xmin:
                x_model = np.logspace(
                    np.log10(xmin),
                    np.log10(xmax_model),
                    300,
                )

                y_tr = ccdf_truncated_powerlaw(
                    x_model,
                    xmin=xmin,
                    xmax=xmax_data,
                    alpha=alpha,
                )

                if y_tr is not None:
                    y_tr = S0 * y_tr
                    keep = np.isfinite(y_tr) & (y_tr > 0)

                    ax.loglog(
                        x_model[keep],
                        y_tr[keep],
                        "-",
                        color="0.10",
                        lw=1.0,
                        label="Truncated PL",
                    )

                y_ex = ccdf_shifted_exp(
                    x_model,
                    xmin=xmin,
                    lam=lam,
                )

                if y_ex is not None:
                    y_ex = S0 * y_ex
                    keep = np.isfinite(y_ex) & (y_ex > 0)

                    ax.loglog(
                        x_model[keep],
                        y_ex[keep],
                        "--",
                        color="0.40",
                        lw=1.05,
                        label="Shifted exp.",
                    )

            ax.axvline(
                xmin,
                color="0.25",
                ls=":",
                lw=0.85,
            )

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("Flight length")
    ax.grid(which="major", color="0.88", linewidth=0.35)
    ax.grid(which="minor", color="0.94", linewidth=0.22)

    ax.tick_params(which="major", length=3.0, width=0.25, pad=2)
    ax.tick_params(which="minor", length=1.7, width=0.25)

    despine(ax)


def main():
    set_paper_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.8, 2.25),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )

    for i, (ax, (npz_name, metrics_name, label, color)) in enumerate(zip(axes, CONDITIONS)):
        X = load_positions_npz(TRAJ_DIR / npz_name, key="pos")
        cx, cy = center_of_mass_trajectory(X)
        step = center_step_lengths(cx, cy)

        metrics_path = METRIC_DIR / metrics_name
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        metrics = pd.read_csv(metrics_path)
        row = choose_metric_row(metrics, prefer_N=300, rep=0, seed=30000000)

        threshold = float(row["threshold"])
        flights = flight_lengths_from_series(step, threshold)

        plot_one_ccdf(
            ax=ax,
            flights=flights,
            row=row,
            color=color,
            title=f"({chr(97 + i)}) {label}",
        )

    axes[0].set_ylabel(r"$P(X \geq x)$")
    axes[0].set_ylim(1e-3, 1)

    axes[-1].legend(
        frameon=False,
        fontsize=6,
        loc="upper right",
        handlelength=1.5,
    )

    out_pdf = OUT_DIR / "Figure_6B.pdf"
    out_png = OUT_DIR / "Figure_6B.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()