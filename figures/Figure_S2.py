#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_S2.py

Visualize the group-size dependence of collective states in the
polarity-milling plane.

Fixed parameters:
    kappa_per = 2.5
    kappa_op  = 3.0

Group sizes:
    N = 10, 30, 50, 70

Panels:
    (A) Representative PM trajectory for N = 10
    (B) Representative PM trajectory for N = 30
    (C) Representative PM trajectory for N = 50
    (D) Representative PM trajectory for N = 70
    (E) Mean PM position as a function of group size

Input:
    data/trajectories/PM_trajectory_N10_kper2p5_kop3.npz
    data/trajectories/PM_trajectory_N30_kper2p5_kop3.npz
    data/trajectories/PM_trajectory_N50_kper2p5_kop3.npz
    data/trajectories/PM_trajectory_N70_kper2p5_kop3.npz

Output:
    figures/output/Figure_S2.pdf
    figures/output/Figure_S2.png
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


try:
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None


EPS = 1e-12
TIME_CMAP = "viridis"

ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    10: (
        ROOT
        / "data"
        / "trajectories"
        / "PM_trajectory_N10_kper2p5_kop3.npz"
    ),
    30: (
        ROOT
        / "data"
        / "trajectories"
        / "PM_trajectory_N30_kper2p5_kop3.npz"
    ),
    50: (
        ROOT
        / "data"
        / "trajectories"
        / "PM_trajectory_N50_kper2p5_kop3.npz"
    ),
    70: (
        ROOT
        / "data"
        / "trajectories"
        / "PM_trajectory_N70_kper2p5_kop3.npz"
    ),
}

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


N_VALUES = [10, 30, 50, 70]

PANEL_LABELS = {
    10: "(A)",
    30: "(B)",
    50: "(C)",
    70: "(D)",
}

# Remove the initial transient.
BURN_IN_FRACTION = 0.10

# Downsampling used only for plotting.
# Mean P and M are calculated from the full post-burn-in time series.
PLOT_STEP = 1


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 6,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
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


def normalize_position_shape(X):
    """
    Convert position data to shape (T, 2, N).
    """

    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            f"Position array must be three-dimensional. Got {X.shape}"
        )

    if X.shape[1] == 2:
        return X

    if X.shape[2] == 2:
        return np.transpose(X, (0, 2, 1))

    raise ValueError(
        "Expected shape (T, 2, N) or (T, N, 2), "
        f"but got {X.shape}"
    )


def load_positions_npz(path, preferred_key="pos"):
    """
    Load position data from an NPZ file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing trajectory file: {path}"
        )

    with np.load(path) as data:
        if preferred_key in data:
            return normalize_position_shape(
                data[preferred_key]
            )

        for key in data.files:
            candidate = data[key]

            if candidate.ndim != 3:
                continue

            if candidate.shape[1] == 2 or candidate.shape[2] == 2:
                print(
                    f"Warning: using key '{key}' "
                    f"instead of '{preferred_key}' in {path.name}"
                )

                return normalize_position_shape(
                    candidate
                )

        raise KeyError(
            f"No suitable trajectory array found in {path}. "
            f"Available keys: {list(data.files)}"
        )


def theta_from_positions(X):
    """
    Estimate heading angles from successive positions.
    """

    dX = X[1:] - X[:-1]

    return np.arctan2(
        dX[:, 1, :],
        dX[:, 0, :],
    )


def polarity_timeseries(X):
    theta = theta_from_positions(X)

    mean_cos = np.cos(theta).mean(axis=1)
    mean_sin = np.sin(theta).mean(axis=1)

    return np.sqrt(
        mean_cos**2 + mean_sin**2
    )


def milling_timeseries(X):
    theta = theta_from_positions(X)

    # Positions corresponding to each velocity interval.
    pos = X[:-1]

    x = pos[:, 0, :]
    y = pos[:, 1, :]

    center_x = x.mean(axis=1, keepdims=True)
    center_y = y.mean(axis=1, keepdims=True)

    radial_x = x - center_x
    radial_y = y - center_y

    radial_norm = np.sqrt(
        radial_x**2 + radial_y**2
    ) + EPS

    radial_x /= radial_norm
    radial_y /= radial_norm

    velocity_x = np.cos(theta)
    velocity_y = np.sin(theta)

    cross_product = (
        radial_x * velocity_y
        - radial_y * velocity_x
    )

    return np.abs(
        cross_product.mean(axis=1)
    )


def polarity_milling_timeseries(X):
    P = polarity_timeseries(X)
    M = milling_timeseries(X)

    valid = np.isfinite(P) & np.isfinite(M)

    return P[valid], M[valid]


def remove_burn_in(P, M, fraction):
    if not 0.0 <= fraction < 1.0:
        raise ValueError(
            "BURN_IN_FRACTION must satisfy 0 <= value < 1."
        )

    start = int(len(P) * fraction)

    P = P[start:]
    M = M[start:]

    if len(P) < 2:
        raise ValueError(
            "Not enough points remain after burn-in removal."
        )

    return P, M


def load_group_data(group_size):
    """
    Load one representative trajectory for a given group size.
    """

    path = DATA_FILES[group_size]

    X = load_positions_npz(
        path,
        preferred_key="pos",
    )

    actual_n = X.shape[2]

    if actual_n != group_size:
        print(
            f"Warning: expected N={group_size}, "
            f"but {path.name} contains N={actual_n}."
        )

    P, M = polarity_milling_timeseries(X)

    P, M = remove_burn_in(
        P,
        M,
        BURN_IN_FRACTION,
    )

    result = {
        "path": path,
        "P": P,
        "M": M,
        "mean_P": float(np.mean(P)),
        "mean_M": float(np.mean(M)),
    }

    print(
        f"N={group_size}: "
        f"points={len(P)}, "
        f"mean P={result['mean_P']:.4f}, "
        f"mean M={result['mean_M']:.4f}"
    )

    return result


def configure_pm_axis(ax):
    """
    Apply common formatting to a polarity-milling axis.
    """

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_locator(
        MultipleLocator(0.2)
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.xaxis.set_minor_locator(
        MultipleLocator(0.1)
    )
    ax.yaxis.set_minor_locator(
        MultipleLocator(0.1)
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.grid(
        which="major",
        alpha=0.15,
        linewidth=0.3,
    )
    ax.grid(
        which="minor",
        alpha=0.10,
        linewidth=0.2,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(r"Polarity $P$")
    ax.set_ylabel(r"Milling $M$")

    despine(ax)


def add_time_colored_trajectory(
    ax,
    P,
    M,
    cmap=TIME_CMAP,
):
    """
    Draw a PM trajectory colored by normalized simulation time.

    Returns the scatter object used for the shared colorbar.
    """

    P_plot = P[::PLOT_STEP]
    M_plot = M[::PLOT_STEP]

    if len(P_plot) < 2:
        raise ValueError(
            "Not enough points to draw trajectory."
        )

    time = np.linspace(
        0.0,
        1.0,
        len(P_plot),
    )

    points = np.column_stack(
        [P_plot, M_plot]
    ).reshape(-1, 1, 2)

    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1,
    )

    norm = plt.Normalize(
        vmin=0.0,
        vmax=1.0,
    )

    line_collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
    )

    line_collection.set_array(
        time[:-1]
    )
    line_collection.set_linewidth(
        0.35
    )
    line_collection.set_alpha(
        0.75
    )

    ax.add_collection(
        line_collection
    )

    scatter = ax.scatter(
        P_plot,
        M_plot,
        c=time,
        cmap=cmap,
        norm=norm,
        s=1.8,
        linewidths=0,
        alpha=0.45,
        zorder=2,
    )

    return scatter


def plot_single_trajectory(
    ax,
    data,
    group_size,
):
    """
    Plot one representative PM trajectory.
    """

    scatter = add_time_colored_trajectory(
        ax=ax,
        P=data["P"],
        M=data["M"],
    )

    # Mean position of the trajectory.
    ax.scatter(
        data["mean_P"],
        data["mean_M"],
        s=40,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=10,
    )

    ax.set_title(
        rf"{PANEL_LABELS[group_size]} $N={group_size}$",
        loc="left",
        pad=4,
    )

    configure_pm_axis(
        ax
    )

    return scatter


def plot_summary(
    ax,
    all_group_data,
    colors,
):
    """
    Plot mean PM positions as a function of group size.
    """

    mean_P = np.array(
        [
            all_group_data[group_size]["mean_P"]
            for group_size in N_VALUES
        ]
    )

    mean_M = np.array(
        [
            all_group_data[group_size]["mean_M"]
            for group_size in N_VALUES
        ]
    )

    # Line connecting mean states in increasing order of N.
    ax.plot(
        mean_P,
        mean_M,
        color="black",
        linewidth=1.1,
        zorder=3,
    )

    for group_size, x, y in zip(
        N_VALUES,
        mean_P,
        mean_M,
    ):
        ax.scatter(
            x,
            y,
            s=55,
            color=colors[group_size],
            edgecolor="white",
            linewidth=0.9,
            zorder=10,
        )

        ax.annotate(
            rf"$N={group_size}$",
            xy=(x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=7,
            color="black",
            zorder=11,
        )

    ax.set_title(
        "(E) Group-size dependence",
        loc="left",
        pad=4,
    )

    configure_pm_axis(
        ax
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[group_size],
            markeredgecolor="white",
            markersize=6,
            label=rf"$N={group_size}$",
        )
        for group_size in N_VALUES
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        handletextpad=0.3,
        borderaxespad=0.3,
    )


def plot_figure(
    all_group_data,
    out_pdf,
    out_png,
):
    """
    Create the complete supplementary figure.
    """

    fig = plt.figure(
        figsize=(9.2, 5.8),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[
            1.0,
            1.0,
            1.15,
        ],
    )

    trajectory_axes = {
        10: fig.add_subplot(grid[0, 0]),
        30: fig.add_subplot(grid[0, 1]),
        50: fig.add_subplot(grid[1, 0]),
        70: fig.add_subplot(grid[1, 1]),
    }

    summary_ax = fig.add_subplot(
        grid[:, 2]
    )

    # Colors used only in the summary panel.
    group_cmap = plt.get_cmap("viridis")

    colors = {
        group_size: group_cmap(
            index / (len(N_VALUES) - 1)
        )
        for index, group_size in enumerate(N_VALUES)
    }

    colorbar_mappable = None

    for group_size in N_VALUES:
        colorbar_mappable = plot_single_trajectory(
            ax=trajectory_axes[group_size],
            data=all_group_data[group_size],
            group_size=group_size,
        )

    plot_summary(
        ax=summary_ax,
        all_group_data=all_group_data,
        colors=colors,
    )

    # One shared colorbar for panels A-D.
    cbar = fig.colorbar(
        colorbar_mappable,
        ax=list(trajectory_axes.values()),
        orientation="horizontal",
        fraction=0.045,
        pad=0.06,
        aspect=35,
    )

    cbar.set_label(
        "Normalized simulation time"
    )

    cbar.set_ticks(
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    cbar.ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(all_group_data):
    print()
    print("Group-size summary")
    print("-" * 35)

    print(
        f"{'N':>5} "
        f"{'mean P':>12} "
        f"{'mean M':>12}"
    )

    for group_size in N_VALUES:
        data = all_group_data[group_size]

        print(
            f"{group_size:5d} "
            f"{data['mean_P']:12.4f} "
            f"{data['mean_M']:12.4f}"
        )


def main():
    set_paper_style()

    all_group_data = {
        group_size: load_group_data(group_size)
        for group_size in N_VALUES
    }

    print_summary(
        all_group_data
    )

    out_pdf = OUT_DIR / "Figure_S2.pdf"
    out_png = OUT_DIR / "Figure_S2.png"

    plot_figure(
        all_group_data=all_group_data,
        out_pdf=out_pdf,
        out_png=out_png,
    )

    print()
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()