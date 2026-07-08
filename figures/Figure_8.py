#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns

EPS = 1e-12

CONDITIONS = [
    ("kappa_0p5__okappa_1p5", "#34495E"),
    ("kappa_1p5__okappa_2", "#C06C3E"),
    ("kappa_2p5__okappa_3", "#5B8A72"),
]

KINDS = [
    ("schooling_tot", "schooling", "-"),
    ("milling_tan", "tangential", "--"),
    ("milling_rad", "radial", ":"),
]


def setup_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def load_positions_npz(path, key="pos"):
    X = np.load(path)[key]
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T,2,N), got {X.shape}")
    return X


def center_of_mass_trajectory(X):
    return X[:, 0, :].mean(axis=1), X[:, 1, :].mean(axis=1)


def relative_positions(X, cx, cy):
    return X[:, 0, :] - cx[:, None], X[:, 1, :] - cy[:, None]


def decompose_rad_tan(rx_all, ry_all):
    rx = rx_all[:-1]
    ry = ry_all[:-1]
    vx = np.diff(rx_all, axis=0)
    vy = np.diff(ry_all, axis=0)

    rnorm = np.hypot(rx, ry)
    erx = rx / np.maximum(rnorm, EPS)
    ery = ry / np.maximum(rnorm, EPS)

    Lz = (rx * vy - ry * vx).mean(axis=1)
    sgn = np.sign(Lz)
    sgn[sgn == 0] = 1.0

    etx = -ery * sgn[:, None]
    ety = erx * sgn[:, None]

    abs_v_rad = np.abs(vx * erx + vy * ery)
    abs_v_tan = np.abs(vx * etx + vy * ety)
    abs_v = np.hypot(vx, vy)

    return abs_v_rad, abs_v_tan, abs_v, Lz


def polarization_series(X):
    vx = np.diff(X[:, 0, :], axis=0)
    vy = np.diff(X[:, 1, :], axis=0)
    speed = np.hypot(vx, vy)

    ux = vx / np.maximum(speed, EPS)
    uy = vy / np.maximum(speed, EPS)

    return np.hypot(ux.mean(axis=1), uy.mean(axis=1))


def make_threshold(series, q=0.80):
    s = np.asarray(series, float)
    s = s[np.isfinite(s)]
    return np.quantile(s, q) if s.size else np.nan


def flight_events_with_mask(series, threshold, mask):
    flights = []
    acc = 0.0
    cnt = 0

    for v, m in zip(series, mask):
        ok = bool(m) and np.isfinite(v) and v >= threshold
        if ok:
            acc += float(v)
            cnt += 1
        else:
            if cnt > 0:
                flights.append(acc)
                acc = 0.0
                cnt = 0

    if cnt > 0:
        flights.append(acc)

    return np.asarray(flights, float)


def ccdf_truncated_powerlaw(grid, xmin, xmax, alpha):
    if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(alpha)):
        return None
    if xmin <= 0 or xmax <= xmin:
        return None

    grid = np.asarray(grid, float)
    out = np.zeros_like(grid)

    out[grid < xmin] = 1.0
    out[grid > xmax] = 0.0

    m = (grid >= xmin) & (grid <= xmax)
    a1 = 1.0 - alpha

    if abs(a1) > 1e-12:
        out[m] = (xmax ** a1 - grid[m] ** a1) / (xmax ** a1 - xmin ** a1)
    else:
        out[m] = np.log(xmax / grid[m]) / np.log(xmax / xmin)

    return out


def reconstruct_individual_flights_from_npz(
    npz_path,
    individual,
    kind,
    threshold,
    milling_q=0.80,
    schooling_q=0.80,
):
    X = load_positions_npz(npz_path)

    cx, cy = center_of_mass_trajectory(X)
    rx, ry = relative_positions(X, cx, cy)

    abs_v_rad, abs_v_tan, abs_v, Lz = decompose_rad_tan(rx, ry)
    P = polarization_series(X)

    milling_thr = make_threshold(np.abs(Lz), q=milling_q)
    schooling_thr = make_threshold(P, q=schooling_q)

    milling_mask = np.abs(Lz) >= milling_thr
    schooling_mask = P >= schooling_thr

    if kind == "milling_rad":
        return flight_events_with_mask(abs_v_rad[:, individual], threshold, milling_mask)
    if kind == "milling_tan":
        return flight_events_with_mask(abs_v_tan[:, individual], threshold, milling_mask)
    if kind == "schooling_tot":
        return flight_events_with_mask(abs_v[:, individual], threshold, schooling_mask)

    raise ValueError(f"unknown kind: {kind}")


def plot_figure8B_sample_fit(data_root: Path, out_dir: Path):
    setup_style()

    folder = "kappa_2p5__okappa_3"
    individual = 241

    npz_path = data_root / "figure8_sample_N300_kappa2p5_okappa3_rep00_seed30000000.npz"
    csv_path = data_root / folder / "levy_individual_metrics.csv"

    df = pd.read_csv(csv_path)

    sample_rows = df[
        (df["N"].astype(int) == 300)
        & (df["rep"].astype(int) == 0)
        & (df["seed"].astype(int) == 30000000)
        & (df["individual"].astype(int) == individual)
    ].copy()

    if sample_rows.empty:
        raise ValueError("Representative individual was not found in levy_individual_metrics.csv")

    fig, ax = plt.subplots(figsize=(4.25, 2.75), constrained_layout=True)

    colors = {
        "schooling_tot": "#34495E",
        "milling_tan": "#5B8A72",
        "milling_rad": "#C06C3E",
    }

    markers = {
        "schooling_tot": "o",
        "milling_tan": "^",
        "milling_rad": "s",
    }

    linestyles = {
        "schooling_tot": "-",
        "milling_tan": "--",
        "milling_rad": ":",
    }

    labels = {
        "schooling_tot": "schooling",
        "milling_tan": "tangential",
        "milling_rad": "radial",
    }

    handles = []
    handle_labels = []
    stats_lines = []

    for kind, _, _ in KINDS:
        row_df = sample_rows[sample_rows["kind"] == kind]
        if row_df.empty:
            continue

        row = row_df.iloc[0]

        threshold = float(row["threshold"])
        flights = reconstruct_individual_flights_from_npz(
            npz_path=npz_path,
            individual=individual,
            kind=kind,
            threshold=threshold,
        )

        x, y = empirical_ccdf(flights)
        if x is None:
            continue

        color = colors[kind]
        marker = markers[kind]
        ls = linestyles[kind]
        label = labels[kind]

        ax.loglog(
            x,
            y,
            linestyle="none",
            marker=marker,
            ms=2.6,
            color=color,
            alpha=0.75,
            markeredgewidth=0.0,
        )

        xmin = float(row["truncated_xmin"])
        alpha = float(row["truncated_alpha"])

        tail = finite_positive(flights)
        tail = tail[tail >= xmin]

        if tail.size > 0 and np.isfinite(xmin) and np.isfinite(alpha):
            idx0 = np.searchsorted(x, xmin, side="left")
            S0 = float(y[idx0] if idx0 < len(y) else y[-1])

            xmax = float(tail.max())
            if xmax > xmin:
                xx = np.logspace(np.log10(xmin), np.log10(xmax * 0.98), 250)
                yy = ccdf_truncated_powerlaw(xx, xmin, xmax, alpha)

                if yy is not None:
                    ax.loglog(xx, S0 * yy, linestyle=ls, color=color, lw=1.25)

        delta_aic = float(row["delta_aic_trunc_minus_exp"])
        stats_lines.append(rf"{label}: $\alpha={alpha:.2f}$, $\Delta$AIC={delta_aic:.1f}")

        h = ax.scatter([], [], s=18, marker=marker, color=color, edgecolor="none", alpha=0.85)
        handles.append(h)
        handle_labels.append(label)

    ax.set_xlabel("flight length")
    ax.set_ylabel(r"$P(X \geq x)$")
    ax.set_ylim(1e-3, 1)

    ax.grid(which="major", color="0.88", linewidth=0.35)
    ax.grid(which="minor", color="0.94", linewidth=0.22)

    ax.legend(
        handles,
        handle_labels,
        frameon=False,
        fontsize=6,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
    )

    ax.text(
        1.03,
        0.02,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=5.5,
        va="bottom",
        ha="left",
        clip_on=False,
    )

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.75)
        sp.set_color("0.15")

    out_pdf = out_dir / "Figure_8B.pdf"
    out_png = out_dir / "Figure_8B.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


def plot_figure8C_alpha_violin(data_root: Path, out_dir: Path):
    setup_style()

    regime_labels = {
        "kappa_0p5__okappa_1p5": "SMS",
        "kappa_1p5__okappa_2": "MS",
        "kappa_2p5__okappa_3": "Milling",
    }

    kind_labels = {
        "schooling_tot": "schooling",
        "milling_tan": "tangential",
        "milling_rad": "radial",
    }

    kind_order = ["schooling_tot", "milling_tan", "milling_rad"]

    kind_markers = {
        "schooling_tot": "o",
        "milling_tan": "^",
        "milling_rad": "s",
    }

    regime_kind_colors = {
        "kappa_0p5__okappa_1p5": {
            "schooling_tot": "#243B53",
            "milling_tan": "#4E79A7",
            "milling_rad": "#A6C8E1",
        },
        "kappa_1p5__okappa_2": {
            "schooling_tot": "#8C3F1F",
            "milling_tan": "#C06C3E",
            "milling_rad": "#E8B88A",
        },
        "kappa_2p5__okappa_3": {
            "schooling_tot": "#2E5E3E",
            "milling_tan": "#5B8A72",
            "milling_rad": "#B7D7B0",
        },
    }

    base_positions = np.arange(len(CONDITIONS), dtype=float)[::-1]

    kind_offsets = {
        "schooling_tot": 0.22,
        "milling_tan": 0.00,
        "milling_rad": -0.22,
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    rng = np.random.default_rng(123)

    for r, (folder, _) in enumerate(CONDITIONS):
        csv_path = data_root / folder / "levy_individual_metrics.csv"
        df = pd.read_csv(csv_path)

        base = base_positions[r]

        for kind in kind_order:
            vals = pd.to_numeric(
                df.loc[df["kind"] == kind, "truncated_alpha"],
                errors="coerce",
            ).to_numpy(float)

            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= 1.6) & (vals < 8)]

            if vals.size == 0:
                continue

            pos = base + kind_offsets[kind]
            color = regime_kind_colors[folder][kind]

            vp = ax.violinplot(
                vals,
                positions=[pos],
                orientation="horizontal",
                widths=0.26,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.36)

            n_show = min(vals.size, 180)
            idx = rng.choice(vals.size, size=n_show, replace=False)
            jitter = rng.normal(0.0, 0.020, size=n_show)

            ax.scatter(
                vals[idx],
                np.full(n_show, pos) + jitter,
                s=8.0,
                marker=kind_markers[kind],
                facecolor=color,
                edgecolor="none",
                alpha=0.75,
                rasterized=True,
            )

            med = np.median(vals)
            ax.plot(
                [med, med],
                [pos - 0.07, pos + 0.07],
                color=color,
                lw=1.35,
                solid_capstyle="round",
                zorder=4,
            )

    ax.axvline(3.0, color="0.30", linestyle="--", linewidth=0.75, zorder=0)

    ax.text(
        3.03,
        base_positions[0] + 0.55,
        r"$\alpha=3$",
        fontsize=6,
        color="0.25",
        ha="left",
        va="bottom",
    )

    ax.set_yticks(base_positions)
    ax.set_yticklabels([regime_labels[f] for f, _ in CONDITIONS])
    ax.set_xlabel(r"individual $\alpha$")
    ax.set_xlim(1.6, 4.25)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(which="major", axis="x", color="0.88", linewidth=0.35)
    ax.grid(which="minor", axis="x", color="0.94", linewidth=0.22)

    handles = []
    labels = []

    for kind in kind_order:
        h = ax.scatter(
            [],
            [],
            s=18,
            marker=kind_markers[kind],
            facecolor="0.25",
            edgecolor="none",
            alpha=0.85,
        )
        handles.append(h)
        labels.append(kind_labels[kind])

    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6,
        loc="lower right",
        handletextpad=0.4,
        borderpad=0.2,
    )

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.75)
        sp.set_color("0.15")

    out_pdf = out_dir / "Figure_8C.pdf"
    out_png = out_dir / "Figure_8C.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="data/processed/figure8",
        help="Directory containing Figure 8 processed data.",
    )
    ap.add_argument(
        "--out",
        default="figures/output",
        help="Output directory for Figure 8 panels.",
    )

    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    ensure_dir(out_dir)

    plot_figure8B_sample_fit(data_root, out_dir)
    plot_figure8C_alpha_violin(data_root, out_dir)


if __name__ == "__main__":
    main()