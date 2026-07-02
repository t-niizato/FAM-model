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
from matplotlib.collections import LineCollection

EPS = 1e-12

CONDITIONS = [
    ("kappa_0p5__okappa_1p5", r"$\kappa_{\rm per}=0.5,\ \kappa_{\rm op}=1.5$", "#34495E"),
    ("kappa_1p5__okappa_2",   r"$\kappa_{\rm per}=1.5,\ \kappa_{\rm op}=2.0$", "#C06C3E"),
    ("kappa_2p5__okappa_3",   r"$\kappa_{\rm per}=2.5,\ \kappa_{\rm op}=3.0$", "#5B8A72"),
]


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


def fmt_float_for_dir(x: float, nd: int = 6) -> str:
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def build_npz_path_from_row(row, position_root: Path) -> Path:
    N = int(row["N"])
    pk = float(row["percept_kappa"])
    ok = float(row["option_kappa"])
    rep = int(row["rep"])
    seed = int(row["seed"])

    return (
        position_root
        / f"N{N}"
        / f"kappa_{fmt_float_for_dir(pk)}"
        / f"okappa_{fmt_float_for_dir(ok)}"
        / f"pos_rep{rep:02d}_seed{seed}.npz"
    )


def resolve_npz_path(row, position_root: Path) -> Path:
    p_raw = row.get("path", None)
    if p_raw is not None and str(p_raw) not in {"", "nan", "None"}:
        p = Path(str(p_raw)).expanduser()
        if p.exists():
            return p.resolve()

    return build_npz_path_from_row(row, position_root).resolve()


def load_positions_npz(path, key="pos"):
    data = np.load(path)
    X = data[key]
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T,2,N), got {X.shape}")
    return X


def center_of_mass_trajectory(X):
    return X[:, 0, :].mean(axis=1), X[:, 1, :].mean(axis=1)


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


def params_from_row(row, prefix):
    def get(col):
        try:
            return float(row.get(f"{prefix}_{col}", np.nan))
        except Exception:
            return np.nan

    return {
        "xmin": get("xmin"),
        "alpha": get("alpha"),
        "xc": get("xc"),
        "aic": get("aic"),
        "p_value": get("p_value"),
    }


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

    if abs(a1) > 1e-12:
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


def setup_style():
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


def plot_com_alpha_summary(analysis_root: Path, out_dir: Path):
    setup_style()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.8, 2.25),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    for i, (ax, (folder, title, color)) in enumerate(zip(axes, CONDITIONS)):
        csv_path = analysis_root / folder / "com_all_runs" / "com_all_runs_metrics.csv"
        df = pd.read_csv(csv_path)

        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df["truncated_alpha"] = pd.to_numeric(df["truncated_alpha"], errors="coerce")
        df = df[np.isfinite(df["N"]) & np.isfinite(df["truncated_alpha"])]

        tmp = (
            df.groupby("N")["truncated_alpha"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("N")
        )

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

        ax.set_title(f"({chr(97+i)})", fontsize=8, pad=3)
        ax.set_xlabel(r"$N$")

        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(which="major", color="0.88", linewidth=0.35)
        ax.grid(which="minor", color="0.94", linewidth=0.22)

        ax.tick_params(which="major", length=3.0, width=0.65, pad=2)
        ax.tick_params(which="minor", length=1.7, width=0.45)

        sns.despine(ax=ax)

    axes[0].set_ylabel(r"COM $\alpha$")

    out_pdf = out_dir / "paper_com_alpha_summary_three_conditions.pdf"
    out_png = out_dir / "paper_com_alpha_summary_three_conditions.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")


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
        cx[-1], cy[-1],
        s=12,
        facecolor="white",
        edgecolor="0.12",
        linewidth=0.7,
        zorder=3
    )

    pad_x = 0.05 * (np.nanmax(cx) - np.nanmin(cx) + 1e-12)
    pad_y = 0.05 * (np.nanmax(cy) - np.nanmin(cy) + 1e-12)

    ax.set_xlim(np.nanmin(cx) - pad_x, np.nanmax(cx) + pad_x)
    ax.set_ylim(np.nanmin(cy) - pad_y, np.nanmax(cy) + pad_y)

    # 箱の大きさを揃える。データは正規化しない。
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel(r"$x$")
    ax.grid(which="major", color="0.90", linewidth=0.30)

    ax.tick_params(which="major", length=3.0, width=0.65, pad=2)

    # 四角で囲う
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("0.15")

def plot_com_trajectory_three_conditions(
    analysis_root: Path,
    position_root: Path,
    out_dir: Path,
    prefer_N=None,
):
    setup_style()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.8, 2.25),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    for i, (ax, (folder, title, color)) in enumerate(zip(axes, CONDITIONS)):
        csv_path = analysis_root / folder / "com_all_runs" / "com_all_runs_metrics.csv"
        df = pd.read_csv(csv_path)

        row = choose_com_row(df, prefer_N=prefer_N)
        if row is None:
            ax.set_title(f"({chr(97+i)})")
            continue

        npz_path = resolve_npz_path(row, position_root)
        X = load_positions_npz(npz_path, key="pos")
        cx, cy = center_of_mass_trajectory(X)

        label = f"({chr(97+i)}) N={int(row['N'])}"
        plot_one_com_trajectory(ax, cx, cy, color, label)

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    suffix = "representative" if prefer_N is None else f"N{int(prefer_N)}"

    out_pdf = out_dir / f"paper_COM_trajectory_three_conditions_{suffix}.pdf"
    out_png = out_dir / f"paper_COM_trajectory_three_conditions_{suffix}.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")

def choose_com_row(df, prefer_N=None):
    df = df.copy()
    if prefer_N is not None:
        sub = df[df["N"].astype(int) == int(prefer_N)].copy()
        if not sub.empty:
            df = sub

    df["flight_count_num"] = pd.to_numeric(df["flight_count"], errors="coerce")
    df["delta_num"] = pd.to_numeric(df["delta_aic_trunc_minus_exp"], errors="coerce")

    df = df[np.isfinite(df["flight_count_num"])].copy()
    if df.empty:
        return None

    return df.sort_values(
        ["flight_count_num", "delta_num"],
        ascending=[False, True],
    ).iloc[0]


def plot_one_ccdf(ax, flights, trunc_params, exp_params, color, title):
    x_all, y_all = empirical_ccdf(flights)

    if x_all is None:
        ax.set_title(title)
        return

    ax.loglog(
        x_all,
        y_all,
        ".",
        ms=2.4,
        color=color,
        alpha=0.50,
        label="Empirical",
    )

    xmin = trunc_params.get("xmin", np.nan)
    alpha = trunc_params.get("alpha", np.nan)

    if not np.isfinite(xmin):
        xmin = exp_params.get("xmin", np.nan)

    lam = exp_params.get("alpha", np.nan)

    if np.isfinite(xmin) and xmin > 0:
        idx0 = np.searchsorted(x_all, xmin, side="left")
        S0 = float(y_all[idx0] if idx0 < len(y_all) else y_all[-1])

        tail = np.sort(finite_positive(flights))
        tail = tail[tail >= xmin]

        if tail.size > 0:
            x_tail, y_tail_cond = empirical_ccdf(tail)
            y_tail = S0 * y_tail_cond

            # ax.loglog(
            #     x_tail,
            #     y_tail,
            #     "o",
            #     ms=3.0,
            #     color=color,
            #     alpha=0.85,
            #     markeredgewidth=0.0,
            #     label="Tail",
            # )

            xmax_data = float(tail.max())
            xmax_model = xmax_data * 0.98

            if xmax_model > xmin:
                x_model = np.logspace(np.log10(xmin), np.log10(xmax_model), 300)

                y_tr = ccdf_truncated_powerlaw(x_model, xmin, xmax_data, alpha)
                if y_tr is not None:
                    y_tr = S0 * y_tr
                    keep = np.isfinite(y_tr) & (y_tr > 0)
                    ax.loglog(
                        x_model[keep],
                        y_tr[keep],
                        "-",
                        color="0.10",
                        lw=1,
                        label="Truncated PL",
                    )

                y_ex = ccdf_shifted_exp(x_model, xmin, lam)
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

            ax.axvline(xmin, color="0.25", ls=":", lw=0.85)

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("flight length")
    ax.grid(which="major", color="0.88", linewidth=0.35)
    ax.grid(which="minor", color="0.94", linewidth=0.22)
    ax.tick_params(which="major", length=3.0, width=0.25, pad=2)
    ax.tick_params(which="minor", length=1.7, width=0.25)

    sns.despine(ax=ax)

def plot_com_ccdf_three_conditions(
    analysis_root: Path,
    position_root: Path,
    out_dir: Path,
    prefer_N=None,
):
    setup_style()

    for i, (folder, title, color) in enumerate(CONDITIONS):
        csv_path = analysis_root / folder / "com_all_runs" / "com_all_runs_metrics.csv"
        df = pd.read_csv(csv_path)

        row = choose_com_row(df, prefer_N=prefer_N)
        if row is None:
            print(f"[skip] no COM row: {folder}")
            continue

        npz_path = resolve_npz_path(row, position_root)
        X = load_positions_npz(npz_path, key="pos")
        cx, cy = center_of_mass_trajectory(X)
        step = center_step_lengths(cx, cy)

        thr = float(row["threshold"])
        flights = flight_lengths_from_series(step, thr)

        trunc_params = params_from_row(row, "truncated")
        exp_params = params_from_row(row, "shifted_exp")

        fig, ax = plt.subplots(
            figsize=(3.2, 2.7),
            constrained_layout=True,
        )

        label = f"({chr(97+i)})"
        plot_one_ccdf(
            ax,
            flights,
            trunc_params,
            exp_params,
            color,
            label,
        )

        ax.set_xlabel("flight length")
        ax.set_ylabel(r"$P(X \geq x)$")
        ax.set_ylim(1e-3, 1)

        ax.legend(
            frameon=False,
            fontsize=6,
            loc="upper right",
            handlelength=1.5,
        )

        suffix = "representative" if prefer_N is None else f"N{int(prefer_N)}"

        out_pdf = out_dir / f"paper_COM_ccdf_with_fit_{folder}_{suffix}.pdf"
        out_png = out_dir / f"paper_COM_ccdf_with_fit_{folder}_{suffix}.png"

        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"saved: {out_pdf}")
        print(f"saved: {out_png}")

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--analysis-root",
        required=True,
        help="directory containing kappa_*__okappa_* folders from levy analysis",
    )
    ap.add_argument(
        "--position-root",
        required=True,
        help="root directory of original position npz files",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output directory for paper figures",
    )
    ap.add_argument(
        "--ccdf-N",
        type=int,
        default=None,
        help="optional N used for CCDF panels; if omitted, the run with largest flight_count is used",
    )

    args = ap.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    position_root = Path(args.position_root).expanduser().resolve()

    if args.out is None:
        out_dir = analysis_root / "paper_figures"
    else:
        out_dir = Path(args.out).expanduser().resolve()

    ensure_dir(out_dir)

    plot_com_alpha_summary(analysis_root, out_dir)
    plot_com_ccdf_three_conditions(
        analysis_root=analysis_root,
        position_root=position_root,
        out_dir=out_dir,
        prefer_N=args.ccdf_N,
    )
    plot_com_trajectory_three_conditions(
        analysis_root=analysis_root,
        position_root=position_root,
        out_dir=out_dir,
        prefer_N=args.ccdf_N,
    )


if __name__ == "__main__":
    main()