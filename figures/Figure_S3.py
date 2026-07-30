#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supporting_Figure_Omega_Flips_full_timeseries.py

Generate supporting examples of rotational-direction flips
for multiple parameter conditions.

For each parameter condition:
    1. Plot the full Omega time series.
    2. Mark the selected flip and snapshot times.
    3. Show seven snapshots around the selected flip.

Inputs:
    data/trajectories/PM_trajectory_N300_kper0p5_kop1p5.npz
    data/trajectories/PM_trajectory_N300_kper0p5_kop3.npz
    data/trajectories/PM_trajectory_N300_kper2p5_kop3.npz

Outputs:
    figures/output/Supporting_Figure_Omega_Flips.pdf
    figures/output/Supporting_Figure_Omega_Flips.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


EPS = 1e-12

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DIR = ROOT / "data" / "trajectories"

OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Conditions shown in the supporting figure
# ============================================================

CONDITIONS = [
    {
        "file": "PM_trajectory_N300_kper0p5_kop1p5.npz",
        "label": (
            r"$\kappa_{\mathrm{per}}=0.5,\ "
            r"\kappa_{\mathrm{op}}=1.5$"
        ),
        "start_step": 5000,
    },
    {
        "file": "PM_trajectory_N300_kper0p5_kop3.npz",
        "label": (
            r"$\kappa_{\mathrm{per}}=0.5,\ "
            r"\kappa_{\mathrm{op}}=3.0$"
        ),
        "start_step": 5000,
    },
    {
        "file": "PM_trajectory_N300_kper2p5_kop3.npz",
        "label": (
            r"$\kappa_{\mathrm{per}}=2.5,\ "
            r"\kappa_{\mathrm{op}}=3.0$"
        ),
        "start_step": 5000,
    },
]

def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
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


def load_positions(
    path: Path,
    key: str = "pos",
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")

    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in {path}. "
                f"Available keys: {list(data.keys())}"
            )

        pos = np.asarray(data[key])

    if pos.ndim != 3 or pos.shape[1] != 2:
        raise ValueError(
            f"Expected pos shape (T, 2, N), got {pos.shape}"
        )

    return np.transpose(pos, (0, 2, 1))  # (T, N, 2)


def velocities_from_positions(
    X: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    return np.diff(X, axis=0) / dt


def normalized_spin_omega_2d(
    X: np.ndarray,
    dt: float = 1.0,
    remove_translation: bool = True,
) -> np.ndarray:
    V = velocities_from_positions(X, dt=dt)
    Xmid = X[:-1]

    xcm = Xmid.mean(axis=1, keepdims=True)
    R = Xmid - xcm

    if remove_translation:
        Vcm = V.mean(axis=1, keepdims=True)
        U = V - Vcm
    else:
        U = V

    cross = (
        R[..., 0] * U[..., 1]
        - R[..., 1] * U[..., 0]
    )

    num = cross.sum(axis=1)

    den = (
        np.linalg.norm(R, axis=-1)
        * np.linalg.norm(U, axis=-1)
    ).sum(axis=1)

    return num / (den + EPS)


def moving_average_reflect(
    x: np.ndarray,
    w: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if w <= 1:
        return x.copy()

    if w % 2 == 0:
        w += 1

    pad = w // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    kernel = np.ones(w, dtype=float) / w

    return np.convolve(xp, kernel, mode="valid")


def forward_fill_sign(
    x: np.ndarray,
    eps: float,
) -> np.ndarray:
    signs = np.sign(x).astype(int)
    signs[np.abs(x) <= eps] = 0

    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    if len(signs) > 0 and signs[0] == 0:
        nonzero = np.flatnonzero(signs)

        if len(nonzero) > 0:
            signs[: nonzero[0]] = signs[nonzero[0]]
        else:
            signs[:] = 1

    return signs


def detect_flip_times(
    omega: np.ndarray,
    smooth_window: int = 21,
    sign_eps: float = 0.01,
    min_gap: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    omega_s = moving_average_reflect(
        omega,
        smooth_window,
    )

    signs = forward_fill_sign(
        omega_s,
        sign_eps,
    )

    raw = np.where(
        signs[1:] * signs[:-1] < 0
    )[0] + 1

    if len(raw) == 0:
        return omega_s, np.array([], dtype=int)

    kept = [int(raw[0])]

    for t in raw[1:]:
        t = int(t)

        if t - kept[-1] >= min_gap:
            kept.append(t)

    return omega_s, np.asarray(kept, dtype=int)


def select_strongest_flip_event(
    omega: np.ndarray,
    start_step: int,
    smooth_window: int,
    sign_eps: float,
    min_gap: int,
    evaluation_window: int = 100,
    exclusion_window: int = 20,
    min_mean_abs: float = 0.20,
):
    """
    Select the clearest large-scale reversal of omega.

    For every detected sign flip, calculate the mean omega before and
    after the event. The event with the largest persistent difference
    between the two sides is selected.

    Parameters
    ----------
    omega
        Raw omega time series.

    start_step
        Ignore events occurring before this step.

    smooth_window
        Window used for flip detection and event scoring.

    sign_eps
        Values close to zero are treated as neutral during sign detection.

    min_gap
        Minimum distance between successive flip candidates.

    evaluation_window
        Number of steps used on each side of the flip.

    exclusion_window
        Steps immediately surrounding the zero crossing that are excluded
        from the before/after averages.

    min_mean_abs
        Minimum absolute mean omega required on both sides.
    """

    start_step = int(
        np.clip(
            start_step,
            0,
            max(len(omega) - 1, 0),
        )
    )

    omega_s = moving_average_reflect(
        omega,
        smooth_window,
    )

    _, flip_local = detect_flip_times(
        omega[start_step:],
        smooth_window=smooth_window,
        sign_eps=sign_eps,
        min_gap=min_gap,
    )

    flip_times = flip_local + start_step

    if len(flip_times) == 0:
        raise RuntimeError(
            f"No flips detected after step {start_step}"
        )

    candidates = []

    for flip_t in flip_times:
        flip_t = int(flip_t)

        before_start = flip_t - exclusion_window - evaluation_window
        before_end = flip_t - exclusion_window

        after_start = flip_t + exclusion_window
        after_end = flip_t + exclusion_window + evaluation_window

        # Skip events too close to the beginning or end.
        if before_start < 0 or after_end > len(omega_s):
            continue

        before = omega_s[before_start:before_end]
        after = omega_s[after_start:after_end]

        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))

        # A genuine reversal must have opposite signs.
        if mean_before * mean_after >= 0:
            continue

        # Reject weak fluctuations around zero.
        if (
            abs(mean_before) < min_mean_abs
            or abs(mean_after) < min_mean_abs
        ):
            continue

        reversal_amplitude = abs(
            mean_after - mean_before
        )

        # Fraction of points agreeing with the dominant sign.
        before_sign_consistency = np.mean(
            np.sign(before) == np.sign(mean_before)
        )

        after_sign_consistency = np.mean(
            np.sign(after) == np.sign(mean_after)
        )

        consistency = (
            before_sign_consistency
            * after_sign_consistency
        )

        score = reversal_amplitude * consistency

        candidates.append(
            {
                "flip_t": flip_t,
                "score": score,
                "mean_before": mean_before,
                "mean_after": mean_after,
                "amplitude": reversal_amplitude,
                "consistency": consistency,
            }
        )

    if len(candidates) == 0:
        raise RuntimeError(
            "Flip candidates were detected, but none satisfied "
            "the strong-reversal criteria. Try reducing "
            "min_mean_abs or evaluation_window."
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    return (
        omega_s,
        flip_times,
        int(best["flip_t"]),
        candidates,
    )

def add_trails(
    ax,
    X: np.ndarray,
    t: int,
    trail_len: int = 40,
    alpha: float = 0.055,
    lw: float = 0.16,
) -> None:
    t0 = max(0, t - trail_len + 1)
    segment = X[t0 : t + 1]

    if segment.shape[0] < 2:
        return

    for i in range(segment.shape[1]):
        ax.plot(
            segment[:, i, 0],
            segment[:, i, 1],
            color="black",
            alpha=alpha,
            linewidth=lw,
            zorder=1,
        )


def axis_limits_from_window(
    X: np.ndarray,
    t_center: int,
    half_window: int,
    pad_fraction: float = 0.08,
) -> tuple[float, float, float, float]:
    t0 = max(0, t_center - half_window)
    t1 = min(
        X.shape[0] - 1,
        t_center + half_window,
    )

    window = X[t0 : t1 + 1]

    xmin = float(np.min(window[..., 0]))
    xmax = float(np.max(window[..., 0]))
    ymin = float(np.min(window[..., 1]))
    ymax = float(np.max(window[..., 1]))

    size = max(
        xmax - xmin,
        ymax - ymin,
        1e-9,
    )

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    half = 0.5 * size * (1.0 + 2.0 * pad_fraction)

    return (
        cx - half,
        cx + half,
        cy - half,
        cy + half,
    )


def choose_x_major_step(n_steps: int) -> int:
    """
    Choose a readable major-tick interval for the full time series.
    """
    if n_steps <= 10_000:
        return 2_000

    if n_steps <= 50_000:
        return 10_000

    if n_steps <= 150_000:
        return 25_000

    if n_steps <= 300_000:
        return 50_000

    return 100_000


def plot_supporting_figure(
    results: list[dict],
    out_pdf: Path,
    out_png: Path,
    panel_offsets=(-120, -60, -20, 0, 20, 60, 120),
    spatial_half_window: int = 120,
    trail_len: int = 40,
    point_size: float = 3.0,
) -> None:
    n_conditions = len(results)
    n_snapshots = len(panel_offsets)

    # Each condition uses two rows:
    #   upper row: full Omega time series
    #   lower row: snapshots around the selected flip
    fig = plt.figure(
        figsize=(
            1.18 * n_snapshots,
            2.15 * n_conditions,
        ),
        constrained_layout=True,
    )

    height_ratios = []

    for _ in range(n_conditions):
        height_ratios.extend([0.72, 1.15])

    gs = fig.add_gridspec(
        2 * n_conditions,
        n_snapshots,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.05,
    )

    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for condition_index, result in enumerate(results):
        X = result["X"]
        omega = result["omega"]
        flip_t = result["flip_t"]
        label = result["label"]

        T = X.shape[0]

        times = [
            int(np.clip(flip_t + offset, 0, T - 1))
            for offset in panel_offsets
        ]

        x0, x1, y0, y1 = axis_limits_from_window(
            X,
            flip_t,
            half_window=spatial_half_window,
        )

        omega_row = 2 * condition_index
        snapshot_row = omega_row + 1

        # ====================================================
        # Full Omega time series
        # ====================================================
        ax_omega = fig.add_subplot(
            gs[omega_row, :]
        )

        time_axis = np.arange(len(omega))

        ax_omega.plot(
            time_axis,
            omega,
            color="black",
            linewidth=0.50,
            label=r"$\Omega$",
        )

        ax_omega.axhline(
            0.0,
            color="0.45",
            linestyle="--",
            linewidth=0.45,
        )

        ax_omega.axvline(
            flip_t,
            color="black",
            linestyle=":",
            linewidth=0.75,
            zorder=4,
        )

        for tt in times:
            ax_omega.axvline(
                tt,
                color="0.65",
                alpha=0.35,
                linewidth=0.35,
                zorder=3,
            )

        ax_omega.set_xlim(
            0,
            max(len(omega) - 1, 1),
        )
        ax_omega.set_ylim(-1.05, 1.05)

        ax_omega.set_ylabel(
            r"$\Omega$",
            fontsize=6,
        )
        ax_omega.set_xlabel(
            "Simulation step",
            fontsize=6,
            labelpad=1,
        )

        ax_omega.yaxis.set_major_locator(
            MultipleLocator(0.5)
        )
        ax_omega.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )

        x_major_step = choose_x_major_step(
            len(omega)
        )

        ax_omega.xaxis.set_major_locator(
            MultipleLocator(x_major_step)
        )
        ax_omega.xaxis.set_major_formatter(
            FormatStrFormatter("%.0f")
        )

        ax_omega.tick_params(
            labelsize=5,
            width=0.45,
            length=2,
        )

        ax_omega.grid(
            True,
            alpha=0.16,
            linewidth=0.25,
        )

        ax_omega.text(
            0.008,
            0.94,
            f"{panel_letters[condition_index]}  {label}",
            transform=ax_omega.transAxes,
            ha="left",
            va="top",
            fontsize=6,
        )

        ax_omega.text(
            0.992,
            0.06,
            rf"$t_{{\mathrm{{flip}}}}={flip_t}$",
            transform=ax_omega.transAxes,
            ha="right",
            va="bottom",
            fontsize=4.8,
        )

        if condition_index == 0:
            ax_omega.legend(
                frameon=False,
                fontsize=4.8,
                loc="lower right",
                handlelength=1.6,
            )

        for spine in ax_omega.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_color("0.25")

        # ====================================================
        # Snapshots around the selected flip
        # ====================================================
        for col, (tt, offset) in enumerate(
            zip(times, panel_offsets)
        ):
            ax = fig.add_subplot(
                gs[snapshot_row, col]
            )

            add_trails(
                ax,
                X,
                tt,
                trail_len=trail_len,
            )

            ax.scatter(
                X[tt, :, 0],
                X[tt, :, 1],
                s=point_size,
                facecolors="#023abd",
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect(
                "equal",
                adjustable="box",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if offset == 0:
                title = r"$t_{\mathrm{flip}}$"
            elif offset > 0:
                title = (
                    rf"$t_{{\mathrm{{flip}}}}+{offset}$"
                )
            else:
                title = (
                    rf"$t_{{\mathrm{{flip}}}}{offset}$"
                )

            ax.set_title(
                title,
                fontsize=4.8,
                pad=1.2,
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(
                    0.8 if offset == 0 else 0.45
                )
                spine.set_color("black")

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


def main() -> None:
    set_paper_style()

    smooth_window = 21
    sign_eps = 0.01
    min_gap = 200

    results = []

    for condition in CONDITIONS:
        path = TRAJECTORY_DIR / condition["file"]

        X = load_positions(path)

        omega = normalized_spin_omega_2d(
            X,
            dt=1.0,
            remove_translation=True,
        )

        omega_s, all_flips, flip_t, candidates = (
            select_strongest_flip_event(
                omega=omega,
                start_step=condition["start_step"],
                smooth_window=smooth_window,
                sign_eps=sign_eps,
                min_gap=min_gap,
                evaluation_window=100,
                exclusion_window=20,
                min_mean_abs=0.20,
            )
        )

        best = candidates[0]

        print(
            f"{condition['file']}: "
            f"{len(all_flips)} flips detected, "
            f"selected flip_t={flip_t}, "
            f"mean_before={best['mean_before']:.3f}, "
            f"mean_after={best['mean_after']:.3f}, "
            f"amplitude={best['amplitude']:.3f}, "
            f"consistency={best['consistency']:.3f}, "
            f"score={best['score']:.3f}"
        )

        results.append(
            {
                "X": X,
                "omega": omega,
                "omega_s": omega_s,
                "flip_t": flip_t,
                "label": condition["label"],
                "file": condition["file"],
            }
        )

    out_pdf = (
        OUT_DIR
        / "Figure_S3.pdf"
    )

    out_png = (
        OUT_DIR
        / "Figure_S3.png"
    )

    plot_supporting_figure(
        results=results,
        out_pdf=out_pdf,
        out_png=out_png,
        panel_offsets=(-120, -60, -20, 0, 20, 60, 120),
        spatial_half_window=120,
        trail_len=40,
        point_size=3.0,
    )

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()