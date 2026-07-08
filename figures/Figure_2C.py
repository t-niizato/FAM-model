#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_2C.py

Generate Figure 2C from a representative trajectory sample.

Input:
    data/trajectories/PM_trajectory_N300_kper2p5_kop3.npz

Output:
    figures/output/Figure_2C.pdf
    figures/output/Figure_2C.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


try:
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None


EPS = 1e-12

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "trajectories"
    / "PM_trajectory_N300_kper2p5_kop3.npz"
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


def load_positions(path: Path, key: str = "pos") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")

    data = np.load(path, allow_pickle=False)

    if key not in data:
        raise KeyError(
            f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}"
        )

    pos = data[key]

    if pos.ndim != 3 or pos.shape[1] != 2:
        raise ValueError(f"Expected pos shape (T, 2, N), got {pos.shape}")

    return np.transpose(pos, (0, 2, 1))  # (T, N, 2)


def velocities_from_positions(X: np.ndarray, dt: float = 1.0) -> np.ndarray:
    return (X[1:] - X[:-1]) / dt


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

    cross = R[..., 0] * U[..., 1] - R[..., 1] * U[..., 0]
    num = cross.sum(axis=1)
    den = (
        np.linalg.norm(R, axis=-1)
        * np.linalg.norm(U, axis=-1)
    ).sum(axis=1) + EPS

    return num / den


def moving_average_reflect(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, float)

    if w <= 1:
        return x.copy()

    if w % 2 == 0:
        w += 1

    pad = w // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    ker = np.ones(w, dtype=float) / w

    return np.convolve(xp, ker, mode="valid")


def forward_fill_sign(x: np.ndarray, eps: float) -> np.ndarray:
    s = np.sign(x).astype(int)
    s[np.abs(x) <= eps] = 0

    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]

    if len(s) > 0 and s[0] == 0:
        nz = np.flatnonzero(s)

        if len(nz) > 0:
            s[: nz[0]] = s[nz[0]]
        else:
            s[:] = 1

    return s


def detect_flip_times(
    omega: np.ndarray,
    smooth_window: int = 21,
    sign_eps: float = 0.01,
    min_gap: int = 200,
):
    omega_s = moving_average_reflect(omega, smooth_window)
    s = forward_fill_sign(omega_s, sign_eps)

    raw = np.where(s[1:] * s[:-1] < 0)[0] + 1

    if len(raw) == 0:
        return omega_s, np.array([], dtype=int)

    kept = [int(raw[0])]

    for t in raw[1:]:
        if int(t) - kept[-1] >= int(min_gap):
            kept.append(int(t))

    return omega_s, np.asarray(kept, dtype=int)


def add_trails(
    ax,
    X: np.ndarray,
    t: int,
    trail_len: int = 40,
    alpha: float = 0.10,
    lw: float = 0.25,
):
    t0 = max(0, t - trail_len + 1)
    seg = X[t0 : t + 1]

    if seg.shape[0] < 2:
        return

    for i in range(seg.shape[1]):
        ax.plot(
            seg[:, i, 0],
            seg[:, i, 1],
            color="black",
            alpha=alpha,
            linewidth=lw,
            zorder=1,
        )


def axis_limits_from_window(
    X: np.ndarray,
    t_center: int,
    half_window: int,
    pad: float = 1.0,
):
    t0 = max(0, t_center - half_window)
    t1 = min(X.shape[0] - 1, t_center + half_window)

    xx = X[t0 : t1 + 1, :, 0]
    yy = X[t0 : t1 + 1, :, 1]

    xmin, xmax = float(np.min(xx)), float(np.max(xx))
    ymin, ymax = float(np.min(yy)), float(np.max(yy))

    size = max(xmax - xmin, ymax - ymin, 1e-9)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * size + pad

    return cx - half, cx + half, cy - half, cy + half


def plot_single_omega_flip_figure(
    X: np.ndarray,
    omega: np.ndarray,
    flip_t: int,
    out_pdf: Path,
    out_png: Path,
    panel_offsets=(-120, -60, -20, 0, 20, 60, 120),
    axis_half_window: int = 120,
    point_size: float = 3.5,
    trail_len: int = 40,
):
    T = X.shape[0]

    times = [
        int(np.clip(flip_t + int(dt), 0, T - 1))
        for dt in panel_offsets
    ]

    dt_labels = [
        f"{int(dt):+d}" if int(dt) != 0 else "+0"
        for dt in panel_offsets
    ]

    x0, x1, y0, y1 = axis_limits_from_window(
        X,
        flip_t,
        axis_half_window,
        pad=1.0,
    )

    n = len(times)

    fig = plt.figure(figsize=(1.18 * n, 2.25), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[0.75, 1.25])

    ax_top = fig.add_subplot(gs[0, :])

    t = np.arange(len(omega))

    ax_top.plot(
        t,
        omega,
        color="black",
        linewidth=0.55,
        label=r"$\Omega$",
    )

    ax_top.axhline(
        0.0,
        color="0.45",
        linestyle="--",
        linewidth=0.45,
    )

    ax_top.axvline(
        flip_t,
        color="black",
        linestyle=":",
        linewidth=0.65,
    )

    for tt in times:
        ax_top.axvline(
            tt,
            color="0.65",
            alpha=0.35,
            linewidth=0.35,
        )

    ax_top.set_xlim(0, len(omega) - 1)
    ax_top.set_ylim(-1.05, 1.05)

    ax_top.set_ylabel(r"$\Omega$", fontsize=6)
    ax_top.set_xlabel("Simulation step", fontsize=6, labelpad=1)

    ax_top.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_top.xaxis.set_major_locator(MultipleLocator(25000))
    ax_top.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax_top.tick_params(labelsize=5, width=0.45, length=2)
    ax_top.grid(True, alpha=0.16, linewidth=0.25)

    ax_top.legend(
        frameon=False,
        fontsize=4.8,
        loc="lower right",
        handlelength=1.6,
    )

    for spine in ax_top.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("0.25")

    for j, (tt, dt_lab) in enumerate(zip(times, dt_labels)):
        ax = fig.add_subplot(gs[1, j])

        add_trails(
            ax,
            X,
            tt,
            trail_len=trail_len,
            alpha=0.055,
            lw=0.16,
        )

        ax.scatter(
            X[tt, :, 0],
            X[tt, :, 1],
            s=point_size,
            facecolors="#023abd",
            edgecolors="white",
            linewidths=0.30,
            zorder=2,
        )

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"t = {tt}\n(dt={dt_lab})",
            fontsize=5.0,
            pad=1.5,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.50)
            spine.set_color("black")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    set_paper_style()

    X = load_positions(DATA_FILE, key="pos")

    omega = normalized_spin_omega_2d(
        X,
        dt=1.0,
        remove_translation=True,
    )

    start_step = 5000
    smooth_window = 21
    sign_eps = 0.01
    min_gap = 200
    event_index = 0

    start_step = min(
        max(int(start_step), 0),
        max(len(omega) - 1, 0),
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
            f"No omega flips detected after start_step={start_step} in {DATA_FILE}"
        )

    if event_index >= len(flip_times):
        raise RuntimeError(
            f"event_index={event_index} but only {len(flip_times)} flips detected"
        )

    flip_t = int(flip_times[event_index])

    out_pdf = OUT_DIR / "Figure_2C.pdf"
    out_png = OUT_DIR / "Figure_2C.png"

    plot_single_omega_flip_figure(
        X=X,
        omega=omega,
        flip_t=flip_t,
        out_pdf=out_pdf,
        out_png=out_png,
        panel_offsets=(-120, -60, -20, 0, 20, 60, 120),
        axis_half_window=120,
        point_size=3.5,
        trail_len=40,
    )

    print(f"flip_t: {flip_t}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()