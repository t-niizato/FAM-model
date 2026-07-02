#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure_omega_flip_single.py

Paper-style single omega-flip snapshot figure.

Default target:
    kappa_per = 2.5
    kappa_op  = 3.0
    N         = 300

This script selects one representative npz file, detects omega sign flips,
chooses one flip event, and exports one clean multi-snapshot figure around it.

Assumed directory structure:
    position_root/
      N300/
        kappa_2p5/
          okappa_3/
            pos_repXX_seedYY.npz

Outputs:
    Omega_flip_N300_kper2.5_kop3.pdf
    Omega_flip_N300_kper2.5_kop3.png
    selected_omega_flip_single.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

try:
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None

EPS = 1e-12
DEFAULT_CMAP = "viridis"


# =========================================================
# Style
# =========================================================
def set_paper_style():
    plt.rcParams.update({
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
    })


def despine(ax):
    if sns is not None:
        sns.despine(ax=ax)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# =========================================================
# Paths / metadata
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt_float_for_dir(x: float, nd: int = 6) -> str:
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def parse_run_info(path: str) -> dict:
    p = Path(path)
    N = None
    kappa = None
    okappa = None
    rep = None
    seed = None

    for s in p.parts:
        if re.fullmatch(r"N\d+", s):
            N = int(s[1:])
        elif s.startswith("kappa_"):
            kappa = float(s[len("kappa_"):].replace("m", "-").replace("p", "."))
        elif s.startswith("okappa_"):
            okappa = float(s[len("okappa_"):].replace("m", "-").replace("p", "."))

    m = re.search(r"pos_rep(\d+)_seed(\d+)\.npz$", p.name)
    if m:
        rep = int(m.group(1))
        seed = int(m.group(2))

    return {
        "path": str(path),
        "N": N,
        "percept_kappa": kappa,
        "option_kappa": okappa,
        "rep": rep,
        "seed": seed,
    }


def find_npz(position_root: Path, N: int, kappa_per: float, kappa_op: float):
    n_dir = position_root / f"N{N}"
    target = (
        n_dir
        / f"kappa_{fmt_float_for_dir(kappa_per)}"
        / f"okappa_{fmt_float_for_dir(kappa_op)}"
    )
    files = sorted(target.glob("*.npz"))
    if files:
        return files

    # fallback for minor naming differences
    files = []
    if n_dir.exists():
        for p in sorted(n_dir.glob("kappa_*/okappa_*/*.npz")):
            meta = parse_run_info(str(p))
            if (
                meta["N"] == N
                and np.isclose(meta["percept_kappa"], kappa_per)
                and np.isclose(meta["option_kappa"], kappa_op)
            ):
                files.append(p)
    return files


# =========================================================
# Omega and flip detection
# =========================================================
def load_positions(path: Path, key="pos") -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    pos = data[key]
    if pos.ndim != 3 or pos.shape[1] != 2:
        raise ValueError(f"Expected pos shape (T,2,N), got {pos.shape}")
    return np.transpose(pos, (0, 2, 1))  # (T,N,2)


def velocities_from_positions(X: np.ndarray, dt=1.0):
    return (X[1:] - X[:-1]) / dt


def normalized_spin_omega_2d(X: np.ndarray, dt=1.0, remove_translation=True):
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
    den = (np.linalg.norm(R, axis=-1) * np.linalg.norm(U, axis=-1)).sum(axis=1) + EPS
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
            s[:nz[0]] = s[nz[0]]
        else:
            s[:] = 1
    return s


def detect_flip_times(omega, smooth_window=21, sign_eps=0.01, min_gap=200):
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


def choose_representative_file(npz_files, key, dt, start_step, smooth_window, sign_eps, min_gap):
    """Choose the first file that has at least one flip after start_step."""
    rows = []
    for p in sorted(npz_files):
        try:
            X = load_positions(p, key=key)
            omega = normalized_spin_omega_2d(X, dt=dt, remove_translation=True)
            ss = min(max(int(start_step), 0), max(len(omega) - 1, 0))
            omega_s_after, flip_local = detect_flip_times(
                omega[ss:], smooth_window=smooth_window, sign_eps=sign_eps, min_gap=min_gap
            )
            n_flips = len(flip_local)
            first_flip = int(flip_local[0] + ss) if n_flips else -1
            rows.append({"npz_path": str(Path(p).resolve()), "n_flips": n_flips, "first_flip": first_flip})
        except Exception as e:
            rows.append({"npz_path": str(Path(p).resolve()), "n_flips": -1, "first_flip": -1, "error": repr(e)})

    df = pd.DataFrame(rows)
    ok = df[df["n_flips"] > 0].copy()
    if ok.empty:
        return Path(sorted(npz_files)[0]).resolve(), df
    return Path(ok.sort_values(["first_flip", "npz_path"]).iloc[0]["npz_path"]).resolve(), df


# =========================================================
# Plot helpers
# =========================================================
def speed_at_t(X: np.ndarray, t: int):
    if t <= 0:
        return np.zeros(X.shape[1], dtype=float)
    d = X[t] - X[t - 1]
    return np.hypot(d[:, 0], d[:, 1])


def add_trails(ax, X, t, trail_len=40, alpha=0.10, lw=0.25):
    t0 = max(0, t - trail_len + 1)
    seg = X[t0:t + 1]
    if seg.shape[0] < 2:
        return
    for i in range(seg.shape[1]):
        ax.plot(seg[:, i, 0], seg[:, i, 1], color="black", alpha=alpha, linewidth=lw, zorder=1)


def axis_limits_from_window(X, t_center, half_window, pad=1.0):
    t0 = max(0, t_center - half_window)
    t1 = min(X.shape[0] - 1, t_center + half_window)
    xx = X[t0:t1 + 1, :, 0]
    yy = X[t0:t1 + 1, :, 1]

    xmin, xmax = float(np.min(xx)), float(np.max(xx))
    ymin, ymax = float(np.min(yy)), float(np.max(yy))
    size = max(xmax - xmin, ymax - ymin, 1e-9)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * size + pad
    return cx - half, cx + half, cy - half, cy + half


def plot_single_omega_flip_figure(
    X,
    omega,
    omega_s,
    flip_t,
    out_pdf,
    out_png,
    panel_offsets=(-120, -60, -20, 0, 20, 60, 120),
    axis_half_window=120,
    point_size=2.5,
    trail_len=40,
    cmap=DEFAULT_CMAP,
):
    """Combined figure: omega trace on top, boxed snapshots below."""
    T = X.shape[0]
    times = [int(np.clip(flip_t + int(dt), 0, T - 1)) for dt in panel_offsets]
    dt_labels = [f"{int(dt):+d}" if int(dt) != 0 else "+0" for dt in panel_offsets]

    x0, x1, y0, y1 = axis_limits_from_window(X, flip_t, axis_half_window, pad=1.0)
    n = len(times)

    fig = plt.figure(figsize=(1.18 * n, 2.25), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[0.75, 1.25])

    # ---------- top: omega trace ----------
    ax_top = fig.add_subplot(gs[0, :])
    t = np.arange(len(omega))
    ax_top.plot(t, omega, color="black", linewidth=0.55, label=r"$\Omega$")
    ax_top.axhline(0.0, color="0.45", linestyle="--", linewidth=0.45)
    ax_top.axvline(flip_t, color="black", linestyle=":", linewidth=0.65)

    for tt in times:
        ax_top.axvline(tt, color="0.65", alpha=0.35, linewidth=0.35)

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
    ax_top.legend(frameon=False, fontsize=4.8, loc="lower right", handlelength=1.6)

    # Keep the top panel boxed lightly.
    for spine in ax_top.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("0.25")

    # ---------- bottom: snapshots ----------
    for j, (tt, dt_lab) in enumerate(zip(times, dt_labels)):
        ax = fig.add_subplot(gs[1, j])
        add_trails(ax, X, tt, trail_len=trail_len, alpha=0.055, lw=0.16)

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
        ax.set_title(f"t = {tt}\n(dt={dt_lab})", fontsize=5.0, pad=1.5)

        # Box each snapshot panel explicitly.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.50)
            spine.set_color("black")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_omega_trace_for_check(omega, omega_s, flip_t, out_png):
    fig, ax = plt.subplots(figsize=(3.2, 1.1), constrained_layout=True)
    t = np.arange(len(omega))
    ax.plot(t, omega, color="0.7", linewidth=0.25)
    ax.plot(t, omega_s, color="black", linewidth=0.5)
    ax.axhline(0, color="0.4", linestyle="--", linewidth=0.4)
    ax.axvline(flip_t, color="black", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel(r"$\Omega$")
    ax.xaxis.set_major_locator(MultipleLocator(25000))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(labelsize=5)
    despine(ax)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Make one paper-style omega-flip figure")
    ap.add_argument("--position-root", required=True, help="root of original position npz files")
    ap.add_argument("--output-root", required=True, help="directory where the figure will be saved")
    ap.add_argument("--key", default="pos")
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--kappa-per", type=float, default=2.5)
    ap.add_argument("--kappa-op", type=float, default=3.0)
    ap.add_argument("--npz", default=None, help="optional explicit npz path")
    ap.add_argument("--event-index", type=int, default=0, help="which detected flip to plot")
    ap.add_argument("--start-step", type=int, default=5000)
    ap.add_argument("--smooth-window", type=int, default=21)
    ap.add_argument("--sign-eps", type=float, default=0.01)
    ap.add_argument("--min-gap", type=int, default=200)
    ap.add_argument("--panel-offsets", default="-120,-60,-20,0,20,60,120")
    ap.add_argument("--axis-half-window", type=int, default=120)
    ap.add_argument("--point-size", type=float, default=3.5)
    ap.add_argument("--trail-len", type=int, default=40)
    args = ap.parse_args()

    set_paper_style()

    position_root = Path(args.position_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)

    if args.npz:
        npz_path = Path(args.npz).expanduser().resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"npz not found: {npz_path}")
        candidates_df = pd.DataFrame([{"npz_path": str(npz_path), "n_flips": np.nan, "first_flip": np.nan}])
    else:
        files = find_npz(position_root, args.N, args.kappa_per, args.kappa_op)
        if not files:
            raise RuntimeError(
                f"No npz found for N={args.N}, kappa_per={args.kappa_per}, "
                f"kappa_op={args.kappa_op} under {position_root}"
            )
        npz_path, candidates_df = choose_representative_file(
            files,
            key=args.key,
            dt=1.0,
            start_step=args.start_step,
            smooth_window=args.smooth_window,
            sign_eps=args.sign_eps,
            min_gap=args.min_gap,
        )

    X = load_positions(npz_path, key=args.key)
    omega = normalized_spin_omega_2d(X, dt=1.0, remove_translation=True)

    start_step = min(max(int(args.start_step), 0), max(len(omega) - 1, 0))
    omega_s_after, flip_local = detect_flip_times(
        omega[start_step:],
        smooth_window=args.smooth_window,
        sign_eps=args.sign_eps,
        min_gap=args.min_gap,
    )
    flip_times = flip_local + start_step
    omega_s = moving_average_reflect(omega, args.smooth_window)

    if len(flip_times) == 0:
        raise RuntimeError(f"No omega flips detected after start_step={start_step} in {npz_path}")
    if args.event_index >= len(flip_times):
        raise RuntimeError(f"event-index={args.event_index} but only {len(flip_times)} flips detected")

    flip_t = int(flip_times[args.event_index])
    panel_offsets = tuple(int(x) for x in args.panel_offsets.split(",") if x.strip())

    out_base = f"Omega_flip_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}"
    out_pdf = output_root / f"{out_base}.pdf"
    out_png = output_root / f"{out_base}.png"

    plot_single_omega_flip_figure(
        X=X,
        omega=omega,
        omega_s=omega_s,
        flip_t=flip_t,
        out_pdf=out_pdf,
        out_png=out_png,
        panel_offsets=panel_offsets,
        axis_half_window=args.axis_half_window,
        point_size=args.point_size,
        trail_len=args.trail_len,
        cmap=DEFAULT_CMAP,
    )

    meta = parse_run_info(str(npz_path))
    selected = {
        "npz_path": str(npz_path),
        "N": args.N,
        "kappa_per": args.kappa_per,
        "kappa_op": args.kappa_op,
        "rep": meta.get("rep"),
        "seed": meta.get("seed"),
        "n_candidates": len(candidates_df),
        "n_flips": len(flip_times),
        "event_index": args.event_index,
        "flip_t": flip_t,
        "panel_offsets": ",".join(map(str, panel_offsets)),
    }
    pd.DataFrame([selected]).to_csv(output_root / "selected_omega_flip_single.csv", index=False)
    candidates_df.to_csv(output_root / "omega_flip_candidate_files.csv", index=False)

    print("[done]")
    print(f"source npz: {npz_path}")
    print(f"flip_t: {flip_t}")
    print(f"figure pdf: {out_pdf}")
    print(f"figure png: {out_png}")
    print(f"catalog: {output_root / 'selected_omega_flip_single.csv'}")


if __name__ == "__main__":
    main()
