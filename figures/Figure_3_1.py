#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure_chi_tau_timeseries_N300_kper1p5_kop2.py

Paper-style figures for one representative run at:
    N = 300, kappa_per = 1.5, kappa_op = 2.0

Outputs:
    Figure_chi_tau_loglog_N300_kper1.5_kop2.pdf/png
    Figure_chi_tau_timeseries_N300_kper1.5_kop2.pdf/png
    Figure_chi_tau_N300_kper1.5_kop2_data.csv
    Figure_chi_tau_N300_kper1.5_kop2_selected.csv

The computations follow the original criticality script:
    Omega(t)
    windowed chi = N * Var(Omega)
    windowed tau_int from autocorrelation
    windowed flip rate
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator

try:
    import seaborn as sns
    sns.set_theme(style="ticks", context="paper")
except Exception:
    sns = None

EPS = 1e-12

DEFAULT_N = 300
DEFAULT_KPER = 1.5
DEFAULT_KOP = 2.0


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
    })


def despine(ax):
    if sns is not None:
        sns.despine(ax=ax)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# =========================================================
# Core computations
# =========================================================
def velocities_from_positions(X, dt=1.0):
    return (X[1:] - X[:-1]) / dt


def normalized_spin_omega_2d(X, dt=1.0, remove_translation=True):
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


def windowed_chi(x, N, W, step):
    T = len(x)
    centers, mu, var, chi = [], [], [], []
    for start in range(0, T - W + 1, step):
        seg = x[start:start + W]
        centers.append(start + W // 2)
        m = seg.mean()
        v = seg.var(ddof=0)
        mu.append(m)
        var.append(v)
        chi.append(N * v)
    return np.asarray(centers), np.asarray(mu), np.asarray(var), np.asarray(chi)


def autocorr_fft(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    return ac / (ac[0] + EPS)


def tau_int_from_ac(ac, cutoff="first_nonpositive"):
    if len(ac) < 2:
        return 0.0
    if cutoff == "first_nonpositive":
        tcut = len(ac) - 1
        for t in range(1, len(ac)):
            if ac[t] <= 0:
                tcut = t - 1
                break
        s = ac[1:tcut + 1].sum() if tcut >= 1 else 0.0
        return float(1.0 + 2.0 * s)
    if cutoff == "full":
        return float(1.0 + 2.0 * ac[1:].sum())
    raise ValueError("unknown cutoff")


def windowed_tau_int(x, W, step, cutoff="first_nonpositive"):
    T = len(x)
    centers, taus = [], []
    for start in range(0, T - W + 1, step):
        seg = x[start:start + W]
        ac = autocorr_fft(seg)
        tau = tau_int_from_ac(ac, cutoff=cutoff)
        centers.append(start + W // 2)
        taus.append(tau)
    return np.asarray(centers), np.asarray(taus)


def windowed_flip_rate(x, W, step, eps=1e-8):
    x = np.asarray(x, float)
    T = len(x)
    centers, rates = [], []
    for start in range(0, T - W + 1, step):
        seg = x[start:start + W]
        s = np.sign(seg)
        s[np.abs(seg) < eps] = 0
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i - 1]
        if len(s) and s[0] == 0:
            nz = np.flatnonzero(s)
            s[0] = s[nz[0]] if len(nz) else 1
            for i in range(1, len(s)):
                if s[i] == 0:
                    s[i] = s[i - 1]
        nflip = np.sum(s[1:] * s[:-1] < 0)
        centers.append(start + W // 2)
        rates.append(nflip / max(1, (W - 1)))
    return np.asarray(centers), np.asarray(rates)


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


# =========================================================
# Paths
# =========================================================
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
    return {"path": str(path), "N": N, "percept_kappa": kappa, "option_kappa": okappa, "rep": rep, "seed": seed}


def find_target_npz(root: Path, N: int, kper: float, kop: float):
    target = root / f"N{N}" / f"kappa_{fmt_float_for_dir(kper)}" / f"okappa_{fmt_float_for_dir(kop)}"
    files = sorted(target.glob("pos_rep*_seed*.npz"))
    if files:
        return files
    return sorted(root.glob(f"N{N}/kappa_*/okappa_*/pos_rep*_seed*.npz"))


def choose_representative_file(files):
    metas = [parse_run_info(str(f)) for f in files]
    valid = [(m["rep"], f) for m, f in zip(metas, files) if m["rep"] is not None]
    if not valid:
        return sorted(files)[0]
    valid.sort(key=lambda t: t[0])
    return Path(valid[len(valid) // 2][1]).resolve()


# =========================================================
# Analysis and caching
# =========================================================
def analyze_file(npz_path: Path, W: int, step: int, dt: float, cutoff: str, flip_eps: float):
    data = np.load(npz_path, allow_pickle=False)
    X = data["pos"]
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected pos shape (T,2,N), got {X.shape}")
    X = np.transpose(X, (0, 2, 1))
    N = X.shape[1]
    omega = normalized_spin_omega_2d(X, dt=dt, remove_translation=True)
    centers, mu, var, chi = windowed_chi(omega, N=N, W=W, step=step)
    centers_tau, tau = windowed_tau_int(omega, W=W, step=step, cutoff=cutoff)
    centers_fr, flip_rate = windowed_flip_rate(omega, W=W, step=step, eps=flip_eps)
    L = min(len(centers), len(centers_tau), len(centers_fr), len(chi), len(tau), len(flip_rate))
    return omega, pd.DataFrame({
        "center": centers[:L],
        "chi": chi[:L],
        "tau_int": tau[:L],
        "flip_rate": flip_rate[:L],
        "omega_mean_window": mu[:L],
        "omega_var_window": var[:L],
    })


# =========================================================
# Plotting
# =========================================================
def save_loglog_figure(df: pd.DataFrame, out_pdf: Path, out_png: Path):
    chi = df["chi"].to_numpy(float)
    tau = df["tau_int"].to_numpy(float)
    m = np.isfinite(chi) & np.isfinite(tau) & (chi > 0) & (tau > 0)

    if m.sum() < 3:
        raise RuntimeError("Not enough valid chi-tau points.")

    x = chi[m]
    y = tau[m]

    alpha, pref, r2, nfit = fit_powerlaw_tau_vs_chi(chi, tau)

    fig, ax = plt.subplots(figsize=(2.65, 2.25), constrained_layout=True)

    ax.scatter(
        x, y,
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
            0.04, 0.96,
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

    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    despine(ax)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_timeseries_figure(omega: np.ndarray, df: pd.DataFrame,
                           out_pdf: Path, out_png: Path):

    centers = df["center"].to_numpy(float)
    chi = df["chi"].to_numpy(float)
    tau = df["tau_int"].to_numpy(float)
    flip_rate = df["flip_rate"].to_numpy(float)

    omega_color = "#2F4F6F"
    stat_color  = "#4E79A7"
    tau_color   = "#5B5F97"
    event_color = "#2B2B2B"

    fig, axes = plt.subplots(
        4, 1,
        figsize=(3.35, 3.0),
        sharex=True,
        constrained_layout=True
    )

    axes[0].plot(np.arange(len(omega)), omega,
                 color=omega_color, linewidth=0.35)
    axes[0].axhline(0, color="0.55", linestyle="--", linewidth=0.35)
    axes[0].set_ylabel(r"$\Omega$")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.5))

    axes[1].plot(centers, chi,
                 color=stat_color, linewidth=0.45)
    axes[1].set_ylabel(r"$\chi_\Omega$")

    axes[2].plot(centers, tau,
                 color=tau_color, linewidth=0.45)
    axes[2].set_ylabel(r"$\tau_{\mathrm{int}}$")

    axes[3].step(centers, flip_rate,
                 where="mid", color=event_color, linewidth=0.45)
    axes[3].set_ylabel("Flip\nrate")
    axes[3].set_xlabel("Time step")

    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(20000))
        ax.xaxis.set_minor_locator(MultipleLocator(10000))

    # y 軸
    axes[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.25))

    axes[1].yaxis.set_major_locator(MultipleLocator(50))
    axes[1].yaxis.set_minor_locator(MultipleLocator(25))

    axes[2].yaxis.set_major_locator(MultipleLocator(200))
    axes[2].yaxis.set_minor_locator(MultipleLocator(100))

    axes[3].yaxis.set_major_locator(MultipleLocator(0.001))
    axes[3].yaxis.set_minor_locator(MultipleLocator(0.0005))

    for ax in axes:
        ax.grid(which="major", color="0.88", linewidth=0.28)
        ax.grid(which="minor", color="0.94", linewidth=0.20)

        ax.tick_params(
            which="major",
            labelsize=6,
            width=0.55,
            length=3.0,
            pad=1.5
        )
        ax.tick_params(
            which="minor",
            width=0.45,
            length=1.7
        )

        sns.despine(ax=ax)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Make paper-style chi/tau figures for N300, kper1.5, kop2.0")
    ap.add_argument("--root", required=True, help="root directory of position records")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--N", type=int, default=DEFAULT_N)
    ap.add_argument("--kappa-per", type=float, default=DEFAULT_KPER)
    ap.add_argument("--kappa-op", type=float, default=DEFAULT_KOP)
    ap.add_argument("--npz", default=None, help="optional explicit npz path")
    ap.add_argument("--W", type=int, default=2000)
    ap.add_argument("--step", type=int, default=660)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--cutoff", default="first_nonpositive", choices=["first_nonpositive", "full"])
    ap.add_argument("--flip-eps", type=float, default=1e-6)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    set_paper_style()
    root = Path(os.path.expanduser(args.root)).resolve()
    out = Path(os.path.expanduser(args.out)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    out_base = f"Figure_chi_tau_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}"
    data_csv = out / f"{out_base}_data.csv"
    omega_npy = out / f"{out_base}_omega.npy"
    selected_csv = out / f"{out_base}_selected.csv"

    if args.npz:
        npz_path = Path(os.path.expanduser(args.npz)).resolve()
    else:
        files = find_target_npz(root, args.N, args.kappa_per, args.kappa_op)
        files = [f for f in files if np.isclose(parse_run_info(str(f))["percept_kappa"], args.kappa_per) and np.isclose(parse_run_info(str(f))["option_kappa"], args.kappa_op)]
        if not files:
            raise RuntimeError(f"No files found for N={args.N}, kper={args.kappa_per}, kop={args.kappa_op} under {root}")
        npz_path = choose_representative_file(files)

    if data_csv.exists() and omega_npy.exists() and not args.recompute:
        df = pd.read_csv(data_csv)
        omega = np.load(omega_npy)
    else:
        omega, df = analyze_file(npz_path, W=args.W, step=args.step, dt=args.dt, cutoff=args.cutoff, flip_eps=args.flip_eps)
        df.to_csv(data_csv, index=False)
        np.save(omega_npy, omega)

    meta = parse_run_info(str(npz_path))
    pd.DataFrame([{
        **meta,
        "W": args.W,
        "step": args.step,
        "cutoff": args.cutoff,
        "data_csv": str(data_csv),
        "omega_npy": str(omega_npy),
    }]).to_csv(selected_csv, index=False)

    save_loglog_figure(
        df,
        out / f"Figure_chi_tau_loglog_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}.pdf",
        out / f"Figure_chi_tau_loglog_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}.png",
    )
    save_timeseries_figure(
        omega,
        df,
        out / f"Figure_chi_tau_timeseries_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}.pdf",
        out / f"Figure_chi_tau_timeseries_N{args.N}_kper{args.kappa_per:g}_kop{args.kappa_op:g}.png",
    )

    print("[done]")
    print(f"source npz: {npz_path}")
    print(f"data csv: {data_csv}")
    print(f"selected csv: {selected_csv}")


if __name__ == "__main__":
    main()
