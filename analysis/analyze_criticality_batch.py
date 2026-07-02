#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


# -----------------------------
# 1) Omega(t): normalized angular momentum (2D)
# -----------------------------
def velocities_from_positions(X, dt=1.0):
    return (X[1:] - X[:-1]) / dt  # (T-1,N,2)


def normalized_spin_omega_2d(X, dt=1.0, remove_translation=True):
    V = velocities_from_positions(X, dt=dt)   # (T-1,N,2)
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


# -----------------------------
# 2) Windowed chi = N*Var(x)
# -----------------------------
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


# -----------------------------
# 3) Autocorr (FFT) + tau_int
# -----------------------------
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


# -----------------------------
# 4) Utility
# -----------------------------
def pearson_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a0 = a[m] - a[m].mean()
    b0 = b[m] - b[m].mean()
    return float((a0 * b0).sum() / (np.sqrt((a0 * a0).sum()) * np.sqrt((b0 * b0).sum()) + EPS))


def fit_powerlaw_tau_vs_chi_no_plot(chi, tau):
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


def valid_loglog_mask(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)


def compute_loglog_ranges(x, y):
    m = valid_loglog_mask(x, y)
    if m.sum() < 1:
        return np.nan, np.nan, 0

    lx = np.log10(np.asarray(x, float)[m])
    ly = np.log10(np.asarray(y, float)[m])
    return float(lx.max() - lx.min()), float(ly.max() - ly.min()), int(m.sum())


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

        if s[0] == 0:
            nz = np.flatnonzero(s)
            s[0] = s[nz[0]] if len(nz) else 1
            for i in range(1, len(s)):
                if s[i] == 0:
                    s[i] = s[i - 1]

        nflip = np.sum(s[1:] * s[:-1] < 0)
        centers.append(start + W // 2)
        rates.append(nflip / max(1, (W - 1)))

    return np.asarray(centers), np.asarray(rates)


# -----------------------------
# 5) Path parsing
# -----------------------------
def parse_run_info(path: str) -> dict:
    p = Path(path)
    parts = p.parts

    N = None
    kappa = None
    okappa = None
    rep = None
    seed = None

    for s in parts:
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


# -----------------------------
# 6) One-run analysis
# -----------------------------
def analyze_one_run(
    npz_path: str,
    dt: float,
    W: int,
    step: int,
    cutoff: str,
    flip_quantile: float,
    flip_eps: float,
    min_hi_fit: int = 4,
    min_flip_pos: int = 4,
):
    meta = parse_run_info(npz_path)

    data = np.load(npz_path, allow_pickle=False)
    X = data["pos"]                  # (T,2,N)
    X = np.transpose(X, (0, 2, 1))   # -> (T,N,2)

    T_frames = X.shape[0]
    N = X.shape[1]

    Omega = normalized_spin_omega_2d(X, dt=dt, remove_translation=True)

    if len(Omega) < W:
        return {
            **meta,
            "n_frames": int(T_frames),
            "n_omega": int(len(Omega)),
            "status": "too_short",
        }

    cent_chi, mu_O, var_O, chi_O = windowed_chi(Omega, N=N, W=W, step=step)
    cent_tau, tau_O = windowed_tau_int(Omega, W=W, step=step, cutoff=cutoff)

    if not np.array_equal(cent_chi, cent_tau):
        L = min(len(chi_O), len(tau_O))
        chi_O = chi_O[:L]
        tau_O = tau_O[:L]
        cent_chi = cent_chi[:L]

    cent_fr, flip_rate = windowed_flip_rate(Omega, W=W, step=step, eps=flip_eps)
    if not np.array_equal(cent_fr, cent_chi):
        L = min(len(flip_rate), len(chi_O), len(tau_O))
        flip_rate = flip_rate[:L]
        chi_O = chi_O[:L]
        tau_O = tau_O[:L]
        cent_chi = cent_chi[:L]

    corr_all = pearson_corr(chi_O, tau_O)
    alpha_all, pref_all, r2_all, n_all_fit = fit_powerlaw_tau_vs_chi_no_plot(chi_O, tau_O)

    log10_chi_range, log10_tau_range, n_loglog = compute_loglog_ranges(chi_O, tau_O)

    corr_flip_chi = pearson_corr(flip_rate, chi_O)
    corr_flip_tau = pearson_corr(flip_rate, tau_O)

    m = np.isfinite(flip_rate) & np.isfinite(chi_O) & np.isfinite(tau_O) & (chi_O > 0) & (tau_O > 0)
    fr = flip_rate[m]
    chi = chi_O[m]
    tau = tau_O[m]

    pos = fr > 0

    if len(fr) < 3:
        thr = np.nan
        hi = np.zeros_like(fr, dtype=bool)
        lo = np.zeros_like(fr, dtype=bool)

    elif pos.sum() == 0:
        thr = 0.0
        hi = np.zeros_like(fr, dtype=bool)
        lo = np.ones_like(fr, dtype=bool)

    elif pos.sum() < 3:
        thr = 0.0
        hi = pos.copy()
        lo = ~pos

    else:
        thr_pos = np.quantile(fr[pos], flip_quantile)
        if not np.isfinite(thr_pos):
            thr = 0.0
            hi = pos.copy()
            lo = ~pos
        else:
            thr = float(thr_pos)
            hi = pos & (fr >= thr)
            lo = ~hi

    corr_hi = pearson_corr(chi[hi], tau[hi]) if hi.sum() >= 3 else np.nan
    corr_lo = pearson_corr(chi[lo], tau[lo]) if lo.sum() >= 3 else np.nan

    alpha_hi, pref_hi, r2_hi, n_hi_fit = fit_powerlaw_tau_vs_chi_no_plot(chi[hi], tau[hi])
    alpha_lo, pref_lo, r2_lo, n_lo_fit = fit_powerlaw_tau_vs_chi_no_plot(chi[lo], tau[lo])

    out = {
        **meta,
        "status": "ok",
        "n_frames": int(T_frames),
        "n_omega": int(len(Omega)),
        "n_windows": int(len(chi_O)),

        "Omega_mean": float(np.nanmean(Omega)),
        "Omega_var": float(np.nanvar(Omega)),

        "chi_mean": float(np.nanmean(chi_O)),
        "chi_var": float(np.nanvar(chi_O)),
        "chi_max": float(np.nanmax(chi_O)),

        "tau_mean": float(np.nanmean(tau_O)),
        "tau_var": float(np.nanvar(tau_O)),
        "tau_max": float(np.nanmax(tau_O)),
        "tau_max_over_W": float(np.nanmax(tau_O) / W),

        "corr_chi_tau": corr_all,

        "alpha_all": alpha_all,
        "pref_all": pref_all,
        "r2_all": r2_all,
        "n_all_fit": int(n_all_fit),

        "log10_chi_range": log10_chi_range,
        "log10_tau_range": log10_tau_range,
        "n_loglog": int(n_loglog),

        "corr_flip_chi": corr_flip_chi,
        "corr_flip_tau": corr_flip_tau,

        "flip_mean": float(np.nanmean(flip_rate)),
        "flip_max": float(np.nanmax(flip_rate)),
        "flip_qthr": float(thr) if np.isfinite(thr) else np.nan,

        "n_flip_pos": int(pos.sum()),
        "n_hi": int(hi.sum()),
        "n_lo": int(lo.sum()),

        "corr_hi": corr_hi,
        "corr_lo": corr_lo,

        "alpha_hi": alpha_hi,
        "pref_hi": pref_hi,
        "r2_hi": r2_hi,
        "n_hi_fit": int(n_hi_fit),

        "alpha_lo": alpha_lo,
        "pref_lo": pref_lo,
        "r2_lo": r2_lo,
        "n_lo_fit": int(n_lo_fit),

        "hi_valid": bool((n_hi_fit >= min_hi_fit) and (pos.sum() >= min_flip_pos)),
        "lo_valid": bool(n_lo_fit >= 4),
        "has_switching": bool(pos.sum() > 0),
    }

    series = {
        "Omega": Omega,
        "chi_O": chi_O,
        "tau_O": tau_O,
        "flip_rate": flip_rate,
        "centers": cent_chi,
    }

    return out, series


# -----------------------------
# 7) Plot helpers
# -----------------------------
def plot_loglog_fit(ax, chi, tau, title, xlabel="chi_Omega", ylabel="tau_int_Omega"):
    chi = np.asarray(chi, float)
    tau = np.asarray(tau, float)
    m = valid_loglog_mask(chi, tau)

    ax.scatter(chi[m], tau[m], s=25)

    if m.sum() >= 3:
        alpha, pref, r2, nfit = fit_powerlaw_tau_vs_chi_no_plot(chi[m], tau[m])
        xs = np.logspace(np.log10(np.min(chi[m])), np.log10(np.max(chi[m])), 200)
        ax.plot(xs, pref * (xs ** alpha))
        ax.set_title(f"{title}\nalpha={alpha:.3f}, R^2={r2:.3f}, n={nfit}")
    else:
        ax.set_title(f"{title}\ninsufficient valid points (n={int(m.sum())})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


# -----------------------------
# 8) Resume helper
# -----------------------------
def load_existing_summary(summary_csv: Path):
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if "path" not in df.columns:
                return pd.DataFrame(), set()
            done_paths = set(df["path"].dropna().astype(str).tolist())
            return df, done_paths
        except Exception:
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()


# -----------------------------
# 9) Plots
# -----------------------------
def save_representative_plots(
    df_summary,
    series_dict,
    out_dir,
    flip_quantile=0.80,
    dt=1.0,
    W=2000,
    step=660,
    cutoff="first_nonpositive",
    flip_eps=1e-6,
):
    out_dir = Path(out_dir)
    rep_dir = out_dir / "representative_plots"
    rep_dir.mkdir(parents=True, exist_ok=True)

    ok_df = df_summary[df_summary["status"] == "ok"].copy()
    if ok_df.empty:
        return

    for N, sub in ok_df.groupby("N"):
        sub = sub.sort_values("corr_chi_tau")
        idx = len(sub) // 2
        row = sub.iloc[idx]
        path = str(row["path"])

        if path in series_dict:
            S = series_dict[path]
            Omega = S["Omega"]
            chi_O = S["chi_O"]
            tau_O = S["tau_O"]
            flip_rate = S["flip_rate"]
            centers = S["centers"]
        else:
            try:
                result = analyze_one_run(
                    path,
                    dt=dt,
                    W=W,
                    step=step,
                    cutoff=cutoff,
                    flip_quantile=flip_quantile,
                    flip_eps=flip_eps,
                )
                if not isinstance(result, tuple):
                    continue
                _, S = result
                Omega = S["Omega"]
                chi_O = S["chi_O"]
                tau_O = S["tau_O"]
                flip_rate = S["flip_rate"]
                centers = S["centers"]
            except Exception:
                continue

        fig = plt.figure(figsize=(10, 10))

        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(np.arange(len(Omega)), Omega)
        ax1.set_ylabel("Omega")
        ax1.set_title(
            f"N={N} rep={row['rep']} seed={row['seed']} "
            f"kappa={row['percept_kappa']} okappa={row['option_kappa']}"
        )

        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(centers, chi_O)
        ax2.set_ylabel("chi")

        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(centers, tau_O)
        ax3.set_ylabel("tau_int")

        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(centers, flip_rate)
        ax4.set_ylabel("flip_rate")
        ax4.set_xlabel("time")

        plt.tight_layout()
        fig.savefig(rep_dir / f"representative_timeseries_N{int(N)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.scatter(chi_O, tau_O, s=25)
        plt.xlabel("chi_Omega")
        plt.ylabel("tau_int_Omega")
        plt.title(f"Representative chi vs tau, N={N}")
        plt.grid(alpha=0.3)
        fig.savefig(rep_dir / f"representative_scatter_linear_N{int(N)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_loglog_fit(ax, chi_O, tau_O, title=f"Representative log-log chi vs tau, N={N}")
        fig.tight_layout()
        fig.savefig(rep_dir / f"representative_scatter_loglog_N{int(N)}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        m = valid_loglog_mask(chi_O, tau_O) & np.isfinite(flip_rate)
        if m.sum() >= 1:
            fr = flip_rate[m]
            chi = chi_O[m]
            tau = tau_O[m]
            thr = np.quantile(fr, flip_quantile) if len(fr) >= 3 else np.nan
            hi = fr >= thr if np.isfinite(thr) else np.zeros_like(fr, dtype=bool)
            lo = ~hi if np.isfinite(thr) else np.ones_like(fr, dtype=bool)

            fig = plt.figure()
            if lo.sum() > 0:
                plt.scatter(chi[lo], tau[lo], s=25, label="low flip")
            if hi.sum() > 0:
                plt.scatter(chi[hi], tau[hi], s=25, label="high flip")
            plt.xlabel("chi_Omega")
            plt.ylabel("tau_int_Omega")
            plt.title(f"chi vs tau colored by flip, N={N}")
            plt.legend()
            plt.grid(alpha=0.3)
            fig.savefig(rep_dir / f"representative_scatter_flipsplit_N{int(N)}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def save_summary_plots(df_summary, out_dir, min_hi_fit=4, min_flip_pos=4):
    out_dir = Path(out_dir)
    plot_dir = out_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ok_all = df_summary[df_summary["status"] == "ok"].copy()
    if ok_all.empty:
        return

    metrics_all = [
        "corr_chi_tau",
        "alpha_all",
        "r2_all",
        "log10_chi_range",
        "log10_tau_range",
        "n_loglog",
        "chi_mean",
        "tau_mean",
        "flip_mean",
        "corr_flip_chi",
        "corr_flip_tau",
        "n_flip_pos",
        "n_hi_fit",
        "n_lo_fit",
    ]

    g_all = ok_all.groupby("N")

    summary_rows = []
    for metric in metrics_all:
        if metric not in ok_all.columns:
            continue

        tmp = g_all[metric].agg(["mean", "std", "count"]).reset_index()
        tmp["metric"] = metric
        summary_rows.append(tmp)

        fig = plt.figure()
        plt.errorbar(tmp["N"], tmp["mean"], yerr=tmp["std"], fmt="o-", capsize=4)
        plt.xlabel("N")
        plt.ylabel(metric)
        plt.title(f"{metric} vs N")
        plt.grid(alpha=0.3)
        fig.savefig(plot_dir / f"{metric}_vs_N.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    ok_hi = ok_all[
        (pd.to_numeric(ok_all["n_hi_fit"], errors="coerce") >= min_hi_fit) &
        (pd.to_numeric(ok_all["n_flip_pos"], errors="coerce") >= min_flip_pos)
    ].copy()

    ok_lo = ok_all[
        pd.to_numeric(ok_all["n_lo_fit"], errors="coerce") >= 4
    ].copy()

    metrics_hi = ["alpha_hi", "corr_hi", "r2_hi"]
    metrics_lo = ["alpha_lo", "corr_lo", "r2_lo"]

    if not ok_hi.empty:
        g_hi = ok_hi.groupby("N")
        for metric in metrics_hi:
            if metric not in ok_hi.columns:
                continue

            tmp = g_hi[metric].agg(["mean", "std", "count"]).reset_index()
            tmp["metric"] = metric
            summary_rows.append(tmp)

            fig = plt.figure()
            plt.errorbar(tmp["N"], tmp["mean"], yerr=tmp["std"], fmt="o-", capsize=4)
            plt.xlabel("N")
            plt.ylabel(metric)
            plt.title(f"{metric} vs N (filtered)")
            plt.grid(alpha=0.3)
            fig.savefig(plot_dir / f"{metric}_vs_N_filtered.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    if not ok_lo.empty:
        g_lo = ok_lo.groupby("N")
        for metric in metrics_lo:
            if metric not in ok_lo.columns:
                continue

            tmp = g_lo[metric].agg(["mean", "std", "count"]).reset_index()
            tmp["metric"] = metric
            summary_rows.append(tmp)

            fig = plt.figure()
            plt.errorbar(tmp["N"], tmp["mean"], yerr=tmp["std"], fmt="o-", capsize=4)
            plt.xlabel("N")
            plt.ylabel(metric)
            plt.title(f"{metric} vs N (filtered)")
            plt.grid(alpha=0.3)
            fig.savefig(plot_dir / f"{metric}_vs_N_filtered.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    if summary_rows:
        df_metric_summary = pd.concat(summary_rows, ignore_index=True)
        df_metric_summary.to_csv(out_dir / "criticality_by_N.csv", index=False)

    avail = ok_all.groupby("N").agg(
        n_runs=("path", "count"),
        n_switch_runs=("has_switching", "sum"),
        mean_n_flip_pos=("n_flip_pos", "mean"),
        mean_n_hi_fit=("n_hi_fit", "mean"),
    ).reset_index()

    if "n_runs" in avail.columns and len(avail) > 0:
        avail["switch_fraction"] = avail["n_switch_runs"] / avail["n_runs"]

        fig = plt.figure()
        plt.plot(avail["N"], avail["switch_fraction"], "o-")
        plt.xlabel("N")
        plt.ylabel("fraction of runs with switching")
        plt.title("Switching availability vs N")
        plt.grid(alpha=0.3)
        fig.savefig(plot_dir / "switch_fraction_vs_N.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(avail["N"], avail["mean_n_flip_pos"], "o-")
        plt.xlabel("N")
        plt.ylabel("mean n_flip_pos")
        plt.title("Mean number of positive-flip windows vs N")
        plt.grid(alpha=0.3)
        fig.savefig(plot_dir / "mean_n_flip_pos_vs_N.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(avail["N"], avail["mean_n_hi_fit"], "o-")
        plt.xlabel("N")
        plt.ylabel("mean n_hi_fit")
        plt.title("Mean n_hi_fit vs N")
        plt.grid(alpha=0.3)
        fig.savefig(plot_dir / "mean_n_hi_fit_vs_N.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        avail.to_csv(out_dir / "switching_availability_by_N.csv", index=False)

    valid_hi = ok_all.assign(
        hi_ok=(
            (pd.to_numeric(ok_all["n_hi_fit"], errors="coerce") >= min_hi_fit) &
            (pd.to_numeric(ok_all["n_flip_pos"], errors="coerce") >= min_flip_pos)
        )
    ).groupby("N").agg(
        n_runs=("path", "count"),
        n_valid_hi=("hi_ok", "sum"),
    ).reset_index()

    if len(valid_hi) > 0:
        valid_hi["valid_hi_fraction"] = valid_hi["n_valid_hi"] / valid_hi["n_runs"]

        fig = plt.figure()
        plt.plot(valid_hi["N"], valid_hi["valid_hi_fraction"], "o-")
        plt.xlabel("N")
        plt.ylabel("valid hi fraction")
        plt.title("Fraction of runs with usable high-flip fit")
        plt.grid(alpha=0.3)
        fig.savefig(plot_dir / "valid_hi_fraction_vs_N.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        valid_hi.to_csv(out_dir / "valid_hi_fraction_by_N.csv", index=False)

    if not ok_hi.empty and not ok_lo.empty:
        comp_hi = ok_hi.groupby("N")[["alpha_hi", "corr_hi"]].mean().reset_index()
        comp_lo = ok_lo.groupby("N")[["alpha_lo", "corr_lo"]].mean().reset_index()

        comp = pd.merge(comp_hi, comp_lo, on="N", how="outer")

        if {"alpha_hi", "alpha_lo"}.issubset(comp.columns):
            fig = plt.figure()
            plt.plot(comp["N"], comp["alpha_hi"], "o-", label="alpha_hi")
            plt.plot(comp["N"], comp["alpha_lo"], "o-", label="alpha_lo")
            plt.xlabel("N")
            plt.ylabel("alpha")
            plt.title("High-flip vs Low-flip alpha")
            plt.legend()
            plt.grid(alpha=0.3)
            fig.savefig(plot_dir / "alpha_hi_lo_vs_N.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        if {"corr_hi", "corr_lo"}.issubset(comp.columns):
            fig = plt.figure()
            plt.plot(comp["N"], comp["corr_hi"], "o-", label="corr_hi")
            plt.plot(comp["N"], comp["corr_lo"], "o-", label="corr_lo")
            plt.xlabel("N")
            plt.ylabel("corr(chi,tau)")
            plt.title("High-flip vs Low-flip correlation")
            plt.legend()
            plt.grid(alpha=0.3)
            fig.savefig(plot_dir / "corr_hi_lo_vs_N.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    for metric in [
        "corr_chi_tau",
        "alpha_all",
        "tau_mean",
        "chi_mean",
        "log10_chi_range",
        "log10_tau_range",
        "corr_flip_chi",
        "corr_flip_tau",
        "n_flip_pos",
        "n_hi_fit",
    ]:
        if metric not in ok_all.columns:
            continue

        fig = plt.figure()
        groups = []
        labels = []
        for N, sub in ok_all.groupby("N"):
            vals = sub[metric].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(str(int(N)))
        if groups:
            plt.boxplot(groups, labels=labels)
            plt.xlabel("N")
            plt.ylabel(metric)
            plt.title(f"{metric} by N")
            plt.grid(alpha=0.3)
            fig.savefig(plot_dir / f"boxplot_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# -----------------------------
# 10) Group helpers
# -----------------------------
def fmt_float_for_dir(x: float, nd: int = 6) -> str:
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def collect_npz_files(root: Path):
    return sorted(glob.glob(str(root / "N*" / "kappa_*" / "okappa_*" / "pos_rep*_seed*.npz")))


def group_files_by_param(files):
    groups = {}
    for f in files:
        meta = parse_run_info(f)
        key = (meta["percept_kappa"], meta["option_kappa"])
        groups.setdefault(key, []).append(f)
    return groups


def analyze_file_list(files, out_dir, args):
    out_dir = Path(out_dir)
    rows = []
    series_dict = {}

    summary_csv = out_dir / "criticality_summary.csv"
    existing_df, done_paths = load_existing_summary(summary_csv)

    print(f"[group out] {out_dir}")
    print(f"[group n_files] {len(files)}")

    if args.force:
        files_todo = list(files)
        print("[mode] force recompute all matching files")
    elif args.resume:
        files_todo = [f for f in files if str(f) not in done_paths]
        print("[mode] resume")
        print(f"[resume check] done={len(files) - len(files_todo)} todo={len(files_todo)} total={len(files)}")
    else:
        files_todo = list(files)
        print("[mode] no-resume (recompute matching files in current selection)")

    for i, f in enumerate(files_todo, start=1):
        print(f"[{i}/{len(files_todo)}] {f}")
        try:
            result = analyze_one_run(
                f,
                dt=args.dt,
                W=args.W,
                step=args.step,
                cutoff=args.cutoff,
                flip_quantile=args.flip_quantile,
                flip_eps=args.flip_eps,
                min_hi_fit=args.min_hi_fit,
                min_flip_pos=args.min_flip_pos,
            )

            if isinstance(result, tuple):
                row, series = result
                rows.append(row)
                series_dict[str(f)] = series
            else:
                rows.append(result)

        except Exception as e:
            meta = parse_run_info(f)
            rows.append({
                **meta,
                "status": "error",
                "error": repr(e),
            })
            print(f"  ERROR: {e}")

    df_new = pd.DataFrame(rows)

    if existing_df.empty:
        df = df_new.copy()
    else:
        df_old = existing_df.copy()

        if args.force or (not args.resume):
            target_paths = set(str(f) for f in files)
            df_old = df_old[~df_old["path"].astype(str).isin(target_paths)]

        df = pd.concat([df_old, df_new], ignore_index=True)

    if not df.empty and "path" in df.columns:
        df["path"] = df["path"].astype(str)
        df = df.drop_duplicates(subset=["path"], keep="last")

    sort_cols = [c for c in ["N", "percept_kappa", "option_kappa", "rep", "seed", "path"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(summary_csv, index=False)
    print(f"[saved] {summary_csv}")

    if "status" in df.columns:
        print(df["status"].value_counts(dropna=False))

    save_representative_plots(
        df,
        series_dict,
        out_dir,
        flip_quantile=args.flip_quantile,
        dt=args.dt,
        W=args.W,
        step=args.step,
        cutoff=args.cutoff,
        flip_eps=args.flip_eps,
    )
    save_summary_plots(
        df,
        out_dir,
        min_hi_fit=args.min_hi_fit,
        min_flip_pos=args.min_flip_pos,
    )


# -----------------------------
# 11) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True,
                    help="root directory of batch_record_positions output")
    ap.add_argument("--out", type=str, default=None,
                    help="output analysis directory (default: <root>/criticality_analysis_by_param)")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--W", type=int, default=2000)
    ap.add_argument("--step", type=int, default=660)
    ap.add_argument("--cutoff", type=str, default="first_nonpositive",
                    choices=["first_nonpositive", "full"])
    ap.add_argument("--flip-quantile", type=float, default=0.85)
    ap.add_argument("--flip-eps", type=float, default=1e-6)
    ap.add_argument("--max-files", type=int, default=None,
                    help="debug用。先頭から何ファイルだけ解析するか")
    ap.add_argument("--percept-kappa", type=float, default=None,
                    help="この値だけ解析したい場合に指定")
    ap.add_argument("--option-kappa", type=float, default=None,
                    help="この値だけ解析したい場合に指定")

    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip files already present in criticality_summary.csv")
    ap.add_argument("--no-resume", action="store_false", dest="resume",
                    help="do not use existing summary for skipping")
    ap.add_argument("--force", action="store_true",
                    help="recompute all matching files and overwrite existing rows")

    ap.add_argument("--min-hi-fit", type=int, default=4,
                    help="minimum n_hi_fit required to use alpha_hi / corr_hi in summary plots")
    ap.add_argument("--min-flip-pos", type=int, default=4,
                    help="minimum n_flip_pos required to regard high-flip split as meaningful")

    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    if args.out is None:
        out_root = root / "criticality_analysis_by_param"
    else:
        out_root = Path(os.path.expanduser(args.out)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = collect_npz_files(root)
    if args.max_files is not None:
        files = files[:args.max_files]

    print(f"[root] {root}")
    print(f"[analysis_root] {out_root}")
    print(f"[n_files_total] {len(files)}")

    if len(files) == 0:
        print("[done] no files found.")
        return

    groups = group_files_by_param(files)

    if args.percept_kappa is not None:
        groups = {
            k: v for k, v in groups.items()
            if np.isclose(k[0], args.percept_kappa)
        }

    if args.option_kappa is not None:
        groups = {
            k: v for k, v in groups.items()
            if np.isclose(k[1], args.option_kappa)
        }

    if len(groups) == 0:
        print("[done] no matching parameter groups.")
        return

    index_rows = []

    for (pk, ok), group_files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        group_name = f"kappa_{fmt_float_for_dir(pk)}__okappa_{fmt_float_for_dir(ok)}"
        group_out = out_root / group_name
        group_out.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"[group] percept_kappa={pk}, option_kappa={ok}")
        print(f"[group_name] {group_name}")

        analyze_file_list(group_files, group_out, args)

        index_rows.append({
            "group_name": group_name,
            "percept_kappa": pk,
            "option_kappa": ok,
            "n_files": len(group_files),
            "out_dir": str(group_out),
        })

    df_index = pd.DataFrame(index_rows)
    index_csv = out_root / "group_index.csv"
    df_index.to_csv(index_csv, index=False)
    print(f"[saved] {index_csv}")
    print("[done]")


if __name__ == "__main__":
    main()