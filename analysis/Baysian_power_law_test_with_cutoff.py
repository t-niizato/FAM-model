#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12


# =========================================================
# basic utilities
# =========================================================
def finite_positive(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x) & (x > 0)]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt(v, digits=4):
    try:
        v = float(v)
        if not np.isfinite(v):
            return "nan"
        return f"{v:.{digits}g}"
    except Exception:
        return str(v)


def empirical_ccdf(x):
    x = np.sort(finite_positive(x))
    if x.size == 0:
        return None, None
    y = 1.0 - np.arange(1, x.size + 1) / x.size
    return x, y


# =========================================================
# data loading / geometry
# =========================================================
def load_positions_npz(path, key="pos"):
    data = np.load(path)
    X = data[key]
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (T,2,N), got {X.shape}")
    return X


def center_of_mass_trajectory(X):
    center_x = X[:, 0, :].mean(axis=1)
    center_y = X[:, 1, :].mean(axis=1)
    return center_x, center_y


def center_step_lengths(center_x, center_y):
    return np.hypot(np.diff(center_x), np.diff(center_y))


def relative_positions(X, center_x, center_y):
    rel_x = X[:, 0, :] - center_x[:, None]
    rel_y = X[:, 1, :] - center_y[:, None]
    return rel_x, rel_y


def decompose_rad_tan(relative_x, relative_y):
    rx = relative_x[:-1]
    ry = relative_y[:-1]
    vx = np.diff(relative_x, axis=0)
    vy = np.diff(relative_y, axis=0)

    rnorm = np.hypot(rx, ry)
    inv = 1.0 / np.maximum(rnorm, EPS)

    erx = rx * inv
    ery = ry * inv

    cross = rx * vy - ry * vx
    Lz = cross.mean(axis=1)
    sgn = np.sign(Lz)
    sgn[sgn == 0] = 1.0

    etx = -ery * sgn[:, None]
    ety = erx * sgn[:, None]

    v_rad = vx * erx + vy * ery
    v_tan = vx * etx + vy * ety

    abs_v_rad = np.abs(v_rad)
    abs_v_tan = np.abs(v_tan)
    abs_v = np.hypot(vx, vy)
    return abs_v_rad, abs_v_tan, abs_v, Lz


def polarization_series(X):
    vx = np.diff(X[:, 0, :], axis=0)
    vy = np.diff(X[:, 1, :], axis=0)

    speed = np.hypot(vx, vy)
    ux = vx / np.maximum(speed, EPS)
    uy = vy / np.maximum(speed, EPS)

    px = ux.mean(axis=1)
    py = uy.mean(axis=1)
    return np.hypot(px, py)


# =========================================================
# thresholds / masks / flights
# =========================================================
def make_threshold(series, mode="quantile", q=0.8, value=None):
    s = np.asarray(series, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan
    if mode == "quantile":
        return float(np.quantile(s, q))
    if mode == "value":
        if value is None:
            raise ValueError("mode='value' requires value")
        return float(value)
    raise ValueError(f"unknown mode: {mode}")


def milling_mask_from_Lz(Lz, mode="quantile", q=0.8, value=None):
    absL = np.abs(Lz)
    thr = make_threshold(absL, mode=mode, q=q, value=value)
    if not np.isfinite(thr):
        return np.ones_like(absL, dtype=bool)
    return absL >= thr


def schooling_mask_from_P(P, mode="quantile", q=0.8, value=None):
    thr = make_threshold(P, mode=mode, q=q, value=value)
    if not np.isfinite(thr):
        return np.ones_like(P, dtype=bool)
    return P >= thr


def flight_lengths_from_series(series, threshold):
    flights = []
    s = 0.0
    for v in series:
        if np.isfinite(v) and v >= threshold:
            s += float(v)
        else:
            if s > 0:
                flights.append(s)
                s = 0.0
    if s > 0:
        flights.append(s)
    return np.asarray(flights, float)


# =========================================================
# model curves for overlay
# NOTE:
# truncated: p(x) ∝ x^{-alpha} exp(-x/xc), x >= xmin
# shifted_exp: assumes stored shifted_exp_alpha is lambda
#              so p(x) ∝ exp(-lambda * (x - xmin)), x >= xmin
# If your shifted_exp definition is different, edit only
# ccdf_shifted_exp() below.
# =========================================================
def _normalized_survival_from_pdf_grid(grid, pdf):
    pdf = np.maximum(np.asarray(pdf, float), 0.0)
    area = np.trapz(pdf, grid)
    if not np.isfinite(area) or area <= 0:
        return None

    pdf = pdf / area
    dx = np.diff(grid)

    surv = np.zeros_like(grid)
    for i in range(len(grid) - 2, -1, -1):
        surv[i] = surv[i + 1] + 0.5 * (pdf[i] + pdf[i + 1]) * dx[i]
    return surv


def ccdf_truncated_powerlaw(grid, xmin, alpha, xc):
    grid = np.asarray(grid, float)
    if not (np.isfinite(xmin) and np.isfinite(alpha) and np.isfinite(xc)):
        return None
    if xmin <= 0 or xc <= 0:
        return None

    pdf = np.zeros_like(grid)
    m = grid >= xmin
    pdf[m] = (grid[m] ** (-alpha)) * np.exp(-grid[m] / xc)
    return _normalized_survival_from_pdf_grid(grid, pdf)


def ccdf_shifted_exp(grid, xmin, lam):
    grid = np.asarray(grid, float)
    if not (np.isfinite(xmin) and np.isfinite(lam)):
        return None
    if lam <= 0:
        return None

    pdf = np.zeros_like(grid)
    m = grid >= xmin
    pdf[m] = np.exp(-lam * (grid[m] - xmin))
    return _normalized_survival_from_pdf_grid(grid, pdf)

def _fit_powerlaw_tail(x, xmin):
    x = finite_positive(x)
    x = x[x >= xmin]
    if len(x) < 2:
        raise ValueError("too few tail samples")

    alpha = 1.0 + len(x) / np.sum(np.log(x / xmin))

    loglik = (
        len(x) * np.log(alpha - 1.0)
        + len(x) * (alpha - 1.0) * np.log(xmin)
        - alpha * np.sum(np.log(x))
    )

    k = 1
    aic = 2 * k - 2 * loglik
    bic = k * np.log(len(x)) - 2 * loglik

    def cdf_func(xx):
        return 1.0 - (xx / xmin) ** (1.0 - alpha)

    ks = _ks_distance(x, cdf_func)

    return FitResult(
        model="powerlaw",
        xmin=float(xmin),
        alpha=float(alpha),
        xc=np.nan,
        p_value=np.nan,
        ks_D=ks,
        aic=float(aic),
        bic=float(bic),
        loglik=float(loglik),
        n_used=int(len(x)),
    )
# =========================================================
# plotting helpers
# =========================================================
def save_com_trajectory(cx, cy, out_png, title):
    plt.figure(figsize=(6, 6))
    plt.plot(cx, cy, lw=1.0)
    plt.scatter(cx[0], cy[0], s=30, label="start")
    plt.scatter(cx[-1], cy[-1], s=30, label="end")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_step_series(series, thr, out_png, title):
    plt.figure(figsize=(10, 4))
    plt.plot(series, lw=0.8, label="series")
    if np.isfinite(thr):
        plt.axhline(thr, ls="--", lw=1.0, label=f"threshold={thr:.4g}")
    plt.xlabel("time")
    plt.ylabel("step length")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_ccdf_with_fits(
    flights,
    trunc_params,
    exp_params,
    out_png,
    title,
    extra_text=None,
):
    x_emp, y_emp = empirical_ccdf(flights)
    if x_emp is None:
        return

    plt.figure(figsize=(6, 4))
    plt.loglog(x_emp, y_emp, "o", ms=3, label="empirical")

    xmin = None
    if trunc_params is not None and np.isfinite(trunc_params.get("xmin", np.nan)):
        xmin = float(trunc_params["xmin"])
    elif exp_params is not None and np.isfinite(exp_params.get("xmin", np.nan)):
        xmin = float(exp_params["xmin"])

    if xmin is not None and xmin > 0:
        xmax_plot = max(np.max(x_emp), xmin * 1.2)
        grid = np.logspace(np.log10(xmin), np.log10(xmax_plot), 300)

        idx0 = np.searchsorted(x_emp, xmin, side="left")
        S0 = float(y_emp[idx0] if idx0 < len(y_emp) else y_emp[-1])

        if trunc_params is not None:
            alpha = trunc_params.get("alpha", np.nan)
            xc = trunc_params.get("xc", np.nan)
            y_trunc = ccdf_truncated_powerlaw(grid, xmin, alpha, xc)
            if y_trunc is not None:
                plt.loglog(grid, S0 * y_trunc, "-", lw=2, label="truncated fit")

        if exp_params is not None:
            lam = exp_params.get("alpha", np.nan)
            y_exp = ccdf_shifted_exp(grid, xmin, lam)
            if y_exp is not None:
                plt.loglog(grid, S0 * y_exp, "--", lw=2, label="shifted exp fit")

        plt.axvline(xmin, color="k", ls=":", lw=1, label=f"xmin={xmin:.3g}")

    plt.xlabel("flight length")
    plt.ylabel("P(X ≥ x)")
    plt.title(title)
    plt.legend()

    if extra_text:
        plt.gcf().text(
            0.02, 0.02, extra_text,
            ha="left", va="bottom", fontsize=8, family="monospace"
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_text(text, out_txt):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


# =========================================================
# selection helpers
# =========================================================
def pick_com_row_for_N(df_com, N):
    sub = df_com[df_com["N"] == N].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(
        ["flight_count", "delta_aic_trunc_minus_exp"],
        ascending=[False, True]
    )
    return sub.iloc[0]


def pick_individual_examples_for_N(df_ind, N, kind):
    sub = df_ind[(df_ind["N"] == N) & (df_ind["kind"] == kind)].copy()
    if sub.empty:
        return []

    sub["delta_num"] = pd.to_numeric(sub["delta_aic_trunc_minus_exp"], errors="coerce")
    sub = sub[np.isfinite(sub["delta_num"])].copy()
    if sub.empty:
        return []

    chosen = []

    best_trunc = sub.sort_values(["delta_num", "flight_count"], ascending=[True, False]).iloc[0]
    chosen.append(("best_trunc", best_trunc))

    best_exp = sub.sort_values(["delta_num", "flight_count"], ascending=[False, False]).iloc[0]
    if int(best_exp["individual"]) not in {int(r["individual"]) for _, r in chosen}:
        chosen.append(("best_exp", best_exp))

    sub["abs_delta"] = np.abs(sub["delta_num"])
    middle = sub.sort_values(["abs_delta", "flight_count"], ascending=[True, False]).iloc[0]
    if int(middle["individual"]) not in {int(r["individual"]) for _, r in chosen}:
        chosen.append(("middle", middle))

    return chosen


# =========================================================
# reconstruction
# =========================================================
def reconstruct_individual_series(npz_path, kind, ind, milling_q=0.8, schooling_q=0.8):
    X = load_positions_npz(npz_path)

    cx, cy = center_of_mass_trajectory(X)
    rx, ry = relative_positions(X, cx, cy)
    abs_v_rad, abs_v_tan, abs_v, Lz = decompose_rad_tan(rx, ry)
    P = polarization_series(X)

    milling_mask = milling_mask_from_Lz(Lz, q=milling_q)
    schooling_mask = schooling_mask_from_P(P, q=schooling_q)

    if kind == "milling_rad":
        series = abs_v_rad[milling_mask, ind]
    elif kind == "milling_tan":
        series = abs_v_tan[milling_mask, ind]
    elif kind == "schooling_tot":
        series = abs_v[schooling_mask, ind]
    else:
        raise ValueError(f"unknown kind: {kind}")

    return X, cx, cy, series


# =========================================================
# text helpers
# =========================================================
def params_from_row(row, prefix):
    out = {}
    for k in ["xmin", "alpha", "xc", "p_value", "ks_D", "aic", "bic", "loglik", "n_used"]:
        col = f"{prefix}_{k}"
        out[k] = row[col] if col in row.index else np.nan
    return out


def result_text(trunc_params, exp_params, best_model=None, delta_aic=None, path=None, ind=None):
    lines = []

    if ind is not None:
        lines.append(f"ind       = {ind}")
    if best_model is not None:
        lines.append(f"best      = {best_model}")
    if delta_aic is not None:
        lines.append(f"deltaAIC  = {delta_aic}")
    if path is not None:
        lines.append(f"path      = {Path(path).name}")

    if trunc_params is not None:
        lines.append("")
        lines.append("[truncated]")
        lines.append(f"xmin      = {fmt(trunc_params.get('xmin'))}")
        lines.append(f"alpha     = {fmt(trunc_params.get('alpha'))}")
        lines.append(f"xc        = {fmt(trunc_params.get('xc'))}")
        lines.append(f"p         = {fmt(trunc_params.get('p_value'))}")
        lines.append(f"KS        = {fmt(trunc_params.get('ks_D'))}")
        lines.append(f"AIC       = {fmt(trunc_params.get('aic'))}")

    if exp_params is not None:
        lines.append("")
        lines.append("[shifted_exp]")
        lines.append(f"xmin      = {fmt(exp_params.get('xmin'))}")
        lines.append(f"alpha     = {fmt(exp_params.get('alpha'))}")
        lines.append(f"xc        = {fmt(exp_params.get('xc'))}")
        lines.append(f"p         = {fmt(exp_params.get('p_value'))}")
        lines.append(f"KS        = {fmt(exp_params.get('ks_D'))}")
        lines.append(f"AIC       = {fmt(exp_params.get('aic'))}")

    return "\n".join(lines)


# =========================================================
# plotting cores
# =========================================================
def plot_com_for_N(param_dir, out_dir, N):
    csv_path = param_dir / "levy_com_metrics.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    row = pick_com_row_for_N(df, N)
    if row is None:
        return

    npz_path = Path(row["path"])
    if not npz_path.exists():
        print(f"[skip COM N={N}] npz not found: {npz_path}")
        return

    X = load_positions_npz(npz_path)
    cx, cy = center_of_mass_trajectory(X)
    step = center_step_lengths(cx, cy)

    thr = float(row["threshold"])
    flights = flight_lengths_from_series(step, thr)

    trunc_params = params_from_row(row, "truncated")
    exp_params = params_from_row(row, "shifted_exp")

    ensure_dir(out_dir)

    save_com_trajectory(
        cx, cy,
        out_dir / "COM_trajectory.png",
        title=f"COM trajectory (N={N})"
    )

    save_step_series(
        step, thr,
        out_dir / "COM_step_series.png",
        title=f"COM step series (N={N})"
    )

    txt = result_text(
        trunc_params=trunc_params,
        exp_params=exp_params,
        best_model=row.get("best_model"),
        delta_aic=row.get("delta_aic_trunc_minus_exp"),
        path=npz_path,
    )

    save_ccdf_with_fits(
        flights=flights,
        trunc_params=trunc_params,
        exp_params=exp_params,
        out_png=out_dir / "COM_ccdf_with_fit.png",
        title=f"COM flight CCDF (N={N})",
        extra_text=txt,
    )

    write_text(txt, out_dir / "COM_fitinfo.txt")


def plot_individual_for_N(param_dir, out_dir, N, milling_q=0.8, schooling_q=0.8):
    csv_path = param_dir / "levy_individual_metrics.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    ensure_dir(out_dir)

    for kind in ["milling_rad", "milling_tan", "schooling_tot"]:
        examples = pick_individual_examples_for_N(df, N, kind)
        if not examples:
            continue

        kind_dir = out_dir / kind
        ensure_dir(kind_dir)

        for tag, row in examples:
            npz_path = Path(row["path"])
            ind = int(row["individual"])

            if not npz_path.exists():
                print(f"[skip {kind} N={N}] npz not found: {npz_path}")
                continue

            _, _, _, series = reconstruct_individual_series(
                npz_path=npz_path,
                kind=kind,
                ind=ind,
                milling_q=milling_q,
                schooling_q=schooling_q,
            )

            thr = float(row["threshold"])
            flights = flight_lengths_from_series(series, thr)

            trunc_params = params_from_row(row, "truncated")
            exp_params = params_from_row(row, "shifted_exp")

            stem = f"{tag}_ind{ind:03d}"

            save_step_series(
                series, thr,
                kind_dir / f"{stem}_step_series.png",
                title=f"{kind} {tag} ind={ind} (N={N})"
            )

            txt = result_text(
                trunc_params=trunc_params,
                exp_params=exp_params,
                best_model=row.get("best_model"),
                delta_aic=row.get("delta_aic_trunc_minus_exp"),
                path=npz_path,
                ind=ind,
            )

            save_ccdf_with_fits(
                flights=flights,
                trunc_params=trunc_params,
                exp_params=exp_params,
                out_png=kind_dir / f"{stem}_ccdf_with_fit.png",
                title=f"{kind} {tag} ind={ind} (N={N})",
                extra_text=txt,
            )

            write_text(txt, kind_dir / f"{stem}_fitinfo.txt")

from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class FitResult:
    model: str
    xmin: float
    alpha: float
    xc: float
    p_value: float
    ks_D: float
    aic: float
    bic: float
    loglik: float
    n_used: int

def _empirical_survival_tail(x):
    x = np.sort(finite_positive(x))
    n = len(x)
    y = 1.0 - np.arange(1, n + 1) / n
    y = np.maximum(y, 1.0 / n)
    return x, y

def _ks_distance(x, model_cdf_func):
    x = np.sort(finite_positive(x))
    n = len(x)
    if n == 0:
        return np.nan

    F_mod = np.asarray(model_cdf_func(x), dtype=float)

    i = np.arange(1, n + 1, dtype=float)

    D_plus = np.nanmax(i / n - F_mod)
    D_minus = np.nanmax(F_mod - (i - 1.0) / n)

    return float(max(D_plus, D_minus))

def _fit_shifted_exp_tail(x, xmin):
    x = finite_positive(x)
    x = x[x >= xmin]
    if len(x) < 2:
        raise ValueError("too few tail samples")

    z = x - xmin
    lam = 1.0 / max(np.mean(z), EPS)

    loglik = len(x) * np.log(lam) - lam * np.sum(z)
    k = 1
    aic = 2 * k - 2 * loglik
    bic = k * np.log(len(x)) - 2 * loglik

    def cdf_func(xx):
        return 1.0 - np.exp(-lam * (xx - xmin))

    ks = _ks_distance(x, cdf_func)

    return FitResult(
        model="shifted_exp",
        xmin=float(xmin),
        alpha=float(lam),
        xc=np.nan,
        p_value=np.nan,
        ks_D=ks,
        aic=float(aic),
        bic=float(bic),
        loglik=float(loglik),
        n_used=int(len(x)),
    )

def _fit_truncated_tail(x, xmin):
    x = finite_positive(x)
    x = x[x >= xmin]
    if len(x) < 3:
        raise ValueError("too few tail samples")

    xmax = float(np.max(x))
    if not np.isfinite(xmax) or xmax <= xmin:
        raise ValueError("invalid xmax")

    def norm_const(alpha):
        a1 = 1.0 - alpha
        if abs(a1) > 1e-12:
            return (xmax ** a1 - xmin ** a1) / a1
        return np.log(xmax / xmin)

    def neg_loglik(a):
        alpha = float(a[0])
        if alpha <= 1.0:
            return np.inf
        Z = norm_const(alpha)
        if not np.isfinite(Z) or Z <= 0:
            return np.inf
        ll = -alpha * np.sum(np.log(x)) - len(x) * np.log(Z)
        return -ll

    res = minimize(
        neg_loglik,
        x0=np.array([3.0]),
        bounds=[(1.01, 10.0)],
        method="L-BFGS-B",
    )

    if not res.success:
        raise ValueError(res.message)

    alpha = float(res.x[0])
    loglik = float(-res.fun)

    k = 1
    aic = 2 * k - 2 * loglik
    bic = k * np.log(len(x)) - 2 * loglik

    def cdf_func(xx):
        xx = np.asarray(xx, float)
        out = np.zeros_like(xx)
        m1 = xx < xmin
        m2 = (xx >= xmin) & (xx <= xmax)
        m3 = xx > xmax

        out[m1] = 0.0
        out[m3] = 1.0

        a1 = 1.0 - alpha
        if abs(a1) > 1e-12:
            out[m2] = (xx[m2] ** a1 - xmin ** a1) / (xmax ** a1 - xmin ** a1)
        else:
            out[m2] = np.log(xx[m2] / xmin) / np.log(xmax / xmin)

        return out

    ks = _ks_distance(x, cdf_func)

    return FitResult(
        model="truncated",
        xmin=float(xmin),
        alpha=float(alpha),
        xc=float(xmax),
        p_value=np.nan,
        ks_D=ks,
        aic=float(aic),
        bic=float(bic),
        loglik=float(loglik),
        n_used=int(len(x)),
    )

def choose_xmin_by_ks(
        X,
        n_tail_min,
        model="truncated",
):
    x = np.sort(finite_positive(X))
    N = len(x)

    candidate_idx = np.arange(0, N - n_tail_min + 1)

    ks_vals = np.empty(len(candidate_idx), dtype=float)
    xmin_vals = np.empty(len(candidate_idx), dtype=float)

    best_res = None
    best_ks = np.inf

    for j, idx in enumerate(candidate_idx):
        xmin = float(x[idx])
        xmin_vals[j] = xmin

        try:
            res = gof_with_fixed_xmin(
                x,
                xmin=xmin,
                model=model,
                n_boot=0,
                n_jobs=1,
                random_seed=0,
            )
            ks = res.ks_D
        except Exception:
            ks = np.nan
            res = None

        ks_vals[j] = ks

        if np.isfinite(ks) and ks < best_ks:
            best_ks = ks
            best_res = res

    if best_res is None:
        raise ValueError("no valid xmin candidate")

    return best_res, ks_vals, xmin_vals

def fast_powerlaw_gof(
        X,
        model="truncated",
        n_tail_min=200,
        n_boot=1000,
        n_jobs=1,
        random_seed=0,
):
    X = np.sort(finite_positive(X))

    if len(X) < n_tail_min:
        raise ValueError(
            f"too few samples: n={len(X)}, n_tail_min={n_tail_min}"
        )

    best_res, ks_vals, xmin_vals = choose_xmin_by_ks(
        X,
        n_tail_min=n_tail_min,
        model=model,
    )

    return best_res

def gof_with_fixed_xmin(
        X,
        xmin,
        model="truncated",
        n_boot=1000,
        n_jobs=1,
        random_seed=0,
):
    X = finite_positive(X)

    if model == "truncated":
        return _fit_truncated_tail(X, xmin)

    if model == "shifted_exp":
        return _fit_shifted_exp_tail(X, xmin)

    if model == "powerlaw":
        return _fit_powerlaw_tail(X, xmin)

    raise ValueError(f"unknown model: {model}")

def compare_powerlaw_family(
    X,
    models=("shifted_exp", "truncated"),
    n_tail_min=200,
    n_boot=1000,
    fixed_xmin=None,
    n_jobs=1,
    random_seed=0,
):
    """
    Fit multiple models and return their GOF/AIC results.

    If fixed_xmin is given, all models are fit with that xmin.
    Otherwise each model estimates xmin independently via fast_powerlaw_gof().
    """
    results = {}

    for model in models:
        try:
            if fixed_xmin is not None and np.isfinite(fixed_xmin):
                res = gof_with_fixed_xmin(
                    X,
                    xmin=fixed_xmin,
                    model=model,
                    n_boot=n_boot,
                    n_jobs=n_jobs,
                    random_seed=random_seed,
                )
            else:
                res = fast_powerlaw_gof(
                    X,
                    model=model,
                    n_tail_min=n_tail_min,
                    n_boot=n_boot,
                    n_jobs=n_jobs,
                    random_seed=random_seed,
                )

            results[model] = res

        except Exception as e:
            results[model] = {
                "model": model,
                "error": str(e),
                "aic": np.nan,
                "bic": np.nan,
                "loglik": np.nan,
                "xmin": np.nan,
                "alpha": np.nan,
                "xc": np.nan,
                "p_value": np.nan,
                "ks_D": np.nan,
                "n_used": np.nan,
            }

    return results


def _get_attr_or_key(obj, key, default=np.nan):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def summarize_comparison(results):
    """
    Make a text summary compatible with parse_best_model_from_summary().
    Smaller AIC/BIC is better.
    """
    lines = []
    lines.append("Model comparison")
    lines.append("----------------")

    rows = []
    for model, res in results.items():
        aic = _get_attr_or_key(res, "aic", np.nan)
        bic = _get_attr_or_key(res, "bic", np.nan)
        loglik = _get_attr_or_key(res, "loglik", np.nan)
        xmin = _get_attr_or_key(res, "xmin", np.nan)
        alpha = _get_attr_or_key(res, "alpha", np.nan)
        xc = _get_attr_or_key(res, "xc", np.nan)
        p_value = _get_attr_or_key(res, "p_value", np.nan)
        ks_D = _get_attr_or_key(res, "ks_D", np.nan)
        err = _get_attr_or_key(res, "error", None)

        rows.append((model, aic, bic))

        if err:
            lines.append(f"{model}: ERROR {err}")
        else:
            lines.append(
                f"{model}: "
                f"xmin={xmin:.6g}, alpha={alpha:.6g}, xc={xc:.6g}, "
                f"p={p_value:.6g}, KS={ks_D:.6g}, "
                f"loglik={loglik:.6g}, AIC={aic:.6g}, BIC={bic:.6g}"
            )

    finite_aic = [a for _, a, _ in rows if np.isfinite(a)]
    finite_bic = [b for _, _, b in rows if np.isfinite(b)]

    lines.append("")
    lines.append("ΔAIC / ΔBIC")

    if finite_aic:
        best_aic = min(finite_aic)
    else:
        best_aic = np.nan

    if finite_bic:
        best_bic = min(finite_bic)
    else:
        best_bic = np.nan

    for model, aic, bic in rows:
        daic = aic - best_aic if np.isfinite(aic) and np.isfinite(best_aic) else np.nan
        dbic = bic - best_bic if np.isfinite(bic) and np.isfinite(best_bic) else np.nan
        lines.append(f"{model}: ΔAIC = {daic:.6g} ΔBIC = {dbic:.6g}")

    return "\n".join(lines)
# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        required=True,
        help="path to levy_analysis_by_param",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="directory where plots will be saved",
    )
    parser.add_argument(
        "--key",
        default="pos",
        help="npz key for positions (default: pos)",
    )
    parser.add_argument(
        "--milling-q",
        type=float,
        default=0.80,
        help="same q used in batch for milling mask reconstruction",
    )
    parser.add_argument(
        "--schooling-q",
        type=float,
        default=0.80,
        help="same q used in batch for schooling mask reconstruction",
    )
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not analysis_root.exists():
        raise FileNotFoundError(f"analysis root not found: {analysis_root}")

    ensure_dir(output_root)

    for param_dir in sorted(analysis_root.glob("kappa_*")):
        if not param_dir.is_dir():
            continue

        print(f"[param] {param_dir.name}")

        rep_csv = param_dir / "representative_individual_files_by_N.csv"
        if not rep_csv.exists():
            print("  [skip] representative_individual_files_by_N.csv not found")
            continue

        rep_df = pd.read_csv(rep_csv)
        if rep_df.empty or "N" not in rep_df.columns:
            print("  [skip] representative file table is empty")
            continue

        for N in sorted(rep_df["N"].dropna().astype(int).unique()):
            print(f"  [N={N}]")

            out_base = output_root / param_dir.name / f"N{N:03d}"
            out_com = out_base / "com"
            out_ind = out_base / "individual"

            plot_com_for_N(param_dir, out_com, N)
            plot_individual_for_N(
                param_dir, out_ind, N,
                milling_q=args.milling_q,
                schooling_q=args.schooling_q,
            )

    print("[done]")



if __name__ == "__main__":
    main()