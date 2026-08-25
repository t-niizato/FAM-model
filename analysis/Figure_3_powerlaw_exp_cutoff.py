#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_3_powerlaw_exp_cutoff.py

Compare a power law with exponential cutoff against a shifted exponential
for flip intervals in Figure 3.

Models
------
Power law with exponential cutoff:
    p(x) ∝ x^{-alpha} exp(-x / xc),  x >= xmin

Shifted exponential:
    p(x) = lambda exp[-lambda (x - xmin)],  x >= xmin

The same fixed xmin is used for all conditions and for both models.

Input:
    data/processed/figure3/Figure_flip_interval_survival_data.csv

Outputs:
    data/processed/figure3/Figure_3_powerlaw_exp_cutoff.csv
    data/processed/figure3/Figure_3_powerlaw_exp_cutoff.txt
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize


EPS = 1e-12

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "figure3"
    / "Figure_flip_interval_survival_data.csv"
)

OUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "figure3"
    / "Figure_3_powerlaw_exp_cutoff.csv"
)

OUT_TXT = (
    ROOT
    / "data"
    / "processed"
    / "figure3"
    / "Figure_3_powerlaw_exp_cutoff.txt"
)


CONDITIONS = [
    ("I",   "Milling phase", 2.5, 3.0),
    ("III", "MS phase",      1.5, 2.0),
    ("IV",  "MS phase",      2.5, 1.8),
    ("V",   "SMS phase",     0.5, 1.5),
]


# Common lower cutoff used for all conditions.
FIXED_XMIN = 1.0


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


def finite_positive(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


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


# ============================================================
# Shifted exponential
# ============================================================

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
        xx = np.asarray(xx, dtype=float)
        out = np.zeros_like(xx)

        m = xx >= xmin
        out[m] = 1.0 - np.exp(-lam * (xx[m] - xmin))

        return out

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


# ============================================================
# Power law with exponential cutoff
#
# p(x) ∝ x^{-alpha} exp(-x / xc),  x >= xmin
# ============================================================

def _cutoff_norm(alpha, xc, xmin):
    """
    Stable normalization for

        p(x) ∝ x^{-alpha} exp(-x / xc), x >= xmin.

    Use the change of variable x = exp(u):

        Z = ∫ exp[(1-alpha)u - exp(u)/xc] du

    which is much more stable than integrating directly over x when
    alpha ≈ 1 and xc is large.
    """
    if not np.isfinite(alpha) or not np.isfinite(xc):
        return np.nan
    if xmin <= 0 or xc <= 0:
        return np.nan

    umin = np.log(xmin)

    # exp(-x/xc) is negligible beyond roughly 60*xc.
    xmax_num = max(xmin * 10.0, 60.0 * xc)
    umax = np.log(xmax_num)

    def integrand(u):
        return np.exp((1.0 - alpha) * u - np.exp(u) / xc)

    value, _ = quad(
        integrand,
        umin,
        umax,
        epsabs=1e-10,
        epsrel=1e-9,
        limit=300,
    )

    return value


def _fit_powerlaw_exp_cutoff_tail(x, xmin):
    x = finite_positive(x)
    x = x[x >= xmin]

    if len(x) < 3:
        raise ValueError("too few tail samples")

    xmax_obs = float(np.max(x))
    mean_x = float(np.mean(x))

    def neg_loglik(theta):
        alpha = float(theta[0])
        log_xc = float(theta[1])
        xc = float(np.exp(log_xc))

        if not np.isfinite(alpha) or not np.isfinite(xc) or xc <= 0:
            return np.inf

        Z = _cutoff_norm(
            alpha=alpha,
            xc=xc,
            xmin=xmin,
        )

        if not np.isfinite(Z) or Z <= 0:
            return np.inf

        ll = (
            -alpha * np.sum(np.log(x))
            - np.sum(x) / xc
            - len(x) * np.log(Z)
        )

        return -ll

    # Several initial points make the fit more robust.
    initial_guesses = [
        np.array([0.5, np.log(max(mean_x, xmin))]),
        np.array([1.0, np.log(max(mean_x, xmin))]),
        np.array([1.5, np.log(max(mean_x, xmin))]),
        np.array([0.5, np.log(max(xmax_obs / 2.0, xmin))]),
    ]

    best = None

    bounds = [
        (-5.0, 10.0),
        (
            np.log(max(xmin * 0.01, EPS)),
            np.log(max(xmax_obs * 100.0, xmin * 10.0)),
        ),
    ]

    for x0 in initial_guesses:
        res = minimize(
            neg_loglik,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )

        if not res.success:
            continue

        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("cutoff power-law fit failed")

    alpha = float(best.x[0])
    xc = float(np.exp(best.x[1]))
    loglik = float(-best.fun)

    # Two fitted parameters: alpha and xc.
    k = 2
    aic = 2 * k - 2 * loglik
    bic = k * np.log(len(x)) - 2 * loglik

    Z = _cutoff_norm(
        alpha=alpha,
        xc=xc,
        xmin=xmin,
    )

    def cdf_func(xx):
        xx = np.asarray(xx, dtype=float)
        out = np.zeros_like(xx)

        xmax_num = max(xmin * 10.0, 60.0 * xc)
        umax = np.log(xmax_num)

        def log_integrand(u):
            return np.exp((1.0 - alpha) * u - np.exp(u) / xc)

        for i, v in enumerate(xx):
            if v <= xmin:
                out[i] = 0.0
                continue

            uv = np.log(v)

            # Compute the survival probability directly.
            if uv >= umax:
                surv = 0.0
            else:
                tail, _ = quad(
                    log_integrand,
                    uv,
                    umax,
                    epsabs=1e-10,
                    epsrel=1e-9,
                    limit=300,
                )
                surv = tail / Z

            out[i] = 1.0 - surv

        return np.clip(out, 0.0, 1.0)

    ks = _ks_distance(x, cdf_func)

    return FitResult(
        model="powerlaw_exp_cutoff",
        xmin=float(xmin),
        alpha=float(alpha),
        xc=float(xc),
        p_value=np.nan,
        ks_D=ks,
        aic=float(aic),
        bic=float(bic),
        loglik=float(loglik),
        n_used=int(len(x)),
    )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required = {
        "kappa_per",
        "kappa_op",
        "flip_interval",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )

    return df


def analyze_condition(
    df,
    roman,
    phase,
    kper,
    kop,
):
    sub = df[
        np.isclose(
            pd.to_numeric(df["kappa_per"], errors="coerce"),
            kper,
        )
        & np.isclose(
            pd.to_numeric(df["kappa_op"], errors="coerce"),
            kop,
        )
    ]

    x = finite_positive(
        pd.to_numeric(
            sub["flip_interval"],
            errors="coerce",
        ).to_numpy()
    )

    if len(x) == 0:
        raise RuntimeError(
            f"No flip intervals for "
            f"kappa_per={kper}, kappa_op={kop}"
        )

    xmin = FIXED_XMIN

    plc = _fit_powerlaw_exp_cutoff_tail(
        x,
        xmin=xmin,
    )

    exp = _fit_shifted_exp_tail(
        x,
        xmin=xmin,
    )

    delta_aic = exp.aic - plc.aic
    delta_bic = exp.bic - plc.bic

    preferred_aic = (
        "powerlaw_exp_cutoff"
        if delta_aic > 0
        else "shifted_exp"
    )

    preferred_bic = (
        "powerlaw_exp_cutoff"
        if delta_bic > 0
        else "shifted_exp"
    )

    return {
        "condition": roman,
        "phase": phase,
        "kappa_per": kper,
        "kappa_op": kop,
        "n_total": len(x),
        "n_tail": plc.n_used,
        "xmin": plc.xmin,
        "alpha": plc.alpha,
        "xc": plc.xc,
        "plc_KS": plc.ks_D,
        "plc_loglik": plc.loglik,
        "plc_AIC": plc.aic,
        "plc_BIC": plc.bic,
        "exp_lambda": exp.alpha,
        "exp_KS": exp.ks_D,
        "exp_loglik": exp.loglik,
        "exp_AIC": exp.aic,
        "exp_BIC": exp.bic,
        "delta_AIC_exp_minus_plc": delta_aic,
        "delta_BIC_exp_minus_plc": delta_bic,
        "preferred_AIC": preferred_aic,
        "preferred_BIC": preferred_bic,
    }


def main():
    df = load_data(DATA_FILE)

    rows = []

    for roman, phase, kper, kop in CONDITIONS:
        row = analyze_condition(
            df=df,
            roman=roman,
            phase=phase,
            kper=kper,
            kop=kop,
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_CSV,
        index=False,
    )

    table_cols = [
        "condition",
        "phase",
        "kappa_per",
        "kappa_op",
        "n_tail",
        "xmin",
        "alpha",
        "xc",
        "plc_AIC",
        "exp_AIC",
        "delta_AIC_exp_minus_plc",
        "preferred_AIC",
    ]

    printable = result[table_cols].copy()

    text = printable.to_string(
        index=False,
        float_format=lambda v: f"{v:.6g}",
    )

    OUT_TXT.write_text(
        text + "\n",
        encoding="utf-8",
    )

    print(text)
    print()
    print(
        "Delta AIC is defined as "
        "AIC_exp - AIC_powerlaw_exp_cutoff; "
        "positive values favor the power law with exponential cutoff."
    )
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()