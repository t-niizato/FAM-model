#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure_3_fit_check.py

Diagnostic plots for Figure 3 flip-interval fitting.

This script is for visual inspection only, not for publication output.\nThe cutoff-power-law normalization and CCDF use the same stable\nlog-space integration as the analysis script.

For each parameter condition, plot:
    - empirical CCDF
    - fitted power law with exponential cutoff
    - fitted shifted exponential

Models
------
Power law with exponential cutoff:
    p(x) ∝ x^{-alpha} exp(-x / xc), x >= xmin

Shifted exponential:
    p(x) = lambda exp[-lambda (x - xmin)], x >= xmin

Input:
    data/processed/figure3/Figure_flip_interval_survival_data.csv

Output:
    analysis/diagnostics/output/Figure_3_fit_check.png
    analysis/diagnostics/output/Figure_3_fit_check.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize


EPS = 1e-12

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "figure3"
    / "Figure_flip_interval_survival_data.csv"
)

OUT_DIR = (
    ROOT
    / "analysis"
    / "diagnostics"
    / "output"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Figure_3_fit_check.png"
OUT_PDF = OUT_DIR / "Figure_3_fit_check.pdf"


CONDITIONS = [
    ("I",   "Milling phase", 2.5, 3.0),
    ("III", "MS phase",      1.5, 2.0),
    ("IV",  "MS phase",      2.5, 1.8),
    ("V",   "SMS phase",     0.5, 1.5),
]


FIXED_XMIN = 1.0


def finite_positive(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


def empirical_ccdf(x):
    x = np.sort(finite_positive(x))

    if len(x) == 0:
        return np.array([]), np.array([])

    vals, counts = np.unique(x, return_counts=True)
    cumulative = np.cumsum(counts[::-1])[::-1]
    surv = cumulative / cumulative[0]

    return vals, surv


def fit_shifted_exp(x, xmin):
    x = finite_positive(x)
    x = x[x >= xmin]

    if len(x) < 2:
        raise ValueError("too few samples")

    z = x - xmin
    lam = 1.0 / max(np.mean(z), EPS)

    loglik = len(x) * np.log(lam) - lam * np.sum(z)

    aic = 2 - 2 * loglik

    return {
        "lambda": float(lam),
        "loglik": float(loglik),
        "aic": float(aic),
        "n": len(x),
    }


def cutoff_norm(alpha, xc, xmin):
    """
    Stable normalization for

        p(x) ∝ x^{-alpha} exp(-x / xc), x >= xmin

    using x = exp(u).
    """
    if not np.isfinite(alpha) or not np.isfinite(xc):
        return np.nan
    if xmin <= 0 or xc <= 0:
        return np.nan

    umin = np.log(xmin)

    # exp(-x/xc) is negligible beyond roughly 60 * xc.
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


def fit_powerlaw_exp_cutoff(x, xmin):
    x = finite_positive(x)
    x = x[x >= xmin]

    if len(x) < 3:
        raise ValueError("too few samples")

    xmax_obs = float(np.max(x))
    mean_x = float(np.mean(x))

    def neg_loglik(theta):
        alpha = float(theta[0])
        xc = float(np.exp(theta[1]))

        if alpha < 0 or xc <= 0:
            return np.inf

        Z = cutoff_norm(alpha, xc, xmin)

        if not np.isfinite(Z) or Z <= 0:
            return np.inf

        ll = (
            -alpha * np.sum(np.log(x))
            - np.sum(x) / xc
            - len(x) * np.log(Z)
        )

        return -ll

    guesses = [
        np.array([1.0, np.log(max(mean_x, xmin))]),
        np.array([0.5, np.log(max(mean_x, xmin))]),
        np.array([1.5, np.log(max(mean_x, xmin))]),
        np.array([1.0, np.log(max(xmax_obs / 2.0, xmin))]),
    ]

    bounds = [
        (0.0, 10.0),
        (
            np.log(max(xmin * 0.01, EPS)),
            np.log(max(xmax_obs * 100.0, xmin * 10.0)),
        ),
    ]

    best = None

    for x0 in guesses:
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
        raise RuntimeError("power-law cutoff fit failed")

    alpha = float(best.x[0])
    xc = float(np.exp(best.x[1]))
    loglik = float(-best.fun)

    aic = 2 * 2 - 2 * loglik

    return {
        "alpha": alpha,
        "xc": xc,
        "loglik": loglik,
        "aic": float(aic),
        "n": len(x),
    }


def ccdf_shifted_exp(grid, xmin, lam):
    out = np.zeros_like(grid, dtype=float)
    m = grid >= xmin
    out[m] = np.exp(-lam * (grid[m] - xmin))
    return out


def ccdf_powerlaw_exp_cutoff(grid, xmin, alpha, xc):
    """
    Fitted CCDF computed with the same stable log-space integral
    used in the fitting code.
    """
    Z = cutoff_norm(alpha, xc, xmin)

    out = np.zeros_like(grid, dtype=float)

    xmax_num = max(xmin * 10.0, 60.0 * xc)
    umax = np.log(xmax_num)

    def integrand(u):
        return np.exp((1.0 - alpha) * u - np.exp(u) / xc)

    for i, x in enumerate(grid):
        if x <= xmin:
            out[i] = 1.0
            continue

        ux = np.log(x)

        if ux >= umax:
            out[i] = 0.0
            continue

        tail, _ = quad(
            integrand,
            ux,
            umax,
            epsabs=1e-10,
            epsrel=1e-9,
            limit=300,
        )

        out[i] = tail / Z

    return np.clip(out, 0.0, 1.0)


def main():
    df = pd.read_csv(DATA_FILE)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 6.2),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (roman, phase, kper, kop) in zip(axes, CONDITIONS):
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
            ax.set_visible(False)
            continue

        xmin = FIXED_XMIN

        plc = fit_powerlaw_exp_cutoff(
            x,
            xmin=xmin,
        )

        exp = fit_shifted_exp(
            x,
            xmin=xmin,
        )

        x_emp, y_emp = empirical_ccdf(x)

        xmax = float(np.max(x_emp))
        grid = np.logspace(
            np.log10(xmin),
            np.log10(xmax),
            250,
        )

        y_plc = ccdf_powerlaw_exp_cutoff(
            grid,
            xmin=xmin,
            alpha=plc["alpha"],
            xc=plc["xc"],
        )

        y_exp = ccdf_shifted_exp(
            grid,
            xmin=xmin,
            lam=exp["lambda"],
        )

        ax.loglog(
            x_emp,
            y_emp,
            "o",
            markersize=2.0,
            alpha=0.65,
            label="Empirical CCDF",
        )

        ax.loglog(
            grid,
            y_plc,
            "-",
            linewidth=1.4,
            label="Power law + exp. cutoff",
        )

        ax.loglog(
            grid,
            y_exp,
            "--",
            linewidth=1.2,
            label="Shifted exponential",
        )

        delta_aic = exp["aic"] - plc["aic"]

        ax.text(
            0.04,
            0.05,
            (
                rf"$\alpha={plc['alpha']:.3f}$" "\n"
                rf"$x_c={plc['xc']:.3g}$" "\n"
                rf"$\Delta AIC={delta_aic:.1f}$" "\n"
                rf"$n={len(x)}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )

        ax.set_title(
            rf"{roman}: {phase}"
            + "\n"
            + rf"$\kappa_{{\mathrm{{per}}}}={kper},\ "
              rf"\kappa_{{\mathrm{{op}}}}={kop}$",
            fontsize=9,
        )

        ax.set_xlabel("Flip interval")
        ax.set_ylabel(r"$P(T \geq t)$")

        ax.grid(
            which="both",
            alpha=0.18,
            linewidth=0.35,
        )

        ax.legend(
            frameon=False,
            fontsize=7,
            loc="upper right",
        )

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()