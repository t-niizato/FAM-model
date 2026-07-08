#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import power_law_fitting as bplt

EPS = 1e-12


# =========================================================
# Basic utilities
# =========================================================
def finite_positive(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x) & (x > 0)]


def flight_lengths_from_series(series, threshold):
    flights = []
    s = 0.0
    for v in series:
        if v >= threshold:
            s += float(v)
        else:
            if s > 0:
                flights.append(s)
                s = 0.0
    if s > 0:
        flights.append(s)
    return np.asarray(flights, float)


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


def flight_events_with_mask(series, threshold, mask=None, mode="sum"):
    series = np.asarray(series, float)

    if mask is None:
        mask = np.ones(series.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    if series.shape != mask.shape:
        raise ValueError(f"series.shape={series.shape} and mask.shape={mask.shape} must match")

    flights = []
    acc = 0.0
    cnt = 0

    for v, m in zip(series, mask):
        ok = bool(m) and np.isfinite(v) and (v >= threshold)

        if ok:
            acc += float(v)
            cnt += 1
        else:
            if cnt > 0:
                if mode == "sum":
                    flights.append(acc)
                elif mode == "count":
                    flights.append(cnt)
                else:
                    raise ValueError("mode must be 'sum' or 'count'")
                acc = 0.0
                cnt = 0

    if cnt > 0:
        if mode == "sum":
            flights.append(acc)
        elif mode == "count":
            flights.append(cnt)
        else:
            raise ValueError("mode must be 'sum' or 'count'")

    return np.asarray(flights, float)


# =========================================================
# Geometry / decomposition
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


def center_step_lengths(center_x, center_y):
    return np.hypot(np.diff(center_x), np.diff(center_y))


# =========================================================
# State masks
# =========================================================
def milling_mask_from_Lz(Lz, mode="quantile", q=0.7, value=None):
    absL = np.abs(Lz)
    thr = make_threshold(absL, mode=mode, q=q, value=value)
    if not np.isfinite(thr):
        return np.ones_like(absL, dtype=bool)
    return absL >= thr


def polarization_series(X):
    vx = np.diff(X[:, 0, :], axis=0)
    vy = np.diff(X[:, 1, :], axis=0)

    speed = np.hypot(vx, vy)
    ux = vx / np.maximum(speed, EPS)
    uy = vy / np.maximum(speed, EPS)

    px = ux.mean(axis=1)
    py = uy.mean(axis=1)
    return np.hypot(px, py)


def schooling_mask_from_P(P, mode="quantile", q=0.7, value=None):
    thr = make_threshold(P, mode=mode, q=q, value=value)
    if not np.isfinite(thr):
        return np.ones_like(P, dtype=bool)
    return P >= thr


# =========================================================
# Parsing helpers
# =========================================================
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


def fmt_float_for_dir(x: float, nd: int = 6) -> str:
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def group_files_by_param(files):
    groups = {}
    for f in files:
        meta = parse_run_info(f)
        key = (meta["percept_kappa"], meta["option_kappa"])
        groups.setdefault(key, []).append(f)
    return groups


def choose_representative_files_by_N(files):
    """
    各 N について rep が最小のファイルを代表として選ぶ
    """
    best = {}
    for f in files:
        meta = parse_run_info(f)
        N = meta["N"]
        rep = meta["rep"]
        if N not in best:
            best[N] = (rep, f)
        else:
            if rep is not None and best[N][0] is not None:
                if rep < best[N][0]:
                    best[N] = (rep, f)
            elif rep is not None and best[N][0] is None:
                best[N] = (rep, f)

    return {N: f for N, (_, f) in best.items()}


def safe_stem_from_meta(meta: dict) -> str:
    def fnum(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        s = str(x)
        return s.replace("-", "m").replace(".", "p")

    rep_val = meta.get("rep", -1)
    if rep_val is None:
        rep_val = -1

    return (
        f"N{meta.get('N', 'NA')}_"
        f"kappa_{fnum(meta.get('percept_kappa'))}_"
        f"okappa_{fnum(meta.get('option_kappa'))}_"
        f"rep{int(rep_val):02d}_"
        f"seed{meta.get('seed', 'NA')}"
    )


def obj_to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def result_to_metrics_dict(res, fallback_model=None):
    d = obj_to_dict(res)
    return {
        "model": d.get("model", fallback_model),
        "xmin": d.get("xmin"),
        "alpha": d.get("alpha"),
        "xc": d.get("xc"),
        "p_value": d.get("p_value"),
        "ks_D": d.get("ks_D"),
        "aic": d.get("aic"),
        "bic": d.get("bic"),
        "loglik": d.get("loglik"),
        "n_used": d.get("n_used"),
    }


def safe_compare_result(X, models, n_tail_min=200, n_boot=2000, fixed_xmin=None):
    X = finite_positive(X)
    if X.size == 0:
        return {
            "ok": False,
            "error": "no data",
            "summary": "[no data]\n",
            "raw": None,
        }

    try:
        results = bplt.compare_powerlaw_family(
            X,
            models=models,
            n_tail_min=n_tail_min,
            n_boot=n_boot,
            fixed_xmin=fixed_xmin,
        )
        try:
            summary = bplt.summarize_comparison(results) + "\n"
        except Exception:
            summary = str(results) + "\n"
        return {
            "ok": True,
            "error": None,
            "summary": summary,
            "raw": results,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "summary": f"[comparison failed] models={models} fixed_xmin={fixed_xmin} error={e}\n",
            "raw": None,
        }


def safe_gof_result(X, model="truncated", n_tail_min=200, n_boot=2000):
    X = finite_positive(X)
    if X.size == 0:
        return {
            "ok": False,
            "error": "no data",
            "result": None,
            "text": "[no data]\n",
        }

    try:
        res = bplt.fast_powerlaw_gof(
            X, model=model, n_tail_min=n_tail_min, n_boot=n_boot
        )
        rec = result_to_metrics_dict(res, fallback_model=model)
        text = "\n".join([
            f"model: {rec['model']}",
            f"xmin: {rec['xmin']} alpha: {rec['alpha']} xc: {rec['xc']}",
            f"p: {rec['p_value']} KS: {rec['ks_D']}",
            f"AIC: {rec['aic']} BIC: {rec['bic']} loglik: {rec['loglik']}",
        ]) + "\n"
        return {
            "ok": True,
            "error": None,
            "result": rec,
            "text": text,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "result": None,
            "text": f"[gof failed] model={model} error={e}\n",
        }


def safe_gof_result_fixed_xmin(X, xmin, model="truncated", n_boot=2000):
    X = finite_positive(X)
    if X.size == 0:
        return {
            "ok": False,
            "error": "no data",
            "result": None,
            "text": "[no data]\n",
        }

    try:
        res = bplt.gof_with_fixed_xmin(
            X,
            xmin=xmin,
            model=model,
            n_boot=n_boot,
        )
        rec = result_to_metrics_dict(res, fallback_model=model)
        text = "\n".join([
            f"model: {rec['model']}",
            f"xmin: {rec['xmin']} alpha: {rec['alpha']} xc: {rec['xc']}",
            f"p: {rec['p_value']} KS: {rec['ks_D']}",
            f"AIC: {rec['aic']} BIC: {rec['bic']} loglik: {rec['loglik']}",
        ]) + "\n"
        return {
            "ok": True,
            "error": None,
            "result": rec,
            "text": text,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "result": None,
            "text": f"[fixed-xmin gof failed] model={model} xmin={xmin} error={e}\n",
        }


def parse_best_model_from_summary(summary_text, candidate_models):
    if summary_text is None:
        return None

    delta_rows = []
    in_delta_section = False

    for line in summary_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ΔAIC / ΔBIC"):
            in_delta_section = True
            continue
        if in_delta_section:
            if not stripped:
                continue
            matched = None
            for m in candidate_models:
                if stripped.startswith(m):
                    matched = m
                    break
            if matched is None:
                continue

            aic_match = re.search(r"ΔAIC\s*=\s*([^\s]+)", stripped)
            if aic_match is None:
                continue

            try:
                val = float(aic_match.group(1))
            except Exception:
                continue

            delta_rows.append((matched, val))

    if not delta_rows:
        return None

    delta_rows.sort(key=lambda t: t[1])
    return delta_rows[0][0]


def build_text_block_for_kind(label, compare_summary, trunc_text, exp_text=None):
    lines = []
    lines.append(f"=== {label}: model comparison ===")
    lines.append(compare_summary.rstrip())
    lines.append(f"=== {label}: GOF (truncated) ===")
    lines.append(trunc_text.rstrip())
    if exp_text is not None:
        lines.append(f"=== {label}: GOF (shifted_exp) ===")
        lines.append(exp_text.rstrip())
    lines.append("")
    return "\n".join(lines)


# =========================================================
# Worker
# =========================================================
def analyze_one_individual(args):
    (
        ind,
        rad_raw,
        tan_raw,
        tot_raw,
        milling_mask,
        schooling_mask,
        thr_mode,
        thr_q,
        thr_value,
        n_tail_min_individual,
        n_boot_compare,
        n_boot_gof,
    ) = args

    thr_rad = make_threshold(rad_raw[milling_mask], mode=thr_mode, q=thr_q, value=thr_value)
    thr_tan = make_threshold(tan_raw[milling_mask], mode=thr_mode, q=thr_q, value=thr_value)
    thr_tot = make_threshold(tot_raw[schooling_mask], mode=thr_mode, q=thr_q, value=thr_value)

    fl_rad = flight_events_with_mask(rad_raw, thr_rad, mask=milling_mask, mode="sum")
    fl_tan = flight_events_with_mask(tan_raw, thr_tan, mask=milling_mask, mode="sum")
    fl_tot = flight_events_with_mask(tot_raw, thr_tot, mask=schooling_mask, mode="sum")

    rows = []
    text_lines = []

    text_lines.append("============================================================")
    text_lines.append(f"INDIVIDUAL {ind}")
    text_lines.append("============================================================")
    text_lines.append(
        f"thresholds: {{'milling_rad': {thr_rad}, 'milling_tan': {thr_tan}, 'schooling_tot': {thr_tot}}}"
    )
    text_lines.append(
        f"mean/std (milling_rad): {np.mean(rad_raw[milling_mask])} {np.std(rad_raw[milling_mask])}"
    )
    text_lines.append(
        f"mean/std (milling_tan): {np.mean(tan_raw[milling_mask])} {np.std(tan_raw[milling_mask])}"
    )
    text_lines.append(
        f"mean/std (schooling_tot): {np.mean(tot_raw[schooling_mask])} {np.std(tot_raw[schooling_mask])}"
    )
    text_lines.append("")
    text_lines.append(
        f"flight counts: {{'milling_rad': {len(fl_rad)}, 'milling_tan': {len(fl_tan)}, 'schooling_tot': {len(fl_tot)}}}"
    )
    text_lines.append("")

    for kind, series, thr, flights, label in [
        ("milling_rad", rad_raw, thr_rad, fl_rad, "MILLING RAD (radial)"),
        ("milling_tan", tan_raw, thr_tan, fl_tan, "MILLING TAN (tangential)"),
        ("schooling_tot", tot_raw, thr_tot, fl_tot, "SCHOOLING TOTAL (COM-frame relative motion)"),
    ]:
        trunc_full_res = safe_gof_result(
            flights,
            model="truncated",
            n_tail_min=n_tail_min_individual,
            n_boot=n_boot_gof,
        )

        xmin_trunc = None
        if trunc_full_res["result"] is not None:
            xmin_trunc = trunc_full_res["result"].get("xmin")

        cmp_res = safe_compare_result(
            flights,
            models=("shifted_exp", "truncated"),
            n_tail_min=n_tail_min_individual,
            n_boot=n_boot_compare,
            fixed_xmin=xmin_trunc,
        )

        if xmin_trunc is not None and np.isfinite(xmin_trunc):
            trunc_res = safe_gof_result_fixed_xmin(
                flights,
                xmin=xmin_trunc,
                model="truncated",
                n_boot=n_boot_gof,
            )
            exp_res = safe_gof_result_fixed_xmin(
                flights,
                xmin=xmin_trunc,
                model="shifted_exp",
                n_boot=n_boot_gof,
            )
        else:
            trunc_res = {
                "ok": False,
                "error": "failed to estimate truncated xmin",
                "result": None,
                "text": "[fixed-xmin truncated gof skipped]\n",
            }
            exp_res = {
                "ok": False,
                "error": "failed to estimate truncated xmin",
                "result": None,
                "text": "[fixed-xmin shifted_exp gof skipped]\n",
            }

        best_model = parse_best_model_from_summary(
            cmp_res["summary"], ["shifted_exp", "truncated"]
        )

        text_lines.append(
            build_text_block_for_kind(
                label=label,
                compare_summary=cmp_res["summary"],
                trunc_text=trunc_res["text"],
                exp_text=exp_res["text"],
            ).rstrip()
        )

        row = {
            "individual": ind,
            "kind": kind,
            "threshold": float(thr) if np.isfinite(thr) else np.nan,
            "series_mean": float(np.mean(series)) if len(series) else np.nan,
            "series_std": float(np.std(series)) if len(series) else np.nan,
            "flight_count": int(len(flights)),
            "n_positive_flights": int(len(finite_positive(flights))),
            "comparison_ok": bool(cmp_res["ok"]),
            "comparison_error": cmp_res["error"],
            "comparison_summary": cmp_res["summary"],
            "best_model": best_model,
            "fixed_xmin_for_comparison": float(xmin_trunc) if xmin_trunc is not None and np.isfinite(xmin_trunc) else np.nan,
        }

        for prefix, res in [
            ("truncated", trunc_res),
            ("shifted_exp", exp_res),
        ]:
            rr = res["result"] if res["result"] is not None else {}
            row[f"{prefix}_ok"] = bool(res["ok"])
            row[f"{prefix}_error"] = res["error"]
            row[f"{prefix}_model"] = rr.get("model")
            row[f"{prefix}_xmin"] = rr.get("xmin")
            row[f"{prefix}_alpha"] = rr.get("alpha")
            row[f"{prefix}_xc"] = rr.get("xc")
            row[f"{prefix}_p_value"] = rr.get("p_value")
            row[f"{prefix}_ks_D"] = rr.get("ks_D")
            row[f"{prefix}_aic"] = rr.get("aic")
            row[f"{prefix}_bic"] = rr.get("bic")
            row[f"{prefix}_loglik"] = rr.get("loglik")
            row[f"{prefix}_n_used"] = rr.get("n_used")

        if (
            pd.notna(row.get("truncated_aic"))
            and pd.notna(row.get("shifted_exp_aic"))
        ):
            row["delta_aic_trunc_minus_exp"] = (
                row["truncated_aic"] - row["shifted_exp_aic"]
            )
        else:
            row["delta_aic_trunc_minus_exp"] = np.nan

        rows.append(row)
        text_lines.append("")

    return ind, "\n".join(text_lines), rows


# =========================================================
# Plot helpers
# =========================================================
def choose_representative_run(df_individual: pd.DataFrame):
    if df_individual.empty:
        return None

    sub = df_individual[df_individual["kind"] == "schooling_tot"].copy()
    if sub.empty:
        sub = df_individual.copy()

    score = sub["delta_aic_trunc_minus_exp"].copy()
    score = pd.to_numeric(score, errors="coerce")
    sub = sub[np.isfinite(score)]
    if sub.empty:
        return None

    med = np.nanmedian(sub["delta_aic_trunc_minus_exp"].values)
    sub = sub.assign(rep_score=np.abs(sub["delta_aic_trunc_minus_exp"] - med))
    row = sub.sort_values(["rep_score", "flight_count"], ascending=[True, False]).iloc[0]
    return row["path"]


def plot_individual_representative(df_file_individual, out_png):
    kinds = ["milling_rad", "milling_tan", "schooling_tot"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, kind in zip(axes, kinds):
        sub = df_file_individual[df_file_individual["kind"] == kind].copy()
        sub = sub[np.isfinite(pd.to_numeric(sub["delta_aic_trunc_minus_exp"], errors="coerce"))]

        if sub.empty:
            ax.set_title(f"{kind}\nno valid data")
            ax.set_xlabel("individual")
            ax.set_ylabel("delta AIC")
            ax.grid(alpha=0.3)
            continue

        ax.scatter(sub["individual"], sub["delta_aic_trunc_minus_exp"], s=18)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(kind)
        ax.set_xlabel("individual")
        ax.set_ylabel("truncated AIC - shifted_exp AIC")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_com_summary(df_com, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if df_com.empty:
        return

    if "truncated_alpha" in df_com.columns and df_com["truncated_alpha"].notna().any():
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(df_com["N"], df_com["truncated_alpha"], s=30)
        plt.xlabel("N")
        plt.ylabel("COM truncated alpha")
        plt.title("COM Lévy fit alpha by run")
        plt.grid(alpha=0.3)
        fig.savefig(out_dir / "com_alpha_by_run.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if "delta_aic_trunc_minus_exp" in df_com.columns and df_com["delta_aic_trunc_minus_exp"].notna().any():
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(df_com["N"], df_com["delta_aic_trunc_minus_exp"], s=30)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("N")
        plt.ylabel("truncated AIC - shifted_exp AIC")
        plt.title("COM model preference by run")
        plt.grid(alpha=0.3)
        fig.savefig(out_dir / "com_delta_aic_by_run.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if {"N", "truncated_alpha"}.issubset(df_com.columns):
        tmp = df_com.groupby("N")["truncated_alpha"].agg(["mean", "std", "count"]).reset_index()
        tmp = tmp[tmp["count"] > 0]
        if not tmp.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.errorbar(tmp["N"], tmp["mean"], yerr=tmp["std"], fmt="o-", capsize=4)
            plt.xlabel("N")
            plt.ylabel("COM truncated alpha")
            plt.title("COM alpha summary by N")
            plt.grid(alpha=0.3)
            fig.savefig(out_dir / "com_alpha_summary_by_N.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def plot_individual_summary(df_individual, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if df_individual.empty:
        return

    for kind in sorted(df_individual["kind"].dropna().unique()):
        sub = df_individual[df_individual["kind"] == kind].copy()
        if sub.empty:
            continue

        if "truncated_alpha" in sub.columns and sub["truncated_alpha"].notna().any():
            tmp = sub.groupby("N")["truncated_alpha"].agg(["mean", "std", "count"]).reset_index()
            tmp = tmp[tmp["count"] > 0]
            if not tmp.empty:
                fig = plt.figure(figsize=(6, 4))
                plt.errorbar(tmp["N"], tmp["mean"], yerr=tmp["std"], fmt="o-", capsize=4)
                plt.xlabel("N")
                plt.ylabel("truncated alpha")
                plt.title(f"{kind}: alpha summary by N")
                plt.grid(alpha=0.3)
                fig.savefig(out_dir / f"{kind}_alpha_summary_by_N.png", dpi=200, bbox_inches="tight")
                plt.close(fig)

        if "delta_aic_trunc_minus_exp" in sub.columns and sub["delta_aic_trunc_minus_exp"].notna().any():
            tmp = sub.groupby("N")["delta_aic_trunc_minus_exp"].agg(["mean", "std", "count"]).reset_index()
            tmp = tmp[tmp["count"] > 0]
            if not tmp.empty:
                fig = plt.figure(figsize=(6, 4))
                plt.errorbar(tmp["N"], tmp["mean"], yerr=tmp["std"], fmt="o-", capsize=4)
                plt.axhline(0.0, linestyle="--", linewidth=1)
                plt.xlabel("N")
                plt.ylabel("truncated AIC - shifted_exp AIC")
                plt.title(f"{kind}: model preference by N")
                plt.grid(alpha=0.3)
                fig.savefig(out_dir / f"{kind}_delta_aic_summary_by_N.png", dpi=200, bbox_inches="tight")
                plt.close(fig)


# =========================================================
# Analysis cores
# =========================================================
def analyze_individual_for_file(
    npz_path,
    key="pos",
    thr_mode="quantile",
    thr_q=0.80,
    thr_value=None,
    use_milling_mask=True,
    milling_thr_mode="quantile",
    milling_q=0.7,
    milling_value=None,
    use_schooling_mask=True,
    schooling_thr_mode="quantile",
    schooling_q=0.7,
    schooling_value=None,
    n_tail_min_individual=450,
    n_boot_compare=2000,
    n_boot_gof=2000,
    max_workers=None,
):
    meta = parse_run_info(npz_path)

    X = load_positions_npz(npz_path, key=key)
    cx, cy = center_of_mass_trajectory(X)
    rel_x, rel_y = relative_positions(X, cx, cy)
    abs_v_rad, abs_v_tan, abs_v, Lz = decompose_rad_tan(rel_x, rel_y)
    P = polarization_series(X)

    milling_mask = np.ones(Lz.shape[0], dtype=bool)
    if use_milling_mask:
        milling_mask = milling_mask_from_Lz(
            Lz, mode=milling_thr_mode, q=milling_q, value=milling_value
        )

    schooling_mask = np.ones(P.shape[0], dtype=bool)
    if use_schooling_mask:
        schooling_mask = schooling_mask_from_P(
            P, mode=schooling_thr_mode, q=schooling_q, value=schooling_value
        )

    tasks = []
    for ind in range(abs_v.shape[1]):
        tasks.append(
            (
                ind,
                abs_v_rad[:, ind],
                abs_v_tan[:, ind],
                abs_v[:, ind],
                milling_mask,
                schooling_mask,
                thr_mode,
                thr_q,
                thr_value,
                n_tail_min_individual,
                n_boot_compare,
                n_boot_gof,
            )
        )

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    max_workers = max(1, max_workers)

    results_txt = [None] * len(tasks)
    all_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(analyze_one_individual, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Individuals: {Path(npz_path).name}"):
            ind, txt, rows = fut.result()
            results_txt[ind] = txt
            for r in rows:
                r.update(meta)
                r["level"] = "individual"
            all_rows.extend(rows)

    text_report = []
    text_report.append(f"input file: {npz_path}")
    text_report.append(f"shape: {X.shape}")
    text_report.append(f"use_milling_mask: {use_milling_mask}")
    text_report.append(f"milling_keep_ratio: {milling_mask.mean():.6f}")
    text_report.append(f"use_schooling_mask: {use_schooling_mask}")
    text_report.append(f"schooling_keep_ratio: {schooling_mask.mean():.6f}")
    text_report.append("")

    for txt in results_txt:
        text_report.append(txt)
        text_report.append("")

    return {
        "meta": meta,
        "rows": all_rows,
        "text_report": "\n".join(text_report),
    }


def analyze_com_for_file(
    npz_path,
    key="pos",
    thr_mode="quantile",
    thr_q=0.80,
    thr_value=None,
    n_tail_min_com=200,
    n_boot_compare=2000,
    n_boot_gof=2000,
    com_compare_models=("truncated", "powerlaw", "shifted_exp"),
):
    meta = parse_run_info(npz_path)

    X = load_positions_npz(npz_path, key=key)
    cx, cy = center_of_mass_trajectory(X)
    center_step = center_step_lengths(cx, cy)
    center_thr = make_threshold(center_step, mode=thr_mode, q=thr_q, value=thr_value)
    center_flights = flight_lengths_from_series(center_step, center_thr)

    com_trunc_full = safe_gof_result(
        center_flights,
        model="truncated",
        n_tail_min=n_tail_min_com,
        n_boot=n_boot_gof,
    )

    com_xmin_trunc = None
    if com_trunc_full["result"] is not None:
        com_xmin_trunc = com_trunc_full["result"].get("xmin")

    com_cmp = safe_compare_result(
        center_flights,
        models=com_compare_models,
        n_tail_min=n_tail_min_com,
        n_boot=n_boot_compare,
        fixed_xmin=com_xmin_trunc,
    )

    if com_xmin_trunc is not None and np.isfinite(com_xmin_trunc):
        com_trunc = safe_gof_result_fixed_xmin(
            center_flights,
            xmin=com_xmin_trunc,
            model="truncated",
            n_boot=n_boot_gof,
        )
        com_exp = safe_gof_result_fixed_xmin(
            center_flights,
            xmin=com_xmin_trunc,
            model="shifted_exp",
            n_boot=n_boot_gof,
        )
    else:
        com_trunc = {
            "ok": False,
            "error": "failed to estimate truncated xmin",
            "result": None,
            "text": "[fixed-xmin truncated gof skipped]\n",
        }
        com_exp = {
            "ok": False,
            "error": "failed to estimate truncated xmin",
            "result": None,
            "text": "[fixed-xmin shifted_exp gof skipped]\n",
        }

    com_best_model = parse_best_model_from_summary(
        com_cmp["summary"], list(com_compare_models)
    )

    com_row = {
        **meta,
        "level": "com",
        "individual": -1,
        "kind": "com",
        "threshold": float(center_thr) if np.isfinite(center_thr) else np.nan,
        "series_mean": float(np.mean(center_step)) if len(center_step) else np.nan,
        "series_std": float(np.std(center_step)) if len(center_step) else np.nan,
        "flight_count": int(len(center_flights)),
        "n_positive_flights": int(len(finite_positive(center_flights))),
        "comparison_ok": bool(com_cmp["ok"]),
        "comparison_error": com_cmp["error"],
        "comparison_summary": com_cmp["summary"],
        "best_model": com_best_model,
        "fixed_xmin_for_comparison": float(com_xmin_trunc) if com_xmin_trunc is not None and np.isfinite(com_xmin_trunc) else np.nan,
    }

    for prefix, res in [
        ("truncated", com_trunc),
        ("shifted_exp", com_exp),
    ]:
        rr = res["result"] if res["result"] is not None else {}
        com_row[f"{prefix}_ok"] = bool(res["ok"])
        com_row[f"{prefix}_error"] = res["error"]
        com_row[f"{prefix}_model"] = rr.get("model")
        com_row[f"{prefix}_xmin"] = rr.get("xmin")
        com_row[f"{prefix}_alpha"] = rr.get("alpha")
        com_row[f"{prefix}_xc"] = rr.get("xc")
        com_row[f"{prefix}_p_value"] = rr.get("p_value")
        com_row[f"{prefix}_ks_D"] = rr.get("ks_D")
        com_row[f"{prefix}_aic"] = rr.get("aic")
        com_row[f"{prefix}_bic"] = rr.get("bic")
        com_row[f"{prefix}_loglik"] = rr.get("loglik")
        com_row[f"{prefix}_n_used"] = rr.get("n_used")

    if (
        pd.notna(com_row.get("truncated_aic"))
        and pd.notna(com_row.get("shifted_exp_aic"))
    ):
        com_row["delta_aic_trunc_minus_exp"] = (
            com_row["truncated_aic"] - com_row["shifted_exp_aic"]
        )
    else:
        com_row["delta_aic_trunc_minus_exp"] = np.nan

    text_report = []
    text_report.append(f"input file: {npz_path}")
    text_report.append(f"shape: {X.shape}")
    text_report.append("############################################################")
    text_report.append("CENTER OF MASS (NO MASK)")
    text_report.append("############################################################")
    text_report.append(f"COM mean/std: {np.mean(center_step)} {np.std(center_step)}")
    text_report.append(f"COM threshold: {center_thr}")
    text_report.append(f"COM flight count: {len(center_flights)}")
    text_report.append("")
    text_report.append("=== CENTER-OF-MASS flights: model comparison ===")
    text_report.append(com_cmp["summary"])
    text_report.append("=== CENTER-OF-MASS flights: GOF (truncated) ===")
    text_report.append(com_trunc["text"])
    text_report.append("=== CENTER-OF-MASS flights: GOF (shifted_exp) ===")
    text_report.append(com_exp["text"])

    return {
        "meta": meta,
        "rows": [com_row],
        "text_report": "\n".join(text_report),
    }


# =========================================================
# Save helpers
# =========================================================
def save_global_tables(df, out_dir: Path):
    df.to_csv(out_dir / "levy_all_metrics.csv", index=False, encoding="utf-8-sig")

    if "level" in df.columns:
        df[df["level"] == "individual"].to_csv(
            out_dir / "levy_individual_metrics.csv", index=False, encoding="utf-8-sig"
        )
        df[df["level"] == "com"].to_csv(
            out_dir / "levy_com_metrics.csv", index=False, encoding="utf-8-sig"
        )

    if {"level", "kind", "best_model"}.issubset(df.columns):
        wins = (
            df.groupby(["level", "kind", "best_model"])
            .size()
            .reset_index(name="n")
            .sort_values(["level", "kind", "n"], ascending=[True, True, False])
        )
        wins.to_csv(out_dir / "levy_model_wins.csv", index=False, encoding="utf-8-sig")

    summary_targets = [
        "flight_count",
        "threshold",
        "series_mean",
        "series_std",
        "truncated_alpha",
        "truncated_xmin",
        "truncated_xc",
        "truncated_p_value",
        "truncated_ks_D",
        "truncated_aic",
        "shifted_exp_alpha",
        "shifted_exp_xmin",
        "shifted_exp_xc",
        "shifted_exp_p_value",
        "shifted_exp_ks_D",
        "shifted_exp_aic",
        "delta_aic_trunc_minus_exp",
    ]
    use_targets = [c for c in summary_targets if c in df.columns]

    if use_targets and {"level", "kind", "N"}.issubset(df.columns):
        summary = (
            df.groupby(["level", "kind", "N"])[use_targets]
            .agg(["mean", "std", "var", "count"])
            .reset_index()
        )
        summary.columns = [
            "_".join([str(x) for x in col if x != ""]).strip("_")
            for col in summary.columns.to_flat_index()
        ]
        summary.to_csv(out_dir / "levy_summary_by_level_kind_N.csv", index=False, encoding="utf-8-sig")


def save_representative_outputs(df, out_dir: Path):
    rep_dir = out_dir / "representative_individual_run"
    rep_dir.mkdir(parents=True, exist_ok=True)

    df_ind = df[df["level"] == "individual"].copy()
    if df_ind.empty:
        return

    rep_path = choose_representative_run(df_ind)
    if rep_path is None:
        return

    sub = df_ind[df_ind["path"] == rep_path].copy()
    if sub.empty:
        return

    meta = parse_run_info(rep_path)
    stem = safe_stem_from_meta(meta)
    sub.to_csv(rep_dir / f"{stem}_individual_metrics_representative.csv", index=False, encoding="utf-8-sig")
    plot_individual_representative(sub, rep_dir / f"{stem}_individual_delta_aic.png")

    with open(rep_dir / "representative_run_info.txt", "w", encoding="utf-8") as f:
        f.write(f"representative path: {rep_path}\n")
        f.write("criterion: median-nearest delta_aic_trunc_minus_exp within schooling_tot\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


def save_com_outputs(df, out_dir: Path):
    com_dir = out_dir / "com_all_runs"
    com_dir.mkdir(parents=True, exist_ok=True)

    df_com = df[df["level"] == "com"].copy()
    if df_com.empty:
        return

    df_com.to_csv(com_dir / "com_all_runs_metrics.csv", index=False, encoding="utf-8-sig")
    plot_com_summary(df_com, com_dir)


def save_individual_summary_outputs(df, out_dir: Path):
    ind_dir = out_dir / "individual_summary_plots"
    ind_dir.mkdir(parents=True, exist_ok=True)

    df_ind = df[df["level"] == "individual"].copy()
    if df_ind.empty:
        return

    plot_individual_summary(df_ind, ind_dir)


def save_group_readme(out_dir: Path):
    with open(out_dir / "README_outputs.txt", "w", encoding="utf-8") as f:
        f.write("levy_analysis_by_param outputs structure\n")
        f.write("=====================================\n\n")
        f.write("levy_all_metrics.csv\n")
        f.write("  all rows together (individual + com)\n\n")
        f.write("levy_individual_metrics.csv\n")
        f.write("  representative-run individual-level metrics only\n\n")
        f.write("levy_com_metrics.csv\n")
        f.write("  COM metrics for all runs\n\n")
        f.write("levy_summary_by_level_kind_N.csv\n")
        f.write("  grouped summary by level / kind / N\n\n")
        f.write("levy_model_wins.csv\n")
        f.write("  win counts of best_model\n\n")
        f.write("reports/\n")
        f.write("  per-run text reports\n")
        f.write("  - individual reports exist only for representative runs\n")
        f.write("  - com reports exist for all runs\n\n")
        f.write("representative_individual_run/\n")
        f.write("  one representative run visualization inside this parameter group\n\n")
        f.write("com_all_runs/\n")
        f.write("  all COM outputs and plots\n\n")
        f.write("individual_summary_plots/\n")
        f.write("  summary plots across representative individual runs\n")


# =========================================================
# Group analysis
# =========================================================
def analyze_file_group(files, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    error_rows = []

    rep_files_by_N = choose_representative_files_by_N(files)
    rep_file_set = set(rep_files_by_N.values())

    # --- representative index 保存 ---
    rep_index_rows = []
    for N, f in sorted(rep_files_by_N.items()):
        meta = parse_run_info(f)
        rep_index_rows.append({
            "N": N,
            "path": f,
            "rep": meta.get("rep"),
            "seed": meta.get("seed"),
            "percept_kappa": meta.get("percept_kappa"),
            "option_kappa": meta.get("option_kappa"),
        })
    pd.DataFrame(rep_index_rows).to_csv(
        out_dir / "representative_individual_files_by_N.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================
    # COM
    # =========================================================
    if args.run_com:
        print("[COM] analyze all runs")
        for i, f in enumerate(files, start=1):
            print(f"[COM {i}/{len(files)}] {f}")
            try:
                result = analyze_com_for_file(
                    npz_path=f,
                    key=args.key,
                    thr_mode=args.thr_mode,
                    thr_q=args.thr_q,
                    thr_value=args.thr_value,
                    n_tail_min_com=args.n_tail_min_com,
                    n_boot_compare=args.n_boot_compare,
                    n_boot_gof=args.n_boot_gof,
                    com_compare_models=("truncated", "powerlaw", "shifted_exp"),
                )
                rows.extend(result["rows"])

                meta = result["meta"]
                stem = safe_stem_from_meta(meta)
                with open(report_dir / f"{stem}_com_levy_report.txt", "w", encoding="utf-8") as g:
                    g.write(result["text_report"])

            except Exception as e:
                meta = parse_run_info(f)
                error_rows.append({
                    **meta,
                    "path": f,
                    "level": "com",
                    "status": "error",
                    "error": repr(e),
                })
                print(f"  ERROR (COM): {e}")

    # =========================================================
    # INDIVIDUAL
    # =========================================================
    if args.run_individual:
        rep_files_sorted = [rep_files_by_N[N] for N in sorted(rep_files_by_N.keys())]
        print("[INDIVIDUAL] analyze representative run only for each N")

        for i, f in enumerate(rep_files_sorted, start=1):
            print(f"[IND {i}/{len(rep_files_sorted)}] {f}")
            try:
                result = analyze_individual_for_file(
                    npz_path=f,
                    key=args.key,
                    thr_mode=args.thr_mode,
                    thr_q=args.thr_q,
                    thr_value=args.thr_value,
                    use_milling_mask=args.use_milling_mask,
                    milling_thr_mode=args.milling_thr_mode,
                    milling_q=args.milling_q,
                    milling_value=args.milling_value,
                    use_schooling_mask=args.use_schooling_mask,
                    schooling_thr_mode=args.schooling_thr_mode,
                    schooling_q=args.schooling_q,
                    schooling_value=args.schooling_value,
                    n_tail_min_individual=args.n_tail_min_individual,
                    n_boot_compare=args.n_boot_compare,
                    n_boot_gof=args.n_boot_gof,
                    max_workers=args.max_workers,
                )
                rows.extend(result["rows"])

                meta = result["meta"]
                stem = safe_stem_from_meta(meta)
                with open(report_dir / f"{stem}_individual_levy_report.txt", "w", encoding="utf-8") as g:
                    g.write(result["text_report"])

            except Exception as e:
                meta = parse_run_info(f)
                error_rows.append({
                    **meta,
                    "path": f,
                    "level": "individual",
                    "status": "error",
                    "error": repr(e),
                })
                print(f"  ERROR (IND): {e}")

    # =========================================================
    # MERGE（ここが超重要）
    # =========================================================
    existing_path = out_dir / "levy_all_metrics.csv"

    if existing_path.exists():
        df_old = pd.read_csv(existing_path)
    else:
        df_old = pd.DataFrame()

    df_new = pd.DataFrame(rows)

    if not df_old.empty:
        # 上書き対象だけ削除
        if args.run_com and not args.run_individual:
            df_old = df_old[df_old["level"] != "com"]

        if args.run_individual and not args.run_com:
            df_old = df_old[df_old["level"] != "individual"]

        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # 重複除去（安全）
    key_cols = ["path", "level", "individual", "kind"]
    key_cols = [c for c in key_cols if c in df.columns]

    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last")

    # =========================================================
    # SAVE
    # =========================================================
    if not df.empty:
        save_global_tables(df, out_dir)
        save_representative_outputs(df, out_dir)
        save_com_outputs(df, out_dir)
        save_individual_summary_outputs(df, out_dir)

    if error_rows:
        pd.DataFrame(error_rows).to_csv(
            out_dir / "levy_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_group_readme(out_dir)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Parameter-grouped Lévy analysis for swarm position npz files")

    ap.add_argument("--root", type=str, required=True,
                    help="root directory of batch_record_positions outputs")
    ap.add_argument("--out", type=str, default=None,
                    help="outputs analysis directory (default: <root>/levy_analysis_by_param)")
    ap.add_argument("--key", type=str, default="pos")
    ap.add_argument("--max-files", type=int, default=None,
                    help="debug: analyze only first N files globally")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="number of worker processes for per-individual analysis")

    ap.add_argument("--thr-mode", type=str, default="quantile", choices=["quantile", "value"])
    ap.add_argument("--thr-q", type=float, default=0.80)
    ap.add_argument("--thr-value", type=float, default=None)

    ap.add_argument("--use-milling-mask", action="store_true", default=True)
    ap.add_argument("--no-milling-mask", action="store_false", dest="use_milling_mask")
    ap.add_argument("--milling-thr-mode", type=str, default="quantile", choices=["quantile", "value"])
    ap.add_argument("--milling-q", type=float, default=0.80)
    ap.add_argument("--milling-value", type=float, default=None)

    ap.add_argument("--use-schooling-mask", action="store_true", default=True)
    ap.add_argument("--no-schooling-mask", action="store_false", dest="use_schooling_mask")
    ap.add_argument("--schooling-thr-mode", type=str, default="quantile", choices=["quantile", "value"])
    ap.add_argument("--schooling-q", type=float, default=0.80)
    ap.add_argument("--schooling-value", type=float, default=None)

    ap.add_argument("--n-tail-min-individual", type=int, default=1100)
    ap.add_argument("--n-tail-min-com", type=int, default=500)
    ap.add_argument("--n-boot-compare", type=int, default=1000)
    ap.add_argument("--n-boot-gof", type=int, default=1000)

    ap.add_argument("--run-com", action="store_true", default=True)
    ap.add_argument("--no-run-com", action="store_false", dest="run_com")
    ap.add_argument("--run-individual", action="store_true", default=True)
    ap.add_argument("--no-run-individual", action="store_false", dest="run_individual")

    ap.add_argument("--percept-kappa", type=float, default=None,
                    help="analyze only this percept_kappa if given")
    ap.add_argument("--option-kappa", type=float, default=None,
                    help="analyze only this option_kappa if given")

    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    if args.out is None:
        out_dir = root / "levy_analysis_by_param"
    else:
        out_dir = Path(os.path.expanduser(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(root / "N*" / "kappa_*" / "okappa_*" / "pos_rep*_seed*.npz")))
    if args.max_files is not None:
        files = files[:args.max_files]

    print(f"[root] {root}")
    print(f"[out_dir] {out_dir}")
    print(f"[n_files_total] {len(files)}")

    if len(files) == 0:
        print("[done] no files found")
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
        print("[done] no matching parameter groups")
        return

    index_rows = []

    for (pk, ok), group_files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        group_name = f"kappa_{fmt_float_for_dir(pk)}__okappa_{fmt_float_for_dir(ok)}"
        group_out = out_dir / group_name

        print("=" * 80)
        print(f"[group] percept_kappa={pk}, option_kappa={ok}")
        print(f"[group_name] {group_name}")
        print(f"[n_files] {len(group_files)}")

        analyze_file_group(group_files, group_out, args)

        index_rows.append({
            "group_name": group_name,
            "percept_kappa": pk,
            "option_kappa": ok,
            "n_files_total": len(group_files),
            "n_N_values": len(set(parse_run_info(f)["N"] for f in group_files)),
            "out_dir": str(group_out),
        })

    pd.DataFrame(index_rows).to_csv(
        out_dir / "group_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(out_dir / "README_groups.txt", "w", encoding="utf-8") as f:
        f.write("levy_analysis_by_param\n")
        f.write("======================\n\n")
        f.write("Each subdirectory corresponds to one (percept_kappa, option_kappa) condition.\n")
        f.write("Within each condition:\n")
        f.write("- COM analysis uses all replicate files.\n")
        f.write("- individual analysis uses one representative run per N.\n")
        f.write("- representative individual run is chosen as the smallest rep index within each N.\n")

    print("[done]")


if __name__ == "__main__":
    main()