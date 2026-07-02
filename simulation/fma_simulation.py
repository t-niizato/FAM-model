"""
CUDA版（Numba）: Delaunay近傍は毎ステップ更新／サンプリングは統計的に必要（打ち切り近似なし）

- Delaunay: CPU (SciPy) で毎ステップ再計算
- interaction/select_options: GPU (Numba CUDA)
- サンプリング: group_unique_and_sample のロジックを「可変長配列生成なし」でGPU内再現
  - 各近傍(=row)ごとに条件を満たす列数 width を数える
  - 直積が小さい場合のみ全列挙（TH）
  - 大きい場合は with-replacement で size=sample_size パターンを生成
  - ただし候補集合を固定長Kで打ち切る近似はしない（全候補を母集団として扱う）

追加（オプション）:
- compute_stats=True のときだけ、選択統計を出力:
  H_sel[i], U_sel[i], meanH[i], meanU[i]

前提:
- N=100..500
- option_number=30..60 程度
- division <= 63 推奨（rep_mask を int64 に収めるため）
- sample_size <= 100（ローカル配列固定長）
- option_number <= 64（ties固定長）
"""

from __future__ import annotations

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numba import njit
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from scipy.spatial import Delaunay

import move_function_safe as f_cpu
import hetero_function_safe as hetero

from tqdm import trange
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

NAN = np.nan


# =========================
# ユーティリティ（CPU）
# =========================

@njit(cache=True)
def _compute_valid_mask(nx: np.ndarray, ny: np.ndarray, L: float) -> np.ndarray:
    return (nx >= 0.0) & (nx <= L) & (ny >= 0.0) & (ny <= L)


@njit(cache=True)
def _compute_radius2_from_neighbors(position_2N: np.ndarray,
                                   indptr: np.ndarray,
                                   indices: np.ndarray,
                                   V: float) -> np.ndarray:
    """CPUで i ごとの radius^2 を作る。

    CPU版は
      radius = max(||pos[neighbor]-pos[i]||)  (近傍が空なら V*20)
    を使っている。
    """
    N = position_2N.shape[1]
    out = np.empty(N, dtype=np.float64)
    default_r = V * 20.0
    default_r2 = default_r * default_r

    for i in range(N):
        s = indptr[i]
        e = indptr[i + 1]
        if e <= s:
            out[i] = default_r2
            continue
        x0 = position_2N[0, i]
        y0 = position_2N[1, i]
        m2 = 0.0
        for p in range(s, e):
            j = indices[p]
            dx = position_2N[0, j] - x0
            dy = position_2N[1, j] - y0
            d2 = dx * dx + dy * dy
            if d2 > m2:
                m2 = d2
        out[i] = m2
    return out


# =========================
# CUDA device 関数
# =========================

@cuda.jit(device=True, inline=True)
def _wrap_pi(x: float) -> float:
    """[-pi, pi) へ wrap"""
    two_pi = 2.0 * math.pi
    pi = math.pi
    x = (x + pi) % two_pi - pi
    if x >= pi:
        x -= two_pi
    return x


@cuda.jit(device=True, inline=True)
def _bin_search_edges(edges, angle) -> int:
    """edges は単調増加（長さ = division+1）。np.searchsorted(right)-1 相当。"""
    lo = 0
    hi = edges.size
    while lo < hi:
        mid = (lo + hi) >> 1
        if angle < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    idx = lo - 1
    if idx < 0:
        return 0
    nsec = edges.size - 1
    if idx >= nsec:
        return nsec - 1
    return idx


@cuda.jit(device=True, inline=True)
def _popcount_u64(x: int) -> int:
    """int64 popcount"""
    c = 0
    v = x
    while v:
        v &= v - 1
        c += 1
    return c


@cuda.jit(device=True, inline=True)
def _count_valid_cols_for_neighbor(
    neigh: int,
    nx,
    ny,
    valid_mask,
    tx: float,
    ty: float,
    r2: float,
    rep2: float,
) -> int:
    """近傍個体 neigh の option 列のうち、rep2<dist2<r2 を満たす数を数える。"""
    K = nx.shape[1]
    w = 0
    for c in range(K):
        if not valid_mask[neigh, c]:
            continue
        dx = nx[neigh, c] - tx
        dy = ny[neigh, c] - ty
        d2 = dx * dx + dy * dy
        if d2 < r2 and d2 > rep2:
            w += 1
    return w


@cuda.jit(device=True, inline=True)
def _pick_kth_valid_col(
    neigh: int,
    nx,
    ny,
    valid_mask,
    tx: float,
    ty: float,
    r2: float,
    rep2: float,
    k: int,
) -> int:
    """rep2<dist2<r2 を満たす列を 0..width-1 の順に数え、k番目の列indexを返す。"""
    K = nx.shape[1]
    cnt = 0
    for c in range(K):
        if not valid_mask[neigh, c]:
            continue
        dx = nx[neigh, c] - tx
        dy = ny[neigh, c] - ty
        d2 = dx * dx + dy * dy
        if d2 < r2 and d2 > rep2:
            if cnt == k:
                return c
            cnt += 1
    return -1


@cuda.jit(device=True, inline=True)
def _update_rep_mask_for_neighbor(
    neigh: int,
    nx,
    ny,
    valid_mask,
    tx: float,
    ty: float,
    rep2: float,
    base_theta: float,
    edges,
    rep_mask: int,
) -> int:
    """dist2<rep2 の候補から rep_mask を更新。"""
    K = nx.shape[1]
    pi = math.pi

    for c in range(K):
        if not valid_mask[neigh, c]:
            continue
        dx = nx[neigh, c] - tx
        dy = ny[neigh, c] - ty
        d2 = dx * dx + dy * dy
        if d2 >= rep2:
            continue

        ang = math.atan2(dy, dx)
        d = _wrap_pi(ang - base_theta - pi)
        idx = _bin_search_edges(edges, d)
        rep_mask |= (1 << idx)

    return rep_mask


@cuda.jit(device=True, inline=True)
def _bits_from_pattern(
    active_neigh,
    widths,
    G,
    nx,
    ny,
    valid_mask,
    tx,
    ty,
    r2,
    rep2,
    base_theta,
    edges,
    keep_mask,
    rng_states,
    rng_id,
) -> int:
    """1パターン（各近傍から1列を選ぶ）をサンプリングし、bits を返す。"""
    bits = 0
    pi = math.pi

    for g in range(G):
        w = widths[g]
        if w <= 0:
            continue
        u = xoroshiro128p_uniform_float64(rng_states, rng_id)
        k = int(u * w)
        if k >= w:
            k = w - 1

        neigh = active_neigh[g]
        col = _pick_kth_valid_col(neigh, nx, ny, valid_mask, tx, ty, r2, rep2, k)
        if col < 0:
            continue

        dx = nx[neigh, col] - tx
        dy = ny[neigh, col] - ty

        ang = math.atan2(dy, dx) - base_theta - pi
        ang = _wrap_pi(ang)
        idx = _bin_search_edges(edges, ang)
        bits |= (1 << idx)

    bits &= keep_mask
    return bits


@cuda.jit(device=True, inline=True)
def _unique_count_and_pop_sum(sample_bits, nsamp) -> Tuple[int, int]:
    """sample_bits[0:nsamp] のユニーク数と popcount 和（ユニーク状態だけ）"""
    uniq = cuda.local.array(100, dtype=np.int64)
    ucnt = 0
    pop = 0

    for i in range(nsamp):
        b = sample_bits[i]
        if b == 0:
            continue

        seen = False
        for j in range(ucnt):
            if uniq[j] == b:
                seen = True
                break
        if not seen:
            uniq[ucnt] = b
            ucnt += 1
            pop += _popcount_u64(b)

    return ucnt, pop


# ---- 追加: エントロピー（頻度ベース） ----
@cuda.jit(device=True, inline=True)
def _unique_counts_entropy(sample_bits, nsamp):
    """
    sample_bits[0:nsamp] の頻度分布から
      ucnt = ユニーク状態数
      H    = -sum p log p  (natural log)
    を返す
    """
    uniq = cuda.local.array(100, dtype=np.int64)
    cnts = cuda.local.array(100, dtype=np.int32)
    ucnt = 0

    for i in range(nsamp):
        b = sample_bits[i]
        if b == 0:
            continue
        found = False
        for j in range(ucnt):
            if uniq[j] == b:
                cnts[j] += 1
                found = True
                break
        if not found:
            uniq[ucnt] = b
            cnts[ucnt] = 1
            ucnt += 1

    nsamp_eff = 0
    for j in range(ucnt):
        nsamp_eff += cnts[j]

    H = 0.0
    if nsamp_eff > 0:
        inv = 1.0 / nsamp_eff
        for j in range(ucnt):
            p = cnts[j] * inv
            H -= p * math.log(p)

    return ucnt, H


# =========================
# CUDA kernels
# =========================

@cuda.jit
def kernel_eval_optpop(
    nx,
    ny,
    valid_mask,
    indptr,
    indices,
    theta,
    options,
    radius2,
    rep: float,
    edges,
    sample_size: int,
    TH: int,
    rng_states,
    opt_out,
    pop_out,
):
    """(i,opt) ごとに opt_count/pop_sum を計算。"""
    i, opt_idx = cuda.grid(2)
    N = nx.shape[0]
    K = nx.shape[1]
    if i >= N or opt_idx >= K:
        return

    if not valid_mask[i, opt_idx]:
        opt_out[i, opt_idx] = -1
        pop_out[i, opt_idx] = -1
        return

    rep2 = rep * rep
    r2 = radius2[i]

    tx = nx[i, opt_idx]
    ty = ny[i, opt_idx]

    s = indptr[i]
    e = indptr[i + 1]
    deg = e - s
    if deg <= 0:
        opt_out[i, opt_idx] = 0
        pop_out[i, opt_idx] = 0
        return

    active_neigh = cuda.local.array(64, dtype=np.int32)
    widths = cuda.local.array(64, dtype=np.int32)
    G = 0

    base_theta = theta[i] + options[i, opt_idx]
    rep_mask = 0

    for p in range(s, e):
        neigh = indices[p]
        rep_mask = _update_rep_mask_for_neighbor(neigh, nx, ny, valid_mask, tx, ty, rep2, base_theta, edges, rep_mask)
        w = _count_valid_cols_for_neighbor(neigh, nx, ny, valid_mask, tx, ty, r2, rep2)
        if w > 0:
            if G < 64:
                active_neigh[G] = neigh
                widths[G] = w
                G += 1

    if G == 0:
        opt_out[i, opt_idx] = 0
        pop_out[i, opt_idx] = 0
        return

    division = edges.size - 1
    mask_all = (1 << division) - 1
    keep_mask = (~rep_mask) & mask_all

    small = True
    T = 1
    for g in range(G):
        w = widths[g]
        if w <= 0:
            small = False
            break
        if T > TH // w:
            small = False
            break
        T *= w

    rng_id = i * K + opt_idx

    if small:
        sample_bits = cuda.local.array(100, dtype=np.int64)
        nsamp = 0

        for t in range(T):
            L = t
            bits = 0
            for g in range(G):
                w = widths[g]
                q = L // w
                d = L - q * w
                L = q

                neigh = active_neigh[g]
                col = _pick_kth_valid_col(neigh, nx, ny, valid_mask, tx, ty, r2, rep2, d)
                if col < 0:
                    continue

                dx = nx[neigh, col] - tx
                dy = ny[neigh, col] - ty

                ang = math.atan2(dy, dx) - base_theta - math.pi
                ang = _wrap_pi(ang)
                idx = _bin_search_edges(edges, ang)
                bits |= (1 << idx)

            bits &= keep_mask
            if bits != 0 and nsamp < 100:
                sample_bits[nsamp] = bits
                nsamp += 1

        optc, pop = _unique_count_and_pop_sum(sample_bits, nsamp)
        opt_out[i, opt_idx] = optc
        pop_out[i, opt_idx] = pop
        return

    sample_bits = cuda.local.array(100, dtype=np.int64)
    nsamp = sample_size
    if nsamp > 100:
        nsamp = 100

    for sidx in range(nsamp):
        bits = _bits_from_pattern(
            active_neigh, widths, G,
            nx, ny, valid_mask,
            tx, ty, r2, rep2,
            base_theta, edges, keep_mask,
            rng_states, rng_id,
        )
        sample_bits[sidx] = bits

    optc, pop = _unique_count_and_pop_sum(sample_bits, nsamp)
    opt_out[i, opt_idx] = optc
    pop_out[i, opt_idx] = pop


# ---- 追加: H_out を埋める eval kernel（compute_stats=True 用） ----
@cuda.jit
def kernel_eval_optpop_entropy(
    nx,
    ny,
    valid_mask,
    indptr,
    indices,
    theta,
    options,
    radius2,
    rep: float,
    edges,
    sample_size: int,
    TH: int,
    rng_states,
    opt_out,
    pop_out,
    H_out,
):
    """(i,opt) ごとに U(opt_out) と H(H_out) を計算。pop_out は tie-break 用に残す（不要なら0固定）。"""
    i, opt_idx = cuda.grid(2)
    N = nx.shape[0]
    K = nx.shape[1]
    if i >= N or opt_idx >= K:
        return

    if not valid_mask[i, opt_idx]:
        opt_out[i, opt_idx] = -1
        pop_out[i, opt_idx] = -1
        H_out[i, opt_idx] = NAN
        return

    rep2 = rep * rep
    r2 = radius2[i]

    tx = nx[i, opt_idx]
    ty = ny[i, opt_idx]

    s = indptr[i]
    e = indptr[i + 1]
    deg = e - s
    if deg <= 0:
        opt_out[i, opt_idx] = 0
        pop_out[i, opt_idx] = 0
        H_out[i, opt_idx] = 0.0
        return

    active_neigh = cuda.local.array(64, dtype=np.int32)
    widths = cuda.local.array(64, dtype=np.int32)
    G = 0

    base_theta = theta[i] + options[i, opt_idx]
    rep_mask = 0

    for p in range(s, e):
        neigh = indices[p]
        rep_mask = _update_rep_mask_for_neighbor(neigh, nx, ny, valid_mask, tx, ty, rep2, base_theta, edges, rep_mask)
        w = _count_valid_cols_for_neighbor(neigh, nx, ny, valid_mask, tx, ty, r2, rep2)
        if w > 0:
            if G < 64:
                active_neigh[G] = neigh
                widths[G] = w
                G += 1

    if G == 0:
        opt_out[i, opt_idx] = 0
        pop_out[i, opt_idx] = 0
        H_out[i, opt_idx] = 0.0
        return

    division = edges.size - 1
    mask_all = (1 << division) - 1
    keep_mask = (~rep_mask) & mask_all

    small = True
    T = 1
    for g in range(G):
        w = widths[g]
        if w <= 0:
            small = False
            break
        if T > TH // w:
            small = False
            break
        T *= w

    rng_id = i * K + opt_idx

    if small:
        sample_bits = cuda.local.array(100, dtype=np.int64)
        nsamp = 0

        for t in range(T):
            L = t
            bits = 0
            for g in range(G):
                w = widths[g]
                q = L // w
                d = L - q * w
                L = q

                neigh = active_neigh[g]
                col = _pick_kth_valid_col(neigh, nx, ny, valid_mask, tx, ty, r2, rep2, d)
                if col < 0:
                    continue

                dx = nx[neigh, col] - tx
                dy = ny[neigh, col] - ty

                ang = math.atan2(dy, dx) - base_theta - math.pi
                ang = _wrap_pi(ang)
                idx = _bin_search_edges(edges, ang)
                bits |= (1 << idx)

            bits &= keep_mask
            if bits != 0 and nsamp < 100:
                sample_bits[nsamp] = bits
                nsamp += 1

        # UとH（頻度ベース）
        optc, H = _unique_counts_entropy(sample_bits, nsamp)
        opt_out[i, opt_idx] = optc

        # pop_out は tie-break のために「ユニーク状態のpopcount和」を残す（あなたのルール維持）
        optc2, pop = _unique_count_and_pop_sum(sample_bits, nsamp)
        pop_out[i, opt_idx] = pop

        H_out[i, opt_idx] = H
        return

    sample_bits = cuda.local.array(100, dtype=np.int64)
    nsamp = sample_size
    if nsamp > 100:
        nsamp = 100

    for sidx in range(nsamp):
        bits = _bits_from_pattern(
            active_neigh, widths, G,
            nx, ny, valid_mask,
            tx, ty, r2, rep2,
            base_theta, edges, keep_mask,
            rng_states, rng_id,
        )
        sample_bits[sidx] = bits

    optc, H = _unique_counts_entropy(sample_bits, nsamp)
    opt_out[i, opt_idx] = optc
    optc2, pop = _unique_count_and_pop_sum(sample_bits, nsamp)
    pop_out[i, opt_idx] = pop
    H_out[i, opt_idx] = H


@cuda.jit
def kernel_choose_selected(
    valid_mask,
    options,
    opt_in,
    pop_in,
    rng_states,
    selected_out,
):
    """各 i について best option を選ぶ（choose_index をCUDA化）。"""
    i = cuda.grid(1)
    N = valid_mask.shape[0]
    K = valid_mask.shape[1]
    if i >= N:
        return

    rng_id = i

    best_opt = -1
    best_pop = -1
    best_k = -1
    tie_count = 0
    ties = cuda.local.array(64, dtype=np.int32)  # K<=64 前提

    for k in range(K):
        if not valid_mask[i, k]:
            continue
        o = opt_in[i, k]
        p = pop_in[i, k]
        if o > best_opt or (o == best_opt and p > best_pop):
            best_opt = o
            best_pop = p
            best_k = k
            tie_count = 1
            ties[0] = k
        elif o == best_opt and p == best_pop:
            ties[tie_count] = k
            tie_count += 1

    if best_k < 0:
        u = xoroshiro128p_uniform_float64(rng_states, rng_id)
        selected_out[i] = (math.pi / 2) if (u < 0.5) else (-math.pi / 2)
        return

    if tie_count > 1:
        u = xoroshiro128p_uniform_float64(rng_states, rng_id)
        j = int(u * tie_count)
        if j >= tie_count:
            j = tie_count - 1
        best_k = ties[j]

    selected_out[i] = options[i, best_k]


# ---- 追加: stats を出す choose kernel（compute_stats=True 用） ----
@cuda.jit
def kernel_choose_selected_with_stats(
    valid_mask,
    options,
    opt_in,
    pop_in,
    H_in,
    rng_states,
    selected_out,
    H_sel_out,
    U_sel_out,
    meanH_out,
    meanU_out,
):
    i = cuda.grid(1)
    N = valid_mask.shape[0]
    K = valid_mask.shape[1]
    if i >= N:
        return

    rng_id = i

    best_opt = -1
    best_pop = -1
    best_k = -1

    tie_count = 0
    ties = cuda.local.array(64, dtype=np.int32)  # K<=64 前提

    sumH = 0.0
    sumU = 0.0
    cnt = 0

    for k in range(K):
        if not valid_mask[i, k]:
            continue

        o = opt_in[i, k]
        p = pop_in[i, k]
        h = H_in[i, k]

        # mean 用
        # H_in は nan のことがあるので弾く（基本 valid_mask True なら nan にはしないが保険）
        if not math.isnan(h):
            sumH += h
            sumU += float(o)
            cnt += 1

        # choose（既存ルール維持）
        if o > best_opt or (o == best_opt and p > best_pop):
            best_opt = o
            best_pop = p
            best_k = k
            tie_count = 1
            ties[0] = k
        elif o == best_opt and p == best_pop:
            ties[tie_count] = k
            tie_count += 1

    if best_k < 0:
        u = xoroshiro128p_uniform_float64(rng_states, rng_id)
        selected_out[i] = (math.pi / 2) if (u < 0.5) else (-math.pi / 2)
        H_sel_out[i] = NAN
        U_sel_out[i] = -1
        meanH_out[i] = NAN
        meanU_out[i] = NAN
        return

    if tie_count > 1:
        u = xoroshiro128p_uniform_float64(rng_states, rng_id)
        j = int(u * tie_count)
        if j >= tie_count:
            j = tie_count - 1
        best_k = ties[j]

    selected_out[i] = options[i, best_k]
    H_sel_out[i] = H_in[i, best_k]
    U_sel_out[i] = opt_in[i, best_k]

    if cnt > 0:
        meanH_out[i] = sumH / cnt
        meanU_out[i] = sumU / cnt
    else:
        meanH_out[i] = NAN
        meanU_out[i] = NAN


# =========================
# 実行ラッパ（Python）
# =========================

@dataclass
class CUDABuffers:
    # device arrays (step-updated)
    d_nx: any
    d_ny: any
    d_valid_mask: any
    d_indptr: any
    d_indices: any
    d_theta: any
    d_options: any
    d_radius2: any

    # device arrays (constants)
    d_edges: any

    # device arrays (outputs)
    d_opt: any
    d_pop: any
    d_selected: any

    # RNG
    rng_eval: any
    rng_choose: any

    # CSR capacity管理（Delaunay更新で nnz が変動するため）
    indices_cap: int
    h_indptr_pinned: any
    h_indices_pinned: any

    # ---- 追加: stats 用（必要時だけ遅延確保） ----
    d_H: any = None
    d_H_sel: any = None
    d_U_sel: any = None
    d_meanH: any = None
    d_meanU: any = None


def _ensure_int64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def _ensure_f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def init_cuda_buffers(state: dict, seed: int = 0) -> CUDABuffers:
    """state からGPU常駐バッファを作る。"""
    N = state["N"]
    K = state["option_number"]

    edges = _ensure_f64(state["percept"])
    if edges.size - 1 > 63:
        raise ValueError("division (edges.size-1) must be <= 63 to fit into int64 bitmask")

    nx = _ensure_f64(state["nx"])
    ny = _ensure_f64(state["ny"])
    valid_mask = np.asarray(state["valid_mask"], dtype=np.bool_)
    indptr = _ensure_int64(state["indptr"])
    indices = _ensure_int64(state["indices"])
    nnz = int(indices.size)
    theta = _ensure_f64(state["theta"])
    options = _ensure_f64(state["options"])
    radius2 = _compute_radius2_from_neighbors(state["position"], indptr, indices, state["V"])

    d_nx = cuda.to_device(nx)
    d_ny = cuda.to_device(ny)
    d_valid = cuda.to_device(valid_mask)

    h_indptr_pinned = cuda.pinned_array(indptr.shape, dtype=np.int64)
    np.copyto(h_indptr_pinned, indptr)

    indices_cap = max(1024, _next_pow2(nnz))
    h_indices_pinned = cuda.pinned_array((indices_cap,), dtype=np.int64)
    h_indices_pinned[:nnz] = indices

    d_indptr = cuda.to_device(h_indptr_pinned)
    d_indices = cuda.to_device(h_indices_pinned)
    d_theta = cuda.to_device(theta)
    d_options = cuda.to_device(options)
    d_radius2 = cuda.to_device(radius2)
    d_edges = cuda.to_device(edges)

    d_opt = cuda.device_array((N, K), dtype=np.int32)
    d_pop = cuda.device_array((N, K), dtype=np.int32)
    d_selected = cuda.device_array(N, dtype=np.float64)

    rng_eval = create_xoroshiro128p_states(N * K, seed=seed)
    rng_choose = create_xoroshiro128p_states(N, seed=seed ^ 0xA5A5A5A5)

    return CUDABuffers(
        d_nx=d_nx,
        d_ny=d_ny,
        d_valid_mask=d_valid,
        d_indptr=d_indptr,
        d_indices=d_indices,
        d_theta=d_theta,
        d_options=d_options,
        d_radius2=d_radius2,
        d_edges=d_edges,
        d_opt=d_opt,
        d_pop=d_pop,
        d_selected=d_selected,
        rng_eval=rng_eval,
        rng_choose=rng_choose,
        indices_cap=indices_cap,
        h_indptr_pinned=h_indptr_pinned,
        h_indices_pinned=h_indices_pinned,
    )


def sync_step_to_gpu(state: dict, bufs: CUDABuffers) -> None:
    """毎ステップ更新される配列をGPUへ反映。"""
    nx = _ensure_f64(state["nx"])
    ny = _ensure_f64(state["ny"])
    valid_mask = np.asarray(state["valid_mask"], dtype=np.bool_)

    indptr = _ensure_int64(state["indptr"])
    indices = _ensure_int64(state["indices"])

    theta = _ensure_f64(state["theta"])
    options = _ensure_f64(state["options"])

    radius2 = _compute_radius2_from_neighbors(state["position"], indptr, indices, state["V"])

    bufs.d_nx.copy_to_device(nx)
    bufs.d_ny.copy_to_device(ny)
    bufs.d_valid_mask.copy_to_device(valid_mask)

    np.copyto(bufs.h_indptr_pinned, indptr)
    bufs.d_indptr.copy_to_device(bufs.h_indptr_pinned)

    nnz = int(indices.size)
    if nnz > bufs.indices_cap:
        new_cap = _next_pow2(int(nnz * 1.25))
        bufs.indices_cap = new_cap
        bufs.h_indices_pinned = cuda.pinned_array((new_cap,), dtype=np.int64)
        bufs.d_indices = cuda.device_array((new_cap,), dtype=np.int64)

    bufs.h_indices_pinned[:nnz] = indices
    bufs.d_indices[:nnz].copy_to_device(bufs.h_indices_pinned[:nnz])

    bufs.d_theta.copy_to_device(theta)
    bufs.d_options.copy_to_device(options)
    bufs.d_radius2.copy_to_device(radius2)


def interaction_cuda(
    state: dict,
    bufs: CUDABuffers,
    sample_size: int = 100,
    TH: int = 30,
    compute_stats: bool = False,
) -> None:
    """
    GPUで selected_option を計算し、state["selected_option"] を更新。
    compute_stats=True のときだけ、以下を追加で state に入れる:
      state["H_sel"], state["U_sel"], state["meanH"], state["meanU"]
    """
    N = state["N"]
    K = state["option_number"]

    if K > 64:
        raise ValueError("option_number (K) must be <= 64 for ties local array. Increase ties array if needed.")
    if sample_size > 100:
        raise ValueError("sample_size must be <= 100 (fixed local array).")

    threads = (16, 8)
    blocks = (math.ceil(N / threads[0]), math.ceil(K / threads[1]))

    threads1 = 64
    blocks1 = math.ceil(N / threads1)

    if not compute_stats:
        kernel_eval_optpop[blocks, threads](
            bufs.d_nx,
            bufs.d_ny,
            bufs.d_valid_mask,
            bufs.d_indptr,
            bufs.d_indices,
            bufs.d_theta,
            bufs.d_options,
            bufs.d_radius2,
            float(state["rep"]),
            bufs.d_edges,
            int(sample_size),
            int(TH),
            bufs.rng_eval,
            bufs.d_opt,
            bufs.d_pop,
        )

        kernel_choose_selected[blocks1, threads1](
            bufs.d_valid_mask,
            bufs.d_options,
            bufs.d_opt,
            bufs.d_pop,
            bufs.rng_choose,
            bufs.d_selected,
        )

        state["selected_option"] = bufs.d_selected.copy_to_host()
        return

    # --- stats 必要時のみ遅延確保 ---
    if bufs.d_H is None:
        bufs.d_H = cuda.device_array((N, K), dtype=np.float32)
        bufs.d_H_sel = cuda.device_array(N, dtype=np.float32)
        bufs.d_U_sel = cuda.device_array(N, dtype=np.int32)
        bufs.d_meanH = cuda.device_array(N, dtype=np.float32)
        bufs.d_meanU = cuda.device_array(N, dtype=np.float32)

    kernel_eval_optpop_entropy[blocks, threads](
        bufs.d_nx,
        bufs.d_ny,
        bufs.d_valid_mask,
        bufs.d_indptr,
        bufs.d_indices,
        bufs.d_theta,
        bufs.d_options,
        bufs.d_radius2,
        float(state["rep"]),
        bufs.d_edges,
        int(sample_size),
        int(TH),
        bufs.rng_eval,
        bufs.d_opt,
        bufs.d_pop,
        bufs.d_H,
    )

    kernel_choose_selected_with_stats[blocks1, threads1](
        bufs.d_valid_mask,
        bufs.d_options,
        bufs.d_opt,
        bufs.d_pop,
        bufs.d_H,
        bufs.rng_choose,
        bufs.d_selected,
        bufs.d_H_sel,
        bufs.d_U_sel,
        bufs.d_meanH,
        bufs.d_meanU,
    )

    state["selected_option"] = bufs.d_selected.copy_to_host()
    state["H_sel"] = bufs.d_H_sel.copy_to_host()
    state["U_sel"] = bufs.d_U_sel.copy_to_host()
    state["meanH"] = bufs.d_meanH.copy_to_host()
    state["meanU"] = bufs.d_meanU.copy_to_host()


# =========================
# Public API（init/update）
# =========================

def init_swarm_state(
    N: int,
    L: float,
    R: float,
    division: int,
    option_number: int = 30,
    V: float = 8.0,
    kappa: float = 1.5,
    # kappa: float = 2.25,
    mu: float = 0.0,
    option_kappa: float = 2.0,
    # option_kappa: float = 1.75,
    seed: Optional[int] = None,
) -> Tuple[dict, CUDABuffers]:

    if seed is not None:
        np.random.seed(seed)

    state: dict = {}
    state["N"] = int(N)
    state["L"] = float(L)
    state["t"] = 0
    state["rep"] = float(R)
    state["division"] = int(division)

    state["percept"] = hetero.vonmises_hetero_angles(division, kappa=kappa, mu=mu)
    state["option_kappa"] = float(option_kappa)

    state["theta"] = 2 * np.pi * np.random.rand(state["N"])
    state["position"] = state["L"] * np.random.rand(2, state["N"])
    state["V"] = float(V)

    state["option_number"] = int(option_number)
    options = np.zeros((state["N"], state["option_number"]), dtype=np.float64)
    for i in range(state["N"]):
        options[i, 1:] = np.random.vonmises(0.0, 0.0, state["option_number"] - 1)
    state["options"] = options

    nx, ny = f_cpu.next_position_vectors_numba(state["position"], state["options"], state["theta"], state["V"])
    state["nx"], state["ny"] = nx, ny

    tri = Delaunay(state["position"].T)
    indptr, indices = tri.vertex_neighbor_vertices
    state["tri"], state["indptr"], state["indices"] = tri, indptr, indices

    state["valid_mask"] = _compute_valid_mask(state["nx"], state["ny"], state["L"])
    state["selected_option"] = np.zeros(state["N"], dtype=np.float64)

    bufs = init_cuda_buffers(state, seed=(seed or 0))
    state["update"] = lambda st: update(st, bufs)

    return state, bufs


def update(
    state: dict,
    bufs: CUDABuffers,
    sample_size: int = 100,
    TH: int = 30,
    profile: Optional[dict] = None,
    compute_stats: bool = False,
) -> None:
    """状態(dict)を1ステップ進める（in-place）。"""
    t0 = time.perf_counter()
    sync_step_to_gpu(state, bufs)
    t1 = time.perf_counter()

    interaction_cuda(state, bufs, sample_size=sample_size, TH=TH, compute_stats=compute_stats)
    cuda.synchronize()
    t2 = time.perf_counter()

    state["theta"] = (state["theta"] + state["selected_option"]) % (2 * np.pi)

    sel = state["selected_option"]
    th = state["theta"]
    V = state["V"]

    speed = np.exp(-(sel ** 2) / (2 * 1.5 ** 2))
    state["position"][0, :] += V * speed * np.cos(th)
    state["position"][1, :] += V * speed * np.sin(th)
    # state["position"][0, :] = state["position"][0, :] + V * np.cos(sel / 1.5) * np.cos(th)
    # state["position"][1, :] = state["position"][1, :] + V * np.cos(sel / 1.5) * np.sin(th)
    t3 = time.perf_counter()

    ok = float(state.get("option_kappa", 2.0))
    for i in range(state["N"]):
        state["options"][i, 1:] = np.random.vonmises(0.0, ok, state["option_number"] - 1)
    t4 = time.perf_counter()

    state["nx"], state["ny"] = f_cpu.next_position_vectors_numba(
        state["position"], state["options"], state["theta"], state["V"]
    )

    t5 = time.perf_counter()

    tri = Delaunay(state["position"].T)
    state["tri"] = tri
    state["indptr"], state["indices"] = tri.vertex_neighbor_vertices
    t6 = time.perf_counter()

    bsteps = state.get("boundary_steps", 40)

    if state["t"] < bsteps:
        state["valid_mask"] = _compute_valid_mask(state["nx"], state["ny"], state["L"])
    else:
        if state["t"] == bsteps:
            state["valid_mask"] = np.ones((state["N"], state["option_number"]), dtype=np.bool_)
    t7 = time.perf_counter()

    state["t"] += 1

    if profile is not None:
        profile.setdefault("sync", []).append(t1 - t0)
        profile.setdefault("gpu", []).append(t2 - t1)
        profile.setdefault("theta_pos", []).append(t3 - t2)
        profile.setdefault("options", []).append(t4 - t3)
        profile.setdefault("nxny", []).append(t5 - t4)
        profile.setdefault("delaunay", []).append(t6 - t5)
        profile.setdefault("mask", []).append(t7 - t6)


def run_trajectory(
    state: dict,
    bufs: CUDABuffers,
    T: int,
    save_every: int = 1,
    sample_size: int = 100,
    TH: int = 30,
) -> np.ndarray:
    if save_every <= 0:
        raise ValueError("save_every must be >=1")

    S = (T + save_every - 1) // save_every
    traj = np.empty((S, 2, state["N"]), dtype=np.float64)

    w = 0
    for t in range(T):
        update(state, bufs, sample_size=sample_size, TH=TH)
        if (t % save_every) == 0:
            traj[w, :, :] = state["position"]
            w += 1

    return traj


def run_and_record_profile(
    state: dict,
    bufs: CUDABuffers,
    T: int,
    record_every: int = 1,
    sample_size: int = 100,
    TH: int = 30,
    show_progress: bool = True,
    profile_every: int = 200,
    compute_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tステップ回しつつ、位置(position)と向き(theta)を記録し、実行時間も返す。"""
    if record_every <= 0:
        raise ValueError("record_every must be >= 1")

    N = state["N"]
    S = (T + record_every - 1) // record_every
    pos_hist = np.empty((S, 2, N), dtype=np.float64)
    theta_hist = np.empty((S, N), dtype=np.float64)

    # ---- 追加: stats履歴（必要時だけ確保） ----
    if compute_stats:
        H_sel_hist = np.empty((S, N), dtype=np.float32)
        U_sel_hist = np.empty((S, N), dtype=np.int32)
        meanH_hist = np.empty((S, N), dtype=np.float32)
        meanU_hist = np.empty((S, N), dtype=np.float32)
    else:
        H_sel_hist = None
        U_sel_hist = None
        meanH_hist = None
        meanU_hist = None

    prof: dict = {"sync": [], "gpu": [], "theta_pos": [], "options": [], "nxny": [], "delaunay": [], "mask": []}

    w = 0
    t0 = time.perf_counter()
    last_t = t0
    last_step = 0

    it = trange(T, desc="Sim", unit="step") if show_progress else range(T)

    for t in it:
        update(state, bufs, sample_size=sample_size, TH=TH, profile=prof, compute_stats=compute_stats)

        if (t % record_every) == 0:
            pos_hist[w, :, :] = state["position"]
            theta_hist[w, :] = state["theta"]

            if compute_stats:
                # update(..., compute_stats=True) で state に入っている前提
                H_sel_hist[w, :] = state["H_sel"]
                U_sel_hist[w, :] = state["U_sel"]
                meanH_hist[w, :] = state["meanH"]
                meanU_hist[w, :] = state["meanU"]

            w += 1

        if show_progress and profile_every > 0 and ((t + 1) % profile_every == 0):
            cuda.synchronize()
            now = time.perf_counter()
            dt = now - last_t
            steps = (t + 1) - last_step
            ms_per_step = (dt / max(1, steps)) * 1000.0
            elapsed = now - t0

            def _tail_mean_ms(key: str) -> float:
                arr = prof.get(key, [])
                if not arr:
                    return float("nan")
                tail = np.asarray(arr[-steps:], dtype=np.float64)
                return float(tail.mean() * 1000.0) if tail.size else float("nan")

            it.set_postfix_str(
                f"elapsed={elapsed:.1f}s ms/step={ms_per_step:.2f} "
                f"gpu={_tail_mean_ms('gpu'):.2f} sync={_tail_mean_ms('sync'):.2f} "
                f"del={_tail_mean_ms('delaunay'):.2f}"
            )

            last_t = now
            last_step = (t + 1)

    t1 = time.perf_counter()
    total = t1 - t0
    step = total / max(1, T)
    prof["total_sec"] = total
    prof["step_sec_mean"] = step
    prof["steps"] = T

    if not compute_stats:
        return pos_hist, theta_hist, prof
    return pos_hist, theta_hist, prof, (H_sel_hist, U_sel_hist, meanH_hist, meanU_hist)

def save_record_npz(
    path: str,
    pos_hist: np.ndarray,
    theta_hist: np.ndarray,
    L: float,
    record_every: int,
    # ---- stats は optional ----
    H_sel_hist: Optional[np.ndarray] = None,
    U_sel_hist: Optional[np.ndarray] = None,
    meanH_hist: Optional[np.ndarray] = None,
    meanU_hist: Optional[np.ndarray] = None,
) -> None:
    """
    replay用の記録データを npz 保存。

    stats が None でなければ一緒に保存する。
      - H_sel_hist: (S,N) float32
      - U_sel_hist: (S,N) int32
      - meanH_hist: (S,N) float32
      - meanU_hist: (S,N) float32
    """
    payload = dict(
        pos=pos_hist,
        theta=theta_hist,
        L=float(L),
        record_every=int(record_every),
    )

    # stats を「全部揃っている場合だけ」保存（半端に混ざる事故を防ぐ）
    has_stats = (
        (H_sel_hist is not None)
        and (U_sel_hist is not None)
        and (meanH_hist is not None)
        and (meanU_hist is not None)
    )

    payload["has_stats"] = np.array(has_stats, dtype=np.bool_)

    if has_stats:
        payload["H_sel"] = np.asarray(H_sel_hist, dtype=np.float32)
        payload["U_sel"] = np.asarray(U_sel_hist, dtype=np.int32)
        payload["meanH"] = np.asarray(meanH_hist, dtype=np.float32)
        payload["meanU"] = np.asarray(meanU_hist, dtype=np.float32)

    np.savez_compressed(path, **payload)


def load_record_npz(path: str):
    """
    戻り値:
      - stats無し: (pos, theta, L, record_every)
      - stats有り: (pos, theta, L, record_every, stats_dict)
          stats_dict = {"H_sel":..., "U_sel":..., "meanH":..., "meanU":...}
    """
    d = np.load(path, allow_pickle=False)

    pos = d["pos"]
    theta = d["theta"]
    L = float(d["L"])
    record_every = int(d["record_every"])

    has_stats = False
    if "has_stats" in d.files:
        has_stats = bool(d["has_stats"])
    else:
        # 旧形式互換: キーが揃ってたら stats あり扱い
        keys = set(d.files)
        has_stats = {"H_sel", "U_sel", "meanH", "meanU"}.issubset(keys)

    if not has_stats:
        return pos, theta, L, record_every

    stats = {
        "H_sel": d["H_sel"],
        "U_sel": d["U_sel"],
        "meanH": d["meanH"],
        "meanU": d["meanU"],
    }
    return pos, theta, L, record_every, stats


def make_replay_model(pos_hist: np.ndarray, theta_hist: np.ndarray, L: float):
    if pos_hist.ndim != 3 or pos_hist.shape[1] != 2:
        raise ValueError("pos_hist must be (S,2,N)")
    if theta_hist.ndim != 2:
        raise ValueError("theta_hist must be (S,N)")
    S = pos_hist.shape[0]
    if theta_hist.shape[0] != S:
        raise ValueError("pos_hist and theta_hist length mismatch")

    model = {
        "L": float(L),
        "position": pos_hist[0].copy(),
        "theta": theta_hist[0].copy(),
        "_frame": 0,
        "_S": S,
    }

    def _update(m):
        fidx = m["_frame"] + 1
        if fidx >= m["_S"]:
            return
        m["_frame"] = fidx
        m["position"][:, :] = pos_hist[fidx]
        m["theta"][:] = theta_hist[fidx]

    model["update"] = _update
    return model


def animate_record(
    pos_hist: np.ndarray,
    theta_hist: np.ndarray,
    L: float,
    out_path: str = "swarm.mp4",
    fps: int = 30,
    camera: str = "follow",
    view_size: float = 100.0,
    bounds: str = "none",
    dpi: int = 150,
):
    import matplotlib
    matplotlib.use('Agg')
    model = make_replay_model(pos_hist, theta_hist, L)
    interval = max(1, int(round(1000 / fps)))
    return f_cpu.animate_swarm(
        model,
        steps=pos_hist.shape[0] - 1,
        interval=interval,
        slow=1,
        mode="auto",
        save=True,
        save_path=out_path,
        save_format="auto",
        writer="auto",
        fps=fps,
        dpi=dpi,
        camera=camera,
        view_size=view_size,
        bounds=bounds,
        repeat=False,
    )


if __name__ == "__main__":
    from numba import cuda

    print("CUDA device count:", len(cuda.gpus))
    dev = cuda.get_current_device()
    print("Current device:", dev.id, dev.name.decode() if hasattr(dev.name, "decode") else dev.name)

    # ==== settings ====
    T = 100000
    record_every = 1
    Entropy = True   # ← statsも保存したいなら True、いらないなら False

    # ==== init ====
    state, bufs = init_swarm_state(N=200, L=150.0, V=6.0, R=1.0, option_number=60, division=40, seed=0)

    # ==== run ====
    if not Entropy:
        pos_hist, theta_hist, prof = run_and_record_profile(
            state, bufs,
            T=T,
            record_every=record_every,
            sample_size=100,
            TH=30,
            show_progress=True,
            profile_every=200,
            compute_stats=False,
        )

        rec_path = "rec.npz"
        save_record_npz(rec_path, pos_hist, theta_hist, L=state["L"], record_every=record_every)
        print(f"[saved record] {rec_path}  pos={pos_hist.shape} theta={theta_hist.shape}")

    else:
        pos_hist, theta_hist, prof, stats_hist = run_and_record_profile(state, bufs,
            T=T,
            record_every=record_every,
            sample_size=100,
            TH=30,
            show_progress=True,
            profile_every=200,
            compute_stats=True,
        )

        H_sel_hist, U_sel_hist, meanH_hist, meanU_hist = stats_hist

        rec_path = "rec_stats_300.npz"
        save_record_npz(
            rec_path,
            pos_hist, theta_hist,
            L=state["L"],
            record_every=record_every,
            H_sel_hist=H_sel_hist,
            U_sel_hist=U_sel_hist,
            meanH_hist=meanH_hist,
            meanU_hist=meanU_hist,
        )
        print(f"[saved record] {rec_path}  pos={pos_hist.shape} theta={theta_hist.shape} "
              f"H_sel={H_sel_hist.shape} U_sel={U_sel_hist.shape}")

    # ==== profile print ====
    print(f"[profile] total={prof['total_sec']:.3f} sec  mean_step={prof['step_sec_mean']*1000:.3f} ms/step  steps={prof['steps']}")

    # ==== 3) move_function_safe.animate_swarm で動画保存 ====
    # MP4保存には ffmpeg が必要です。
    # out = "swarm.mp4"
    # animate_record(
    #     pos_hist, theta_hist, L=state["L"],
    #     out_path=out,
    #     fps=30,
    #     camera="follow",
    #     view_size=400.0,
    #     bounds="none",
    #     dpi=150,
    # )