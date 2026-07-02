import numpy as np
from scipy.stats import vonmises
from numba import jit, njit, int64, float64, types
from numba.typed import Dict

try:
    from numba import jit, int64, float64, njit, types

except Exception:  # numba may be unavailable or incompatible
    class _DummyNumbaType:
        def __getitem__(self, _):
            return self
        def __call__(self, *args, **kwargs):
            return self

    class _DummyTypes:
        int64 = _DummyNumbaType()
        float64 = _DummyNumbaType()

    class _DummyDict(dict):
        @classmethod
        def empty(cls, **kwargs):
            return {}

    def jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def _decorator(fn):
            return fn
        return _decorator

    def njit(*args, **kwargs):
        return jit(*args, **kwargs)

    int64 = _DummyNumbaType()
    float64 = _DummyNumbaType()
    types = _DummyTypes()
    Dict = _DummyDict



def vonmises_hetero_angles(num_sections, kappa=1.0, mu=0.0):
    ## kappaは1.5以上だとほぼ意味をなさないのでそれ以下にすること
    ## kappaが0の時は完全に一様分布になる
    theta = np.linspace(-np.pi, np.pi, num_sections + 1)

    # von Mises の CDF をそのまま評価（正規化済み）
    # cdf ∈ [0,1]
    cdf = vonmises.cdf(theta, kappa, loc=mu)

    # CDF を [-π, π] に線形マップ（元コードと同じ操作）
    angles = -np.pi + 2 * np.pi * cdf

    return angles


# @njit(cache=True)
def compute_rep_mask_scan_xy(X, Y, rep2, theta, edges):
    two_pi = 2.0 * np.pi
    pi = np.pi

    n_bins = edges.size - 1
    mask_all = (1 << n_bins) - 1

    rep_mask = 0
    n, m = X.shape

    for r in range(n):
        for c in range(m):
            xx = X[r, c]
            yy = Y[r, c]

            if xx*xx + yy*yy >= rep2:
                continue

            ang = np.arctan2(yy, xx)

            # あなたの仕様：ang - theta - pi を wrap して [-pi, pi)
            d = ang - theta - pi
            d = (d + pi) % two_pi - pi

            if d >= pi:        # 誤差で +pi 側に出たら -pi 側へ
                d -= two_pi

            idx = np.searchsorted(edges, d, side='right') - 1

            # ここは基本的に idx は [0, n_bins-1] に収まるはずだが保険
            if idx < 0:
                idx = 0
            elif idx >= n_bins:
                idx = n_bins - 1

            rep_mask |= (1 << idx)
            if rep_mask == mask_all:
                return rep_mask

    return rep_mask


@njit(int64(float64[:, :], float64[:, :], int64[:], int64[:], float64, float64[:]), cache=True)
def compute_rep_mask_with_arg_minus(X, Y, rep_rows, rep_cols, theta, edges):
    """
    X, Y      : 2D 配列（相対座標）
    rep_rows, rep_cols : rep領域の index
    theta     : 引く角度
    edges     : セクション境界の角度配列（単調増加）
                例: [-pi, ..., pi]  長さ = n_bins + 1

    return   : rep_mask (int64)
    """

    two_pi = 2.0 * np.pi
    pi = np.pi

    rep_mask = 0

    nrep = rep_rows.size
    if nrep == 0:
        return 0

    # ビン数（セクション数）
    n_bins = edges.size - 1


    for k in range(nrep):
        r = rep_rows[k]
        c = rep_cols[k]

        xx = X[r, c]
        yy = Y[r, c]

        # 基本角度（[-pi, pi]）
        ang = np.arctan2(yy, xx)

        # --- arg_minus と同じ wrap（[-π, π)） ---
        d = ang - theta - pi
        d = (d + pi) % two_pi - pi  # [-pi, pi)

        # ------------------------------------
        # ここでは d を [-pi, pi) に保ったまま、edges も [-pi, pi] 前提で使う

        # どのビンか探す
        idx = 0
        found = False
        for i in range(n_bins):
            left = edges[i]
            right = edges[i + 1]

            # 通常の区間: [left, right)
            if d >= left and d < right:
                idx = i
                found = True
                break

        # 端の丸め誤差などで見つからなかった場合:
        if not found:
            if d < edges[0]:
                idx = 0
            else:
                idx = n_bins - 1

        # rep_mask に追加
        rep_mask |= (1 << idx)

    return rep_mask

@njit(float64[:, :](float64, float64[:, :]), cache=True)
def arg_minus_matrix(theta, group_theta):
    """
    group_theta + theta の角度差を [-π, π) に正規化して返す。
    """
    n, m = group_theta.shape
    diff_matrix = np.empty((n, m), dtype=np.float64)
    two_pi = 2 * np.pi
    for i in range(n):
        for j in range(m):
            d = group_theta[i, j] - theta - np.pi
            # 完全版 wrap（任意の角度に対応）
            d = (d + np.pi) % two_pi - np.pi
            diff_matrix[i, j] = d
    return diff_matrix

@njit((float64[:, :], float64[:], int64), cache=True)
def radians_patterns_to_binary_rep_varbins(radians_patterns, edges, rep_mask):
    """
    radians_patterns : (num_patterns, num_angles) の角度 [rad]
    edges            : セクション境界の角度配列（[-π, π] の単調増加, 長さ = num_sections+1）
    rep_mask         : ビットマスク（除外したいセクションを1にしたもの）

    戻り値:
        binary_results : shape = (unique_count, num_sections)
                         各行がそのパターンのビット表現（0/1 配列）
    """

    two_pi = 2.0 * np.pi
    pi = np.pi

    num_patterns, num_angles = radians_patterns.shape
    num_sections = edges.size - 1  # ビン数

    keep_mask = ~rep_mask

    binary_results = np.zeros((num_patterns, num_sections), dtype=np.int64)
    result_count = 0

    # ★ numba.typed.Dict で型付き dict を作る（key/value とも int64）
    seen = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )

    for pattern_idx in range(num_patterns):
        binary_representation = 0

        for angle_idx in range(num_angles):
            angle = radians_patterns[pattern_idx, angle_idx]

            # 角度を [-π, π) に wrap
            angle = (angle + pi) % two_pi - pi

            # どの区間か探索（線形探索版）
            idx = 0
            found = False
            for s in range(num_sections):
                left = edges[s]
                right = edges[s + 1]
                if angle >= left and angle < right:
                    idx = s
                    found = True
                    break

            # 数値誤差などで見つからなかった場合の保険
            if not found:
                if angle < edges[0]:
                    idx = 0
                else:
                    idx = num_sections - 1

            # ビットを立てる
            binary_representation |= (1 << idx)

        # rep_mask 適用（除外セクションは0にする）
        binary_representation &= keep_mask

        # 全ゼロは除外
        if binary_representation == 0:
            continue

        # ★ typed.Dict で unique check
        if binary_representation not in seen:
            seen[binary_representation] = 1  # 値はなんでもよい（ここではダミーで 1）

            # ビット → 0/1 配列に展開
            for s in range(num_sections):
                binary_results[result_count, s] = (binary_representation >> s) & 1

            result_count += 1

    return binary_results[:result_count]

@njit(cache=True)
def popcount_u64(x):
    # int64でもOKだけど、右シフトの挙動が怖い時は符号なしに寄せる
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c

@njit((float64[:, :], float64[:], int64), cache=True)
def radians_patterns_to_optpop_varbins(radians_patterns, edges, rep_mask):
    """
    戻り値:
      opt_count: ユニークなビット表現の数（len(result)と同等）
      pop_sum  : ユニーク表現の1の総和（np.sum(result)と同等）
    """
    two_pi = 2.0 * np.pi
    pi = np.pi

    num_patterns, num_angles = radians_patterns.shape
    num_sections = edges.size - 1

    # ★ keep_mask をビン数に制限
    mask_all = (1 << num_sections) - 1
    keep_mask = (~rep_mask) & mask_all

    # ★ typed.Dict でユニークチェック（値はダミー）
    seen = Dict.empty(key_type=types.int64, value_type=types.uint8)

    opt_count = 0
    pop_sum = 0

    for pattern_idx in range(num_patterns):
        bits = 0

        for angle_idx in range(num_angles):
            angle = radians_patterns[pattern_idx, angle_idx]
            angle = (angle + pi) % two_pi - pi

            # ★ searchsorted（right）で区間を探す： idx in [0, num_sections]
            idx = np.searchsorted(edges, angle, side='right') - 1
            if idx < 0:
                idx = 0
            elif idx >= num_sections:
                idx = num_sections - 1

            bits |= (1 << idx)

        bits &= keep_mask
        if bits == 0:
            continue

        if bits not in seen:
            seen[bits] = 1
            opt_count += 1

            # popcount を足す（np.sum(result) と一致）
            # もし環境が対応してれば bits.bit_count() の方が速い
            pop_sum += popcount_u64(bits)

    return opt_count, pop_sum


@njit((float64[:, :], float64[:, :], float64, float64[:], int64), cache=True)
def xx_yy_to_optpop_varbins(XX, YY, base_theta, edges, rep_mask):
    """
    XX,YY      : (num_patterns, num_angles)
    base_theta : スカラー（self.theta[index] + self.options[index, options]）
    edges      : ビン境界 [-pi, pi] 単調増加, 長さ = num_sections+1
    rep_mask   : 除外セクションbit=1 の int64

    戻り値:
      opt_count : ユニークなビット表現の数
      pop_sum   : ユニーク表現の1の総数（popcountの合計）
    """
    two_pi = 2.0 * np.pi
    pi = np.pi

    num_patterns, num_angles = XX.shape
    num_sections = edges.size - 1

    # keep_mask の上位ビット暴発を防ぐ
    mask_all = (1 << num_sections) - 1
    keep_mask = (~rep_mask) & mask_all

    seen = Dict.empty(key_type=types.int64, value_type=types.int64)

    opt_count = 0
    pop_sum = 0

    for p in range(num_patterns):
        bits = 0

        for a in range(num_angles):
            # 角度差をその場で作る（diff_matrix不要）
            ang = np.arctan2(YY[p, a], XX[p, a]) - base_theta - pi

            # [-pi, pi) wrap
            ang = (ang + pi) % two_pi - pi

            # ビン割当（searchsorted）
            idx = np.searchsorted(edges, ang, side='right') - 1
            if idx < 0:
                idx = 0
            elif idx >= num_sections:
                idx = num_sections - 1

            bits |= (1 << idx)

        bits &= keep_mask
        if bits == 0:
            continue

        if bits not in seen:
            seen[bits] = 1
            opt_count += 1
            pop_sum += popcount_u64(bits)

    return opt_count, pop_sum



# @njit((float64[:, :], float64[:], int64), cache=True)
# def radians_patterns_to_binary_rep(radians_patterns, edges, rep_mask):
#     """
#     radians_patterns : (num_patterns, num_angles) の角度 [rad]
#     edges            : セクション境界の角度配列（単調増加, [-π, π], 長さ = num_sections+1）
#     rep_mask         : ビットマスク（除外したいセクションを1にしたもの）
#
#     戻り値:
#         binary_results : shape = (unique_count, num_sections)
#                          各行がそのパターンのビット表現（0/1 配列）
#     """
#
#     two_pi = 2.0 * np.pi
#     pi = np.pi
#
#     num_patterns, num_angles = radians_patterns.shape
#     num_sections = edges.size - 1  # ビン数
#
#     keep_mask = ~rep_mask
#
#     binary_results = np.zeros((num_patterns, num_sections), dtype=np.int64)
#     result_count = 0
#
#     # NOTE:
#     # Numba の nopython モードで高速にしたいなら、
#     # numba.typed.Dict を使うのが推奨。
#     seen = {}
#
#     for pattern_idx in range(num_patterns):
#         binary_representation = 0
#
#         for angle_idx in range(num_angles):
#             angle = radians_patterns[pattern_idx, angle_idx]
#
#             # 角度を [-π, π) に wrap
#             angle = (angle + pi) % two_pi - pi
#
#             # どの区間か探索
#             idx = 0
#             found = False
#             for s in range(num_sections):
#                 left = edges[s]
#                 right = edges[s + 1]
#                 if angle >= left and angle < right:
#                     idx = s
#                     found = True
#                     break
#
#             # 数値誤差などで見つからない場合の保険
#             if not found:
#                 if angle < edges[0]:
#                     idx = 0
#                 else:
#                     idx = num_sections - 1
#
#             # ビットを立てる
#             binary_representation |= (1 << idx)
#
#         # rep_mask 適用（除外セクションは0にする）
#         binary_representation &= keep_mask
#
#         # 全ゼロは除外
#         if binary_representation == 0:
#             continue
#
#         # unique check
#         if binary_representation not in seen:
#             seen[binary_representation] = 1
#
#             # ビット → 0/1 配列に展開
#             for s in range(num_sections):
#                 binary_results[result_count, s] = (binary_representation >> s) & 1
#
#             result_count += 1
#
#     return binary_results[:result_count]