import numpy as np
import math

try:
    from numba import jit, int64, float64, njit
except Exception:  # numba may be unavailable or incompatible in some environments
    class _DummyNumbaType:
        def __getitem__(self, _):
            return self
        def __call__(self, *args, **kwargs):
            return self
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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from matplotlib.collections import LineCollection

# 固定閾値（必要なら好きに変えてOK）
_THRESHOLD_ENUM = 30

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from collections import deque
from pathlib import Path

def animate_swarm(
    model,
    steps=200,
    interval=50,
    slow=1,
    mode="auto",
    # ===== save options =====
    save=False,
    save_path="swarm.gif",
    save_format="auto",
    writer="auto",
    fps=None,
    dpi=150,
    repeat=True,
    # ===== camera options =====
    camera="fixed",            # "fixed" / "follow" / "follow_auto"
    view_size=10.0,            # follow時: 表示ウィンドウ一辺の長さ（任意）
    camera_smooth=0.25,        # 0〜1：重心追尾の平滑化
    # ===== bounds options =====
    bounds="box",              # "box" / "none"
    # box の時だけ有効
    camera_margin=0.0,         # クリップ時の余白
    # follow_auto 用
    auto_pad=1.5,
    auto_min=5.0,
    auto_max=None,
    zoom_smooth=0.2,
):
    """
    mode="auto"  : アニメーション（Spaceで一時停止）
    mode="manual": Enterキーで1ステップ進む

    camera:
        "fixed"       : 固定表示（boundsに応じて初期枠を決める）
        "follow"      : 重心追尾（固定視野サイズ）
        "follow_auto" : 重心追尾 + 群れサイズに応じて自動ズーム

    bounds:
        "box"  : 0..L の箱（壁反射モデルなど）
        "none" : 無限平面（境界なし）
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import LineCollection
    from collections import deque
    from pathlib import Path
    from numba import jit, njit, int64, float64, types
    from numba.typed import Dict, List

    # --- allow both object-style and dict-style models ---
    if isinstance(model, dict):
        _get_pos = lambda: model["position"]
        _get_theta = lambda: model["theta"]
        _get_L = lambda: model["L"]
        _update = lambda: model["update"](model)
    else:
        _get_pos = lambda: _get_pos()
        _get_theta = lambda: _get_theta()
        _get_L = lambda: _get_L()
        _update = lambda: _update()


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True)

    # ==== 初期データ ====
    x = _get_pos()[0, :]
    y = _get_pos()[1, :]

    def clamp_window(x0, x1, y0, y1):
        """bounds='box' のときに、表示窓を [0,L] に押し戻す"""
        L = _get_L()
        xmin, xmax = 0 + camera_margin, L - camera_margin
        ymin, ymax = 0 + camera_margin, L - camera_margin

        w = x1 - x0
        h = y1 - y0

        # 窓が箱より大きいなら箱に合わせる
        if w >= (xmax - xmin):
            x0, x1 = xmin, xmax
        else:
            if x0 < xmin:
                x1 += (xmin - x0); x0 = xmin
            if x1 > xmax:
                x0 -= (x1 - xmax); x1 = xmax

        if h >= (ymax - ymin):
            y0, y1 = ymin, ymax
        else:
            if y0 < ymin:
                y1 += (ymin - y0); y0 = ymin
            if y1 > ymax:
                y0 -= (y1 - ymax); y1 = ymax

        return x0, x1, y0, y1

    # 初期枠（fixedのとき）
    if camera == "fixed":
        if bounds == "box":
            ax.set_xlim(0, _get_L())
            ax.set_ylim(0, _get_L())
        else:
            # 無限平面：とりあえず初期視野を view_size にする（見えない問題を避ける）
            half = 0.5 * float(view_size)
            cx, cy = x.mean(), y.mean()
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

    scat = ax.scatter(x, y, color="black")

    arrow_len = 0.25
    u = arrow_len * np.cos(_get_theta())
    v = arrow_len * np.sin(_get_theta())
    quiv = ax.quiver(
        x, y, u, v,
        angles="xy", scale_units="xy", scale=1,
        color="red"
    )

    history = deque(maxlen=4)
    history.append(_get_pos().copy())

    trail_lc = LineCollection([], colors="blue", linewidths=0.8, alpha=0.7)
    ax.add_collection(trail_lc)

    # ==== camera state ====
    cam_center = np.array([x.mean(), y.mean()], dtype=float)
    cam_size = float(view_size)

    def redraw():
        nonlocal cam_center, cam_size

        x = _get_pos()[0, :]
        y = _get_pos()[1, :]

        scat.set_offsets(np.c_[x, y])

        u = arrow_len * np.cos(_get_theta())
        v = arrow_len * np.sin(_get_theta())
        quiv.set_offsets(np.c_[x, y])
        quiv.set_UVC(u, v)

        m = len(history)
        if m >= 2:
            arr_t = np.stack(list(history), axis=0)      # (m, 2, N)
            segments = arr_t.transpose(2, 0, 1)          # (N, m, 2)
            trail_lc.set_segments(segments)
        else:
            trail_lc.set_segments([])

        # ==== camera follow ====
        if camera != "fixed":
            # 重心
            target_center = np.array([x.mean(), y.mean()], dtype=float)
            cam_center = (1 - camera_smooth) * cam_center + camera_smooth * target_center

            # 視野サイズ
            if camera == "follow":
                target_size = float(view_size)

            elif camera == "follow_auto":
                w = (x.max() - x.min()) + 2 * auto_pad
                h = (y.max() - y.min()) + 2 * auto_pad
                target_size = max(w, h, auto_min)
                if auto_max is not None:
                    target_size = min(target_size, auto_max)

                cam_size = (1 - zoom_smooth) * cam_size + zoom_smooth * target_size
                target_size = cam_size
            else:
                raise ValueError("camera must be 'fixed' / 'follow' / 'follow_auto'")

            half = 0.5 * target_size
            x0, x1 = cam_center[0] - half, cam_center[0] + half
            y0, y1 = cam_center[1] - half, cam_center[1] + half

            if bounds == "box":
                x0, x1, y0, y1 = clamp_window(x0, x1, y0, y1)
            elif bounds == "none":
                pass
            else:
                raise ValueError("bounds must be 'box' or 'none'")

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        fig.canvas.draw_idle()

    # ===========================
    # 自動アニメーション
    # ===========================
    if mode == "auto":
        paused = {"value": False}

        def toggle_pause(event):
            if event.key == " ":
                paused["value"] = not paused["value"]
                print("Paused" if paused["value"] else "Resumed")

        fig.canvas.mpl_connect("key_press_event", toggle_pause)

        def update(frame):
            if paused["value"]:
                return scat, quiv, trail_lc

            for _ in range(slow):
                _update()
                history.append(_get_pos().copy())

            redraw()
            return scat, quiv, trail_lc

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=interval,
            blit=False,
        )

        # ===== 保存処理 =====
        if save:
            from pathlib import Path
            out = Path(save_path)

            fmt = save_format.lower()
            if fmt == "auto":
                suffix = out.suffix.lower()
                if suffix in [".gif", ".mp4"]:
                    fmt = suffix[1:]
                else:
                    fmt = "gif"
                    out = out.with_suffix(".gif")

            if fps is None:
                fps = max(1, int(round(1000 / interval)))

            w = writer.lower()
            if w == "auto":
                w = "pillow" if fmt == "gif" else "ffmpeg"

            out.parent.mkdir(parents=True, exist_ok=True)

            try:
                if fmt == "gif":
                    ani.save(
                        str(out),
                        writer="pillow",
                        fps=fps,
                        dpi=dpi,
                        savefig_kwargs={"facecolor": "white"},
                    )
                elif fmt == "mp4":
                    ani.save(
                        str(out),
                        writer="ffmpeg",
                        fps=fps,
                        dpi=dpi,
                        savefig_kwargs={"facecolor": "white"},
                    )
                else:
                    raise ValueError(f"Unsupported save_format: {fmt}")

                print(f"[saved] {out.resolve()}")

            except Exception as e:
                print(f"[save failed] {e}")
                print("Hint: GIFなら pillow, MP4なら ffmpeg が必要です。")

        plt.show()
        return ani

    # ===========================
    # 手動モード
    # ===========================
    elif mode == "manual":
        def on_key(event):
            if event.key == "enter":
                _update()
                history.append(_get_pos().copy())
                redraw()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.title("Press ENTER: step by step")
        plt.show()
        return None

    else:
        raise ValueError("mode must be 'auto' or 'manual'")




@njit((int64[:], int64[:], int64), cache=True)
def group_unique_and_sample(rows, cols, size):
    n = rows.size
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.int64)

    # ---- unique_with_counts ----
    unique_vals = np.empty(n, dtype=np.int64)
    counts = np.empty(n, dtype=np.int64)

    current_val = rows[0]
    cnt = 1
    idx = 0

    for i in range(1, n):
        if rows[i] == current_val:
            cnt += 1
        else:
            unique_vals[idx] = current_val
            counts[idx] = cnt
            idx += 1
            current_val = rows[i]
            cnt = 1

    unique_vals[idx] = current_val
    counts[idx] = cnt
    G = idx + 1  # グループ数

    # ---- cumulative_sum ----
    cumulative_sum = np.empty(G, dtype=np.int64)
    cumulative_sum[0] = 0
    for g in range(1, G):
        cumulative_sum[g] = cumulative_sum[g - 1] + counts[g - 1]

    # ---- starts, widths ----
    starts = np.empty(G, dtype=np.int64)
    widths = np.empty(G, dtype=np.int64)

    for g in range(G):
        starts[g] = cumulative_sum[g]
        end = cumulative_sum[g + 1] if (g + 1) < G else cols.size
        widths[g] = end - starts[g]

    # ---- 直積サイズ（オーバーフローしないように判定だけ行う）----
    TH = 30  # 全列挙の閾値

    T = 1
    small = True
    for g in range(G):
        w = widths[g]

        # 念のため 0 以下は列挙対象外（データがおかしい場合）
        if w <= 0:
            small = False
            break

        # これ以上掛けたら TH を超えるなら、小規模ではないと判定して打ち切り
        if T > TH // w:
            small = False
            break

        T *= w

    # ---- 小規模は全列挙 ----
    if small:
        out = np.empty((T, G), dtype=np.int64)
        digits = np.empty(G, dtype=np.int64)

        for t in range(T):
            L = t
            for g in range(G):
                r = widths[g]
                q = L // r
                d = L - q * r
                digits[g] = d
                L = q

            for g in range(G):
                out[t, g] = cols[starts[g] + digits[g]]

        return unique_vals[:G], out

    # ---- 大規模はランダム with-replacement ----
    out = np.empty((size, G), dtype=np.int64)
    for s in range(size):
        for g in range(G):
            w = widths[g]
            # ここも安全のためチェック
            if w <= 0:
                out[s, g] = -1  # or 0, or continue, etc.
            else:
                out[s, g] = cols[starts[g] + np.random.randint(0, w)]

    return unique_vals[:G], out



@njit(int64(float64[:, :], float64[:, :], int64[:], int64[:], float64, int64), cache=True)
def compute_rep_mask_with_arg_minus(X, Y, rep_rows, rep_cols, theta, division):
    """
    X, Y : 2D 配列（相対座標）
    rep_rows, rep_cols : rep領域の index
    theta : 引く角度
    division : セクション分割数

    return : rep_mask (int64)
    """

    two_pi = 2.0 * np.pi
    pi = np.pi
    section_width = two_pi / division

    rep_mask = 0

    nrep = rep_rows.size
    if nrep == 0:
        return 0

    for k in range(nrep):
        r = rep_rows[k]
        c = rep_cols[k]

        xx = X[r, c]
        yy = Y[r, c]

        # 基本角度
        ang = np.arctan2(yy, xx)

        # --- arg_minus_matrix と同じ wrap（[-π, π)） ---
        d = ang - theta
        d = (d + np.pi) % two_pi - np.pi

        # wrap to [0, 2π) に再変換
        if d < 0.0:
            d += two_pi

        # section index
        idx = int(d / section_width)

        # safety
        if idx >= division:
            idx = division - 1
        elif idx < 0:
            idx = 0

        # rep_mask に追加
        rep_mask |= (1 << idx)

    return rep_mask


@njit(int64(float64, int64), cache=True)
def angle_to_section(angle, num_sections):
    """
    angle を 0〜2π の範囲に正規化し、
    num_sections 分割の中の何番目の区間か（0〜num_sections-1）を返す。
    """
    two_pi = 2 * np.pi
    section_width = two_pi / num_sections

    # 高速で安全な wrap
    angle = angle % two_pi
    if angle < 0:  # % が負を返す場合の保険
        angle += two_pi

    idx = int(angle / section_width)

    # float 誤差で idx == num_sections になるのを防ぐ
    if idx >= num_sections:
        idx = num_sections - 1

    return idx


@jit(int64(int64[:], int64[:]), nopython=True)
def choose_index(opt, pup):
    # Step 1: Find the maximum value in opt and its indices
    max_opt_value = np.max(opt)
    max_opt_indices = np.where(opt == max_opt_value)[0]

    if len(max_opt_indices) > 1:
        # Step 2: From these, find the indices with the maximum value in pup
        max_pup_value = np.max(pup[max_opt_indices])
        max_pup_indices = max_opt_indices[np.where(pup[max_opt_indices] == max_pup_value)[0]]

        if len(max_pup_indices) > 1:
            # Step 3: If still tied, choose randomly among these indices
            chosen_index = max_pup_indices[np.random.randint(0, len(max_pup_indices))]
        else:
            chosen_index = max_pup_indices[0]
    else:
        chosen_index = max_opt_indices[0]

    return chosen_index


def next_position_vectors(position, vir, interval):
    X = np.repeat(position[0], vir.shape[1])
    Y = np.repeat(position[1], vir.shape[1])
    random_intervals = 2* interval + (np.random.rand(len(X)) - 0.5) * interval * 0.10  # ±20% の範囲

    U = random_intervals * np.cos(vir).flatten()
    V = random_intervals * np.sin(vir).flatten()

    x = X + U
    y = Y + V

    return x.reshape(vir.shape), y.reshape(vir.shape)

@njit(cache=True, fastmath=True)
def next_position_vectors_numba(position_2N, angles, theta, interval):
    # position_2N: (2, N)
    # angles: (N, K)
    N, K = angles.shape
    x = np.empty((N, K), dtype=np.float64)
    y = np.empty((N, K), dtype=np.float64)

    for i in range(N):
        x0 = position_2N[0, i]
        y0 = position_2N[1, i]
        th = theta[i]

        for k in range(K):
            # step =  interval + (np.random.random() - 0.5) * interval * 0
            #step = interval + np.cos(angles[i, k]) * 0.1
            ang = angles[i, k] + th
            x[i, k] = x0 + interval * np.cos(ang)
            y[i, k] = y0 + interval * np.sin(ang)

    return x, y

@njit(cache=True)
def next_position_vectors_numba_turncost(position, options, theta, V, sigma):
    N = position.shape[1]
    K = options.shape[1]
    nx = np.empty((N, K), dtype=np.float64)
    ny = np.empty((N, K), dtype=np.float64)

    denom = 2.0 * sigma * sigma

    for i in range(N):
        x0 = position[0, i]
        y0 = position[1, i]
        th0 = theta[i]
        for k in range(K):
            phi = options[i, k]
            th = th0 + phi
            speed = math.exp(-(phi * phi) / denom)
            nx[i, k] = x0 + V * speed * math.cos(th)
            ny[i, k] = y0 + V * speed * math.sin(th)

    return nx, ny



def arg_plus(theta, ind_theta):
    ind_theta = np.array(ind_theta)[:, None]  # reshape ind_theta to be a column vector
    return np.arctan2(np.sin(theta + ind_theta), np.cos(theta + ind_theta))


@jit(int64[:, :](int64[:], int64, int64), nopython=True)
def randint_array(data, high, size):
    result = np.empty((high, size), dtype=np.int64)  # 型を明示して空の配列を初期化
    for i in range(high):
        for j in range(size):
            result[i, j] = np.random.randint(0, data[j])  # 乱数を範囲内で生成
    return result


@jit('f8[:, :](f8[:, :], f8[:, :])', nopython=True)
def distance_matrix(X, D):
    n = X.shape[1]
    m = X.shape[0]

    for i in range(n):
        for j in range(i):
            d = 0.0
            for k in range(m):
                # 差を計算
                tmp = X[k, j] - X[k, i]
                d += tmp * tmp

            D[i, j] = np.sqrt(d)
            D[j, i] = D[i, j]
    return D


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
            d = group_theta[i, j] - theta
            # 完全版 wrap（任意の角度に対応）
            d = (d + np.pi) % two_pi - np.pi
            diff_matrix[i, j] = d
    return diff_matrix



@njit((float64[:, :], int64, int64), cache=True)
def radians_patterns_to_binary_rep(radians_patterns, num_sections, rep_mask):

    section_width = (2 * np.pi) / num_sections
    num_patterns, num_angles = radians_patterns.shape

    keep_mask = ~rep_mask

    binary_results = np.zeros((num_patterns, num_sections), dtype=np.int64)
    result_count = 0

    seen = {}

    for pattern_idx in range(num_patterns):
        binary_representation = 0

        for angle_idx in range(num_angles):
            angle = radians_patterns[pattern_idx, angle_idx]

            # ★ 完全 wrap
            angle = (angle + 2*np.pi) % (2*np.pi)

            section_index = int(angle / section_width)
            binary_representation |= (1 << section_index)

        # ★ rep_mask 適用
        binary_representation &= keep_mask

        # ★ 全ゼロは除外（これが必須）
        if binary_representation == 0:
            continue

        # ★ unique check
        if binary_representation not in seen:
            seen[binary_representation] = 1
            for s in range(num_sections):
                binary_results[result_count, s] = (binary_representation >> s) & 1
            result_count += 1

    return binary_results[:result_count]




## ----- 確認用の関数なのでプログラムでは使わない

# @njit((float64[:, :], int64), cache=True)
# def radians_patterns_to_binary(radians_patterns, num_sections):
#     num_patterns, num_angles = radians_patterns.shape
#     unique_map = {}   # Numba 0.60 なら普通の dict が使える
#     max_results_size = num_patterns
#
#     binary_results = np.zeros((max_results_size, num_sections), dtype=np.int64)
#     result_count = 0
#
#     for pattern_idx in range(num_patterns):
#
#         # この pattern のビット表現
#         binary_representation = 0
#
#         for angle_idx in range(num_angles):
#             angle = radians_patterns[pattern_idx, angle_idx]
#
#             # 角度からセクションへ
#             section_index = angle_to_section(angle, num_sections)
#
#             # ビットを立てる
#             binary_representation |= (1 << section_index)
#
#         # すでに出ているパターンか？
#         if binary_representation not in unique_map:
#             unique_map[binary_representation] = 1
#
#             # ビット → 行列形式へ
#             for i in range(num_sections):
#                 binary_results[result_count, i] = (binary_representation >> i) & 1
#
#             result_count += 1
#
#     return binary_results[:result_count]

def int_to_bits(val, num_bits):
    bits = np.zeros(num_bits, dtype=np.int64)
    for i in range(num_bits):
        bits[i] = (val >> i) & 1
    return bits


def bits_to_int(bits):
    # bits[0] が MSB、bits[-1] が LSB として扱う
    powers = 1 << np.arange(bits.size - 1, -1, -1)
    return int(bits @ powers)


def unique_check(result, rep_mask):
    # 結果をコピー（破壊しないように）
    arr = result.copy()

    # rep_mask のビット 1 の部分を強制的に 0 にする
    mask_bits = int_to_bits(rep_mask, arr.shape[1])
    for i in range(arr.shape[0]):
        arr[i] = arr[i] * (1 - mask_bits)  # 1の位置が消える

    # 0 行を削除（Numba版と同じ）
    nonzero = arr.sum(axis=1) > 0
    arr = arr[nonzero]

    # 重複の確認
    int_data = np.array([bits_to_int(row) for row in arr])

    vals, counts = np.unique(int_data, return_counts=True)
    dup_vals = vals[counts > 1]

    return len(vals)