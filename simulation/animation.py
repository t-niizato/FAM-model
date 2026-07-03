import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

def animate_positions(
    pos,                       # (T,2,N)
    theta=None,
    interval=33,
    trail=25,

    # ===== save (manual recording) =====
    record_key="v",            # ★ 録画トグルキー
    record_path="swarm.mp4",
    record_fps=None,
    record_dpi=150,

    # ===== camera =====
    camera="follow_auto",      # "fixed" / "follow" / "follow_auto"
    view_size=10.0,
    camera_smooth=0.20,
    auto_pad=1.5,
    auto_min=5.0,
    auto_max=None,
    zoom_smooth=0.10,          # ★(B) 少し下げた（0.15→0.10）
    camera_every=1,
    window_smooth=0.12,

    # ===== key controls =====
    enable_keys=True,
    zoom_step=1.1,
    zoom_mult_min=0.2,
    zoom_mult_max=5.0,

    # ===== visuals =====
    dark_style=True,
    point_size=55,
    edge_lw=0.35,              # ★(A) 少し細め
    cmap="turbo",
    trail_lw=1.6,
    trail_alpha=0.55,
    speed_percentile=99,

    # ===== (D) glow =====
    glow=True,
    glow_mult= 3,
    glow_alpha=0.4,

    # ===== (C) title =====
    title_alpha=0.35,          # ★薄く
    hide_title_while_recording=True,  # ★録画中だけ消す
):
    """
    keys:
      Space : pause/resume
      Enter : 1 step (paused中)
      Up/Down : zoom in/out
      R : zoom reset
      Left/Right : speed
      S : speed reset
      V : record start/stop (toggle)
      Q/Esc : quit
    """

    pos = np.asarray(pos)
    if pos.ndim != 3 or pos.shape[1] != 2:
        raise ValueError("pos must be (T,2,N)")
    T, _, N = pos.shape

    if theta is not None:
        theta = np.asarray(theta)
        if theta.shape != (T, N):
            raise ValueError("theta must be (T,N)")

    # ========= figure =========
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    if dark_style:
        bg = "#0b1020"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # ===== subtle vignette (adds cinematic focus) =====
        vN = 256
        yy, xx = np.mgrid[-1:1:complex(0, vN), -1:1:complex(0, vN)]
        rr = np.sqrt(xx * xx + yy * yy)
        v = np.clip((rr - 0.15) / (1.0 - 0.15), 0, 1) ** 1.8
        v = 1.0 - 0.55 * v  # strength

        vimg = ax.imshow(
            v,
            extent=[*ax.get_xlim(), *ax.get_ylim()],
            origin="lower",
            cmap="gray",
            alpha=0.18,
            zorder=0,
            interpolation="bilinear",
        )

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(True, which="major", linewidth=0.6, alpha=0.10)
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.4, alpha=0.04)

        edge_color = (1, 1, 1, 0.35)
        trail_color = "white"
        title_color = (1, 1, 1, title_alpha)
    else:
        vimg = None
        ax.grid(True)
        edge_color = (0, 0, 0, 0.35)
        trail_color = "black"
        title_color = (0, 0, 0, title_alpha)

    # 初期表示範囲（全体）
    xmin = float(pos[:, 0, :].min())
    xmax = float(pos[:, 0, :].max())
    ymin = float(pos[:, 1, :].min())
    ymax = float(pos[:, 1, :].max())
    pad = 0.05 * max(xmax - xmin, ymax - ymin, 1e-9)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # ========= speed scale =========
    dx = np.diff(pos[:, 0, :], axis=0)
    dy = np.diff(pos[:, 1, :], axis=0)
    spd_all = np.hypot(dx, dy).ravel()
    spd_max = np.percentile(spd_all, speed_percentile) if spd_all.size else 1.0
    spd_max = max(spd_max, 1e-9)

    # ========= (D) glow scatter =========
    glow_scat = None
    if glow:
        glow_scat = ax.scatter(
            pos[0, 0, :], pos[0, 1, :],
            s=point_size * glow_mult,
            c="white",  # ★固定
            edgecolors="none",
            linewidths=0.0,
            alpha=glow_alpha,
            zorder=1,
        )

    # ========= main scatter =========
    scat = ax.scatter(
        pos[0, 0, :], pos[0, 1, :],
        s=point_size,
        c=np.zeros(N),
        cmap=cmap,
        vmin=0.0, vmax=spd_max,
        edgecolors=edge_color,
        linewidths=edge_lw,
        zorder=3,
    )

    # ========= trail (fade) =========
    trail_lc_layers = []
    if trail and trail > 1:
        fade_gamma = 1.7
        for lag in range(trail - 1):
            a = trail_alpha * (1.0 - lag / max(1, trail - 2)) ** fade_gamma
            lc = LineCollection([], linewidths=trail_lw, alpha=a, colors=trail_color, zorder=2)
            ax.add_collection(lc)
            trail_lc_layers.append(lc)

    def _set_trail(t):
        if not trail_lc_layers:
            return
        t0 = max(0, t - trail + 1)
        seg = pos[t0:t + 1].transpose(2, 0, 1)  # (N, m, 2)
        m = seg.shape[1]
        if m < 2:
            for lc in trail_lc_layers:
                lc.set_segments([])
            return
        for layer_i, lc in enumerate(trail_lc_layers):
            idx = (m - 2) - layer_i
            if idx < 0:
                lc.set_segments([])
            else:
                lc.set_segments(seg[:, idx:idx + 2])

    # ========= camera state =========
    cam_center = np.array([pos[0, 0, :].mean(), pos[0, 1, :].mean()], dtype=float)
    cam_size = float(view_size)
    x0_s, x1_s = ax.get_xlim()
    y0_s, y1_s = ax.get_ylim()
    zoom_mult = {"v": 1.0}

    def apply_camera(x, y):
        nonlocal cam_center, cam_size, x0_s, x1_s, y0_s, y1_s
        if camera == "fixed":
            return

        target_center = np.array([x.mean(), y.mean()], dtype=float)
        cam_center = (1 - camera_smooth) * cam_center + camera_smooth * target_center

        if camera == "follow":
            target_size = float(view_size) * zoom_mult["v"]
            cam_size = target_size
        elif camera == "follow_auto":
            w = (x.max() - x.min()) + 2 * auto_pad
            h = (y.max() - y.min()) + 2 * auto_pad
            target_size = max(float(w), float(h), float(auto_min))
            if auto_max is not None:
                target_size = min(target_size, float(auto_max))
            target_size *= zoom_mult["v"]
            cam_size = (1 - zoom_smooth) * cam_size + zoom_smooth * target_size
        else:
            raise ValueError("camera must be 'fixed' / 'follow' / 'follow_auto'")

        half = 0.5 * cam_size
        tx0, tx1 = cam_center[0] - half, cam_center[0] + half
        ty0, ty1 = cam_center[1] - half, cam_center[1] + half

        a = float(window_smooth)
        x0_s = (1 - a) * x0_s + a * tx0
        x1_s = (1 - a) * x1_s + a * tx1
        y0_s = (1 - a) * y0_s + a * ty0
        y1_s = (1 - a) * y1_s + a * ty1

        ax.set_xlim(x0_s, x1_s)
        ax.set_ylim(y0_s, y1_s)


    # ========= controls =========
    paused = {"v": False}
    step_once = {"v": False}
    stop = {"v": False}
    base_interval = float(interval)
    speed = {"v": 1.0}

    # ========= manual recorder state =========
    recording = {"v": False}
    writer_box = {"writer": None}

    # ========= title =========
    title_text = (
        "Space pause | Enter step | Up/Down zoom | ←/→ speed | "
        f"{record_key.upper()} REC toggle | R zoom reset | Q quit"
    )
    title_obj = ax.set_title(title_text, color=title_color)

    def _title_update():
        if hide_title_while_recording and recording["v"]:
            title_obj.set_visible(False)
        else:
            title_obj.set_visible(True)

    def _record_start():
        if recording["v"]:
            return
        if record_fps is None:
            fps = max(1, int(round(1000 / interval)))
        else:
            fps = int(record_fps)

        w = animation.FFMpegWriter(fps=fps)
        w.setup(fig, record_path, dpi=record_dpi)
        writer_box["writer"] = w
        recording["v"] = True
        _title_update()
        print(f"[REC] start -> {record_path}")

    def _record_stop():
        if not recording["v"]:
            return
        try:
            writer_box["writer"].finish()
        finally:
            writer_box["writer"] = None
            recording["v"] = False
            _title_update()
        print(f"[REC] stop  -> {record_path}")

    def _record_grab():
        if recording["v"] and writer_box["writer"] is not None:
            writer_box["writer"].grab_frame()

    if enable_keys:
        def on_key(event):
            k = event.key
            if k == " ":
                paused["v"] = not paused["v"]
            elif k == "enter":
                if paused["v"]:
                    step_once["v"] = True
            elif k == "up":
                zoom_mult["v"] = max(zoom_mult_min, zoom_mult["v"] / zoom_step)
            elif k == "down":
                zoom_mult["v"] = min(zoom_mult_max, zoom_mult["v"] * zoom_step)
            elif k in ("r", "R"):
                zoom_mult["v"] = 1.0
            elif k in ("q", "Q", "escape"):
                stop["v"] = True
            elif k == "right":
                speed["v"] = min(8.0, speed["v"] * 1.25)
                ani.event_source.interval = base_interval / speed["v"]
            elif k == "left":
                speed["v"] = max(0.125, speed["v"] / 1.25)
                ani.event_source.interval = base_interval / speed["v"]
            elif k in ("s", "S"):
                speed["v"] = 1.0
                ani.event_source.interval = base_interval
            elif k.lower() == record_key.lower():
                if recording["v"]:
                    _record_stop()
                else:
                    _record_start()

        fig.canvas.mpl_connect("key_press_event", on_key)

    # ========= animation =========
    def init():
        x = pos[0, 0, :]
        y = pos[0, 1, :]
        apply_camera(x, y)
        _set_trail(0)
        _title_update()
        _record_grab()
        return ()

    def update(t):
        if stop["v"]:
            if recording["v"]:
                _record_stop()
            plt.close(fig)
            return ()

        if paused["v"] and not step_once["v"]:
            return ()

        step_once["v"] = False

        x = pos[t, 0, :]
        y = pos[t, 1, :]

        if t == 0:
            spd = np.zeros_like(x)
        else:
            dx = x - pos[t - 1, 0, :]
            dy = y - pos[t - 1, 1, :]
            spd = np.hypot(dx, dy)

        scat.set_offsets(np.c_[x, y])
        scat.set_array(np.clip(spd, 0.0, spd_max))

        if glow_scat is not None:
            glow_scat.set_offsets(np.c_[x, y])
            # glow_scat.set_array(np.clip(spd, 0.0, spd_max))
        _set_trail(t)

        if camera != "fixed" and (camera_every <= 1 or (t % camera_every == 0)):
            apply_camera(x, y)

        fig.canvas.draw_idle()
        _record_grab()
        return ()

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=interval, blit=False, cache_frame_data=False
    )

    plt.show()
    if recording["v"]:
        _record_stop()
    return ani


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/rec.npz")
    parser.add_argument("--output", type=str, default="outputs/animation.mp4")
    parser.add_argument("--trail", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    data = np.load(args.input)
    pos = data["pos"]

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)

    animate_positions(
        pos,
        interval=int(1000 / args.fps),
        trail=args.trail,
        camera="follow_auto",
        point_size=10,
        cmap="viridis",
        trail_lw=1.0,
        trail_alpha=0.6,
        record_key="v",
        record_path=str(out),
        record_fps=args.fps,
        glow=True,
        zoom_smooth=0.10,
        title_alpha=0.35,
    )