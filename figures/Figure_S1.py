from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import cmocean

sns.set_theme(style="ticks", context="paper")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "figure1"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "parameter_heatmap_heatmap_polarity_mean.csv",
    "parameter_heatmap_heatmap_milling_mean.csv",
    "parameter_heatmap_heatmap_S_mean.csv",
]

titles = [
    "(A) Mean polarity",
    "(B) Mean milling",
    "(C) Mean S",
]

points = [
    {
        "x": 1.5,
        "y": 0.5,
        "label": "SMS",
        "marker": "o",
        "color": "tab:blue",
        "text_offset": (0.06, 0.06),
    },
    {
        "x": 2.0,
        "y": 1.5,
        "label": "MS",
        "marker": "s",
        "color": "tab:orange",
        "text_offset": (0.06, 0.06),
    },
    {
        "x": 3.0,
        "y": 2.5,
        "label": "Milling",
        "marker": "^",
        "color": "tab:green",
        "text_offset": (0.06, 0.06),
    },
]


def main():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.8),
        constrained_layout=True,
    )

    for ax, filename, title in zip(axes, files, titles):
        csv_path = DATA_DIR / filename

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input file: {csv_path}")

        df = pd.read_csv(csv_path, index_col=0)

        values = df.to_numpy(dtype=float)
        option_kappa = df.columns.astype(float).to_numpy()
        percept_kappa = df.index.astype(float).to_numpy()

        cf = ax.contourf(
            option_kappa,
            percept_kappa,
            values,
            levels=30,
            cmap=cmocean.cm.balance,
        )

        ax.contour(
            option_kappa,
            percept_kappa,
            values,
            levels=12,
            colors="k",
            linewidths=0.3,
            alpha=0.4,
        )

        for point in points:
            ax.scatter(
                point["x"],
                point["y"],
                s=110,
                marker=point["marker"],
                facecolor=point["color"],
                edgecolor="white",
                linewidth=1.8,
                zorder=20,
                clip_on=False,
            )

            dx, dy = point["text_offset"]

            ax.text(
                point["x"] + dx,
                point["y"] + dy,
                point["label"],
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=21,
                bbox={
                    "facecolor": "black",
                    "edgecolor": "white",
                    "linewidth": 0.5,
                    "alpha": 0.75,
                    "pad": 1.5,
                },
            )

        ax.set_xlim(option_kappa.min(), option_kappa.max())
        ax.set_ylim(percept_kappa.min(), percept_kappa.max())

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r"$\kappa_{op}$", fontsize=13)
        ax.set_ylabel(r"$\kappa_{per}$", fontsize=13)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
        )

        sns.despine(ax=ax)

        cbar = fig.colorbar(
            cf,
            ax=ax,
            fraction=0.05,
            pad=0.03,
        )
        cbar.ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.2f")
        )

    out_path = OUT_DIR / "Figure_S1.pdf"

    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()