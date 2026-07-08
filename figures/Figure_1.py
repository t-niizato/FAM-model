from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import seaborn as sns
import cmocean


sns.set_theme(style="ticks", context="paper")


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "figure1"
OUT_DIR = ROOT / "figures" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


files = [
    "parameter_heatmap_heatmap_D_mean.csv",
    "parameter_heatmap_heatmap_D_var.csv",
    "parameter_heatmap_heatmap_Hsel_mean_i_time_mean.csv",
    "parameter_heatmap_heatmap_Hsel_mean_i_time_var.csv",
]

titles = [
    "(A) Mean D",
    "(B) Var D",
    "(C) Mean Hsel",
    "(D) Var Hsel",
]


def main():
    fig, axes = plt.subplots(
        2, 2,
        figsize=(9, 8),
        constrained_layout=True,
    )

    axes = axes.flatten()

    for ax, filename, title in zip(axes, files, titles):
        csv_path = DATA_DIR / filename

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input file: {csv_path}")

        D = pd.read_csv(csv_path, index_col=0).values

        option_kappa = np.linspace(1.0, 4.0, D.shape[1])
        percept_kappa = np.linspace(0.0, 3.0, D.shape[0])

        X, Y = np.meshgrid(option_kappa, percept_kappa)

        if np.min(D) < 0:
            vmax = np.max(np.abs(D))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            cf = ax.contourf(
                X,
                Y,
                D,
                levels=30,
                cmap=cmocean.cm.balance,
                norm=norm,
            )
        else:
            cf = ax.contourf(
                X,
                Y,
                D,
                levels=30,
                cmap=cmocean.cm.balance,
            )

        ax.contour(
            X,
            Y,
            D,
            levels=12,
            colors="k",
            linewidths=0.3,
            alpha=0.4,
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$\kappa_{op}$", fontsize=14)
        ax.set_ylabel(r"$\kappa_{per}$", fontsize=14)

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

        if title == "(D) Var Hsel":
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{x * 1e3:.2f}")
            )
            cbar.set_label(r"$\times 10^{-3}$")
        else:
            cbar.ax.yaxis.set_major_formatter(
                FormatStrFormatter("%.2f")
            )

    out_path = OUT_DIR / "Figure_1.pdf"

    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()