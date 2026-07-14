from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent

PANEL_A_LABELS = [
    "Full",
    "w/o pre-training",
    "w/o adversarial\nalignment",
    "w/o last-layer\nadaptation",
    "w/o L2-SP",
]
PANEL_A_MEAN = np.array([0.777, 0.739, 0.761, 0.775, 0.768])
PANEL_A_STD = np.array([0.119, 0.105, 0.120, 0.116, 0.117])

PANEL_B_LABELS = [
    "w/o pre-training",
    "w/o adversarial\nalignment",
    "w/o last-layer\nadaptation",
    "w/o L2-SP",
]
DELTA_MEAN = np.array([0.044, 0.016, 0.002, 0.009])
CI_LOW = np.array([0.003, 0.006, -0.005, -0.002])
CI_HIGH = np.array([0.057, 0.027, 0.004, 0.018])


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e1e1", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.8)


def plot_panel_a(ax):
    x = np.arange(len(PANEL_A_LABELS))
    colors = ["#c65f4a", "#4f7fa8", "#4f7fa8", "#8d8d8d", "#8d8d8d"]
    ax.errorbar(
        x,
        PANEL_A_MEAN,
        yerr=PANEL_A_STD,
        fmt="none",
        ecolor="#5f666d",
        elinewidth=1.25,
        capsize=4,
        zorder=2,
    )
    ax.scatter(
        x,
        PANEL_A_MEAN,
        s=58,
        facecolors="white",
        edgecolors=colors,
        linewidths=1.8,
        zorder=3,
    )
    ax.axhline(0.5, color="#444444", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(PANEL_A_LABELS)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_ylim(0.49, 0.92)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_title("Outer-test AUC across ablation settings", fontsize=12, pad=9)
    style_axes(ax)


def plot_panel_b(ax):
    x = np.arange(len(PANEL_B_LABELS))
    colors = ["#2b6ea6", "#6a91b8", "#b6b6b6", "#b6b6b6"]
    yerr = np.vstack([DELTA_MEAN - CI_LOW, CI_HIGH - DELTA_MEAN])
    ax.axhline(0, color="#555555", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.bar(x, DELTA_MEAN, color=colors, edgecolor="white", linewidth=0.8, width=0.65, zorder=2)
    ax.errorbar(
        x,
        DELTA_MEAN,
        yerr=yerr,
        fmt="none",
        ecolor="#3f454a",
        elinewidth=1.1,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PANEL_B_LABELS)
    ax.set_ylabel(r"Paired $\Delta$AUC", fontsize=11)
    ax.set_ylim(-0.012, 0.064)
    ax.set_title(r"Contribution estimated by paired $\Delta$AUC", fontsize=12, pad=9)
    style_axes(ax)
    ax.text(
        0.02,
        0.96,
        r"Paired $\Delta$AUC = AUC(Full) $-$ AUC(Ablated)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.2,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.8))
    plot_panel_a(axes[0])
    plot_panel_b(axes[1])

    for label, ax in zip(["A", "B"], axes):
        ax.text(-0.12, 1.12, label, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.078, right=0.985, top=0.86, bottom=0.24, wspace=0.30)

    for suffix in ["pdf", "svg"]:
        fig.savefig(ROOT / f"Figure4_ablation_analysis.{suffix}", bbox_inches="tight")
    fig.savefig(ROOT / "Figure4_ablation_analysis.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
