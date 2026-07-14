from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PCA_CSV = Path(
    r"D:\work1\task3\04_paper\03_1_ml\out_domain_shift_pca_scores_0419\csv\pca_projection_all_samples.csv"
)
RF_OTH_CSV = Path(
    r"D:\work1\task3\04_paper\03_1_ml\out_domain_shift_pca_scores_0419\csv\RandomForest_scores_oth_test.csv"
)
RF_TNBC_CSV = Path(
    r"D:\work1\task3\04_paper\03_1_ml\out_domain_shift_pca_scores_0419\csv\RandomForest_scores_tnbc_test.csv"
)


MODELS = [
    "LightGBM",
    "XGBoost",
    "GBDT",
    "RF",
    "AdaBoost",
    "Bagging Tree",
    "MLP",
    "SVM",
]
SOURCE_AUC = np.array([0.948, 0.948, 0.943, 0.940, 0.940, 0.931, 0.795, 0.730])
TNBC_AUC = np.array([0.528, 0.512, 0.402, 0.549, 0.452, 0.442, 0.506, 0.545])
SOURCE_AUC_STD = np.array([0.016, 0.015, 0.016, 0.015, 0.013, 0.017, 0.029, 0.031])
TNBC_AUC_STD = np.array([0.052, 0.050, 0.054, 0.042, 0.061, 0.043, 0.033, 0.047])


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.2)


def plot_panel_a(ax):
    x = np.arange(len(MODELS))
    ax.errorbar(
        x,
        SOURCE_AUC,
        yerr=SOURCE_AUC_STD,
        fmt="o",
        markersize=5,
        color="#4477AA",
        markeredgecolor="white",
        markeredgewidth=0.8,
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        label="Source internal",
        zorder=3,
    )
    ax.errorbar(
        x,
        TNBC_AUC,
        yerr=TNBC_AUC_STD,
        fmt="o",
        markersize=5,
        color="#CC6677",
        markeredgecolor="white",
        markeredgewidth=0.8,
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        label="TNBC external",
        zorder=3,
    )
    ax.axhline(0.5, color="#6e6e6e", linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.30, 1.00)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_title("Source-domain internal validation versus TNBC external testing", fontsize=11, pad=8)
    leg = ax.legend(frameon=True, loc="lower left", fontsize=8.8, borderpad=0.35, handlelength=1.2)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#d8d8d8")
    leg.get_frame().set_alpha(0.85)
    style_axes(ax)


def plot_panel_b(ax):
    pca = pd.read_csv(PCA_CSV)
    domain_colors = {"OTH": "#2b6ea6", "TNBC": "#c85a3d"}
    label_markers = {"Non-responder": "o", "Responder": "^"}
    domain_names = {"OTH": "Source-domain OTH", "TNBC": "TNBC"}

    for domain, color in domain_colors.items():
        for label, marker in label_markers.items():
            sub = pca[(pca["domain"] == domain) & (pca["label_text"] == label)]
            if sub.empty:
                continue
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=18,
                c=color,
                marker=marker,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.25,
            )

    ax.set_xlabel("PC1 (16.1%)", fontsize=10)
    ax.set_ylabel("PC2 (5.9%)", fontsize=10)
    ax.set_title("PCA projection of source and TNBC target samples", fontsize=11, pad=8)
    style_axes(ax)

    domain_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=7, label=domain_names[domain])
        for domain, color in domain_colors.items()
    ]
    label_handles = [
        Line2D([0], [0], marker=marker, color="#4d4d4d", markerfacecolor="#4d4d4d", linestyle="none", markersize=7, label=label)
        for label, marker in label_markers.items()
    ]
    leg1 = ax.legend(
        handles=domain_handles,
        frameon=False,
        loc="upper right",
        title="Domain",
        fontsize=8.8,
        title_fontsize=9,
        borderaxespad=0.2,
        handletextpad=0.35,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=label_handles,
        frameon=False,
        loc="lower right",
        title="Response",
        fontsize=8.8,
        title_fontsize=9,
        borderaxespad=0.2,
        handletextpad=0.35,
    )


def plot_score_distribution(ax, data, title):
    bins = np.linspace(0, 1, 18)
    label_info = {
        0: ("Non-responder", "#59636e"),
        1: ("Responder", "#c7663d"),
    }
    rng = np.random.default_rng(20260624)
    for y_value, (label, color) in label_info.items():
        scores = data.loc[data["y_true"] == y_value, "y_score"].to_numpy()
        ax.hist(
            scores,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            label=label,
        )
        ax.hist(
            scores,
            bins=bins,
            density=True,
            histtype="step",
            color=color,
            linewidth=1.0,
        )
        if len(scores) > 0:
            rug_y = rng.uniform(-0.040, -0.016, size=len(scores))
            ax.scatter(
                scores,
                rug_y,
                s=7,
                color=color,
                alpha=0.30,
                edgecolors="none",
                clip_on=False,
            )
    ax.axvline(0.5, color="#333333", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.text(
        0.515,
        0.93,
        "Threshold = 0.5",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Prediction score", fontsize=10)
    ax.set_title(title, fontsize=11, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.2)


def plot_panel_c(ax1, ax2):
    oth = pd.read_csv(RF_OTH_CSV)
    tnbc = pd.read_csv(RF_TNBC_CSV)
    plot_score_distribution(ax1, oth, "RF: source-domain internal test")
    plot_score_distribution(ax2, tnbc, "RF: TNBC external test")
    ax1.set_ylabel("Density", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    for ax in (ax1, ax2):
        leg = ax.legend(frameon=True, loc="upper left", fontsize=8.8, borderpad=0.35, handlelength=1.3)
        leg.get_frame().set_linewidth(0.4)
        leg.get_frame().set_edgecolor("#d8d8d8")
        leg.get_frame().set_alpha(0.85)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 0.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(13.8, 8.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.03, 1.0], width_ratios=[1.06, 1.0], hspace=0.42, wspace=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a)
    plot_panel_b(ax_b)
    plot_panel_c(ax_c, ax_d)

    for label, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.12, 1.10, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.955, bottom=0.10)

    for suffix in ["pdf", "svg"]:
        fig.savefig(ROOT / f"Figure2_direct_transfer_failure.{suffix}", bbox_inches="tight")
    fig.savefig(ROOT / "Figure2_direct_transfer_failure.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
