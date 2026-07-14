from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = Path(r"D:\work1\task3\04_paper\04_tf\summary.csv")
OOS_CSV = Path(
    r"D:\work1\task3\04_paper\04_tf\picture\p3\oos_topfigs_double\oos_aggregated_pred_mean.csv"
)


BASELINE_METHODS = ["TIDE", "IC2Bert", "NetBio", "IRnet", "SVM", "DART"]
BASELINE_MEAN = np.array([0.501, 0.602, 0.536, 0.666, 0.545, 0.777])
BASELINE_STD = np.array([0.150, 0.153, 0.124, 0.118, 0.047, 0.119])


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e1e1", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.8)


def roc_curve_and_auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    distinct = np.where(np.diff(score[order]))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tpr = np.r_[0, tps[threshold_idxs] / positives, 1]
    fpr = np.r_[0, fps[threshold_idxs] / negatives, 1]
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc


def plot_panel_a(ax, auc_values):
    rng = np.random.default_rng(20260608)
    x = np.ones_like(auc_values)
    jitter = rng.normal(0, 0.035, size=len(auc_values))

    parts = ax.violinplot([auc_values], positions=[1], widths=0.45, showmeans=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#6a91b8")
        body.set_edgecolor("#385c7d")
        body.set_alpha(0.25)

    ax.boxplot(
        [auc_values],
        positions=[1],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#385c7d", "linewidth": 1.1},
        medianprops={"color": "#385c7d", "linewidth": 1.2},
        whiskerprops={"color": "#385c7d", "linewidth": 1.0},
        capprops={"color": "#385c7d", "linewidth": 1.0},
    )
    ax.scatter(x + jitter, auc_values, s=22, color="#2b6ea6", alpha=0.75, edgecolor="white", linewidth=0.35)
    ax.axhline(0.5, color="#444444", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(1, 0.98, r"Mean $\pm$ SD = 0.777 $\pm$ 0.119", ha="center", va="top", fontsize=10)
    ax.set_xlim(0.58, 1.42)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks([1])
    ax.set_xticklabels(["DART"])
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Repeated target-domain validations", fontsize=12, pad=9)
    style_axes(ax)


def plot_panel_b(ax):
    x = np.arange(len(BASELINE_METHODS))
    colors = ["#b6b6b6", "#b6b6b6", "#b6b6b6", "#b6b6b6", "#8fa6bd", "#c65f4a"]
    ax.errorbar(
        x,
        BASELINE_MEAN,
        yerr=BASELINE_STD,
        fmt="none",
        ecolor="#5f666d",
        elinewidth=1.1,
        capsize=3,
        zorder=1,
    )
    ax.scatter(x, BASELINE_MEAN, s=58, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
    ax.axhline(0.5, color="#444444", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(BASELINE_METHODS, rotation=30, ha="right")
    ax.set_ylim(0.30, 1.02)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Comparison with baseline and public methods", fontsize=12, pad=9)
    style_axes(ax)


def plot_panel_c(ax, oos):
    score_col = "p_oos_mean" if "p_oos_mean" in oos.columns else "p"
    fpr, tpr, auc = roc_curve_and_auc(oos["y"], oos[score_col])
    ax.plot([0, 1], [0, 1], color="#555555", linestyle=(0, (5, 3)), linewidth=1.7, label="_nolegend_")
    ax.plot(fpr, tpr, color="#2b6ea6", linewidth=2.0, label=f"OOS ROC (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, fpr, color="#2b6ea6", alpha=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("Cross-fit out-of-sample ROC", fontsize=12, pad=9)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.grid(color="#e1e1e1", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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

    summary = pd.read_csv(SUMMARY_CSV)
    oos = pd.read_csv(OOS_CSV)
    auc_values = summary["outer_auc"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    plot_panel_a(axes[0], auc_values)
    plot_panel_b(axes[1])
    plot_panel_c(axes[2], oos)

    for label, ax in zip(["A", "B", "C"], axes):
        ax.text(-0.16, 1.12, label, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.068, right=0.985, top=0.86, bottom=0.22, wspace=0.34)

    for suffix in ["pdf", "svg"]:
        fig.savefig(ROOT / f"Figure3_dart_performance.{suffix}", bbox_inches="tight")
    fig.savefig(ROOT / "Figure3_dart_performance.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
