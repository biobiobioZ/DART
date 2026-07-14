from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SOURCE_NPZ = Path(
    r"D:\work1\task3\04_paper\shap\shap_stage1_sourceonly\stage1_sourceonly_OTHval_shap_values.npz"
)
TARGET_NPZ = Path(
    r"D:\work1\task3\04_paper\shap\shap_stage2_agg\single\stage2_final_rep03_outertest_shap_values.npz"
)
STAGE1_CSV = Path(
    r"D:\work1\task3\04_paper\shap\shap_stage1_sourceonly\stage1_sourceonly_OTHval_shap_meanabs.csv"
)
STAGE2_CSV = Path(
    r"D:\work1\task3\04_paper\shap\shap_stage2_agg\stage2_final_agg_shap_meanabs.csv"
)
LOG_PATH = ROOT / "Figure5_model_interpretation_label_correction_log.txt"

def display_gene(gene):
    return str(gene)


def load_shap(path):
    data = np.load(path, allow_pickle=True)
    genes = np.array([str(x) for x in data["genes"]])
    shap = np.asarray(data["shap"], dtype=float)
    x = np.asarray(data["X"], dtype=float)
    return genes, shap, x, data.files


def check_gene_conflicts(source_genes, target_genes):
    tracked = ["CLTA"]
    source_hits = [g for g in tracked if g in set(source_genes)]
    target_hits = [g for g in tracked if g in set(target_genes)]
    conflicts = len(source_hits) > 1 or len(target_hits) > 1
    if conflicts:
        raise RuntimeError(
            "Conflicting CLTA feature names found. "
            f"source={source_hits}; target={target_hits}"
        )
    return source_hits, target_hits


def normalise_feature_values(values):
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def beeswarm_panel(ax, shap, x, genes, title, top_n=20):
    mean_abs = np.abs(shap).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    rng = np.random.default_rng(20260610)

    y_positions = np.arange(top_n)
    for row, idx in enumerate(top_idx):
        vals = shap[:, idx]
        feat = normalise_feature_values(x[:, idx])
        jitter = rng.normal(0, 0.085, size=vals.shape[0])
        ax.scatter(
            vals,
            np.full(vals.shape[0], row) + jitter,
            c=feat,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=12,
            alpha=0.78,
            linewidths=0,
        )

    ax.axvline(0, color="#555555", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([display_gene(genes[i]) for i in top_idx])
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(title, fontsize=12.5, pad=9)
    ax.grid(axis="x", color="#e2e2e2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return top_idx


def load_aggregated_importance_frame():
    stage1 = pd.read_csv(STAGE1_CSV)[["gene", "mean_abs_shap"]].rename(
        columns={"mean_abs_shap": "stage1_mean_abs"}
    )
    stage2 = pd.read_csv(STAGE2_CSV)[["gene", "mean_abs_shap", "topk_freq"]].rename(
        columns={"mean_abs_shap": "stage2_mean_abs"}
    )
    df = stage1.merge(stage2, on="gene", how="inner", validate="one_to_one")
    if len(df) != 800:
        raise RuntimeError(f"Expected 800 shared input genes, found {len(df)}.")
    df["display_gene"] = [display_gene(g) for g in df["gene"]]
    return df


def stratify(df):
    x_thr = df["stage1_mean_abs"].quantile(0.75)
    y_thr = df["stage2_mean_abs"].quantile(0.75)
    conditions = [
        (df["stage1_mean_abs"] >= x_thr) & (df["stage2_mean_abs"] >= y_thr),
        (df["stage1_mean_abs"] >= x_thr) & (df["stage2_mean_abs"] < y_thr),
        (df["stage1_mean_abs"] < x_thr) & (df["stage2_mean_abs"] >= y_thr),
    ]
    labels = ["Q1 stable core", "Q2 source-specific", "Q3 target-adapted"]
    df = df.copy()
    df["stratum"] = np.select(conditions, labels, default="Q4 background")
    return df, x_thr, y_thr


def representative_labels(df):
    manual = [
        "CLTA",
        "AIMP2",
        "CHMP5",
        "CX3CL1",
        "COL4A1",
        "ALG6",
        "COPB2",
        "DLG5",
        "IGF2R",
        "DERL1",
    ]
    present = [g for g in manual if g in set(df["gene"])]
    labels = df[df["gene"].isin(present)].copy()
    return labels


def scatter_panel(ax, df, x_thr, y_thr):
    colors = {
        "Q1 stable core": "#2b6ea6",
        "Q2 source-specific": "#d98c32",
        "Q3 target-adapted": "#3a9850",
        "Q4 background": "#cfcfcf",
    }
    order = ["Q4 background", "Q2 source-specific", "Q3 target-adapted", "Q1 stable core"]
    for group in order:
        sub = df[df["stratum"] == group]
        ax.scatter(
            sub["stage1_mean_abs"],
            sub["stage2_mean_abs"],
            s=17 if group != "Q4 background" else 10,
            color=colors[group],
            alpha=0.78 if group != "Q4 background" else 0.45,
            edgecolor="white" if group != "Q4 background" else "none",
            linewidth=0.25,
            label=group,
        )

    ax.axvline(x_thr, color="#666666", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.axhline(y_thr, color="#666666", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_xlabel(r"Stage 1 $\mathrm{mean}(|\mathrm{SHAP}|)$", fontsize=11)
    ax.set_ylabel(r"Stage 2 $\mathrm{mean}(|\mathrm{SHAP}|)$", fontsize=11)
    ax.set_title("Cross-domain stratification of SHAP importance", fontsize=12.5, pad=9)
    ax.grid(color="#e2e2e2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    label_offsets = {
        "CLTA": (7, 7),
        "AIMP2": (-32, 8),
        "CHMP5": (-45, -18),
        "CX3CL1": (18, 12),
        "COL4A1": (12, -18),
        "COPB2": (8, 8),
        "ALG6": (10, -14),
        "DLG5": (12, 14),
        "IGF2R": (-38, 6),
        "DERL1": (-34, -8),
    }
    for _, row in representative_labels(df).iterrows():
        label = row["display_gene"]
        ax.annotate(
            label,
            (row["stage1_mean_abs"], row["stage2_mean_abs"]),
            xytext=label_offsets.get(label, (5, 4)),
            textcoords="offset points",
            fontsize=10,
            fontstyle="italic",
            arrowprops={
                "arrowstyle": "-",
                "color": "#888888",
                "linewidth": 0.45,
                "shrinkA": 0,
                "shrinkB": 2,
            },
        )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[k], markeredgecolor="none", markersize=6, label=k)
        for k in ["Q1 stable core", "Q2 source-specific", "Q3 target-adapted", "Q4 background"]
    ]
    handles.append(
        Line2D([0], [0], color="#666666", linestyle=(0, (4, 3)), linewidth=0.9, label="P75 cutoffs")
    )
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=10.7)


def write_log(source_hits, target_hits, keys_source, keys_target, df, x_thr, y_thr):
    tracked = ["CLTA"]
    lines = [
        f"source npz: {SOURCE_NPZ}",
        f"target npz: {TARGET_NPZ}",
        f"source keys: {list(keys_source)}",
        f"target keys: {list(keys_target)}",
        f"source contains tracked names: {source_hits}",
        f"target contains tracked names: {target_hits}",
        f"Stage 1 P75 mean(|SHAP|) cutoff: {x_thr:.8f}",
        f"Stage 2 P75 mean(|SHAP|) cutoff: {y_thr:.8f}",
        "display label correction:",
    ]
    for original in tracked:
        if original in set(df["gene"]):
            lines.append(f"  {original} -> {display_gene(original)}")
    lines.append("conflict detected: no")
    lines.append("final displayed gene name for tracked CLTA features: CLTA")
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    source_genes, source_shap, source_x, source_keys = load_shap(SOURCE_NPZ)
    target_genes, target_shap, target_x, target_keys = load_shap(TARGET_NPZ)
    if not np.array_equal(source_genes, target_genes):
        raise RuntimeError("Source and target SHAP files use different gene orders; align by gene symbol before plotting.")

    source_hits, target_hits = check_gene_conflicts(source_genes, target_genes)
    df = load_aggregated_importance_frame()
    df, x_thr, y_thr = stratify(df)
    write_log(source_hits, target_hits, source_keys, target_keys, df, x_thr, y_thr)

    fig = plt.figure(figsize=(14.5, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], hspace=0.34, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    beeswarm_panel(ax_a, source_shap, source_x, source_genes, "Source-trained model SHAP summary", top_n=20)
    beeswarm_panel(ax_b, target_shap, target_x, target_genes, "Target-adapted model SHAP summary", top_n=20)
    scatter_panel(ax_c, df, x_thr, y_thr)

    for label, ax in zip(["A", "B", "C"], [ax_a, ax_b, ax_c]):
        ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

    fig.subplots_adjust(left=0.08, right=0.93, top=0.94, bottom=0.08)

    # One shared colour bar for feature values in panels A and B.
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.945, 0.585, 0.012, 0.22])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Feature value", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    for suffix in ["pdf", "svg"]:
        fig.savefig(ROOT / f"Figure5_model_interpretation.{suffix}", bbox_inches="tight")
    fig.savefig(ROOT / "Figure5_model_interpretation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
