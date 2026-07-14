#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ssGSEA for custom modules + group comparison plots

Updates relative to the original script:
1. x-axis labels use "Non-responder (n=...)" and "Responder (n=...)".
2. outputs a 4-panel boxplot figure for modules.
3. still outputs individual boxplots for each module.
4. adds Benjamini-Hochberg FDR to group comparison table.
5. uses cleaner module titles for plotting.

Requirements:
    pip install pandas matplotlib scipy gseapy numpy
"""

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, spearmanr
except Exception:
    raise SystemExit("Please install scipy first: pip install scipy")

try:
    import gseapy as gp
except Exception:
    raise SystemExit("Please install gseapy first: pip install gseapy")


def read_table(path: str):
    p = Path(path)
    sep = "\t" if p.suffix.lower() in [".tsv", ".txt"] else ","
    return pd.read_csv(p, sep=sep)


def load_expression(expr_path, genes_axis="rows", gene_col=None):
    df = read_table(expr_path)

    if genes_axis == "rows":
        if gene_col is not None:
            if gene_col not in df.columns:
                raise SystemExit(f"gene_col '{gene_col}' not found in expression columns.")
            df = df.set_index(gene_col)
        else:
            df = df.set_index(df.columns[0])
        expr = df
    else:
        if gene_col is not None:
            raise SystemExit("gene_col is only valid when genes_axis=rows.")
        df = df.set_index(df.columns[0])
        expr = df.T

    expr.index = expr.index.astype(str).str.strip()
    expr.columns = expr.columns.astype(str).str.strip()

    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all").fillna(0.0)
    return expr


def load_metadata(meta_path, sample_col, label_col, pred_col=None):
    meta = read_table(meta_path)

    if sample_col not in meta.columns:
        raise SystemExit(f"sample_col '{sample_col}' not found in metadata.")
    if label_col not in meta.columns:
        raise SystemExit(f"label_col '{label_col}' not found in metadata.")
    if pred_col is not None and pred_col not in meta.columns:
        raise SystemExit(f"pred_col '{pred_col}' not found in metadata.")

    meta[sample_col] = meta[sample_col].astype(str).str.strip()
    return meta


def run_ssgsea(expr, modules_gmt, min_size=2, max_size=500):
    """
    expr: DataFrame, rows=genes, cols=samples
    modules_gmt: GMT file path
    """
    res = gp.ssgsea(
        data=expr,
        gene_sets=modules_gmt,
        min_size=min_size,
        max_size=max_size,
        sample_norm_method="rank",
        outdir=None,
        no_plot=True,
        processes=1,
        permutation_num=0,
        scale=False,
        format="pdf",
        verbose=True,
    )
    return res.res2d.copy()


def pivot_scores(scores_long):
    """
    Convert gseapy long-format ssGSEA output to matrix:
    index = samples, columns = modules, values = ES / NES
    """
    cols = {c.lower(): c for c in scores_long.columns}
    sample_col = cols.get("name", None)
    term_col = cols.get("term", None)
    score_col = cols.get("es", cols.get("nes", None))

    if sample_col is None or term_col is None or score_col is None:
        raise SystemExit(f"Unexpected ssGSEA output columns: {list(scores_long.columns)}")

    score_mat = scores_long.pivot(index=sample_col, columns=term_col, values=score_col)
    score_mat.index.name = "SampleID"
    score_mat = score_mat.apply(pd.to_numeric, errors="coerce")
    return score_mat


def _bh_fdr(pvals):
    """
    Benjamini-Hochberg FDR correction
    """
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out

    p = pvals[ok]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    tmp = np.empty_like(q)
    tmp[order] = q
    out[ok] = tmp
    return out


def _module_title(mod):
    title_map = {
        "MODULE_MEMBRANE_VESICLE_TRAFFICKING": "Membrane trafficking / vesicle transport",
        "MODULE_PROTEIN_PROCESSING_ER_GOLGI_GLYCOSYLATION": "ER-Golgi processing / glycosylation",
        "MODULE_AUTOPHAGY_STRESS_METABOLIC_ADAPTATION": "Autophagy / stress / metabolic adaptation",
        "MODULE_MICROENVIRONMENT_ECM_ENDOTHELIAL_INTERACTION": "Microenvironment / ECM / endothelial interaction",
    }
    return title_map.get(mod, mod.replace("_", " "))


def _module_order(modules):
    preferred = [
        "MODULE_MEMBRANE_VESICLE_TRAFFICKING",
        "MODULE_PROTEIN_PROCESSING_ER_GOLGI_GLYCOSYLATION",
        "MODULE_AUTOPHAGY_STRESS_METABOLIC_ADAPTATION",
        "MODULE_MICROENVIRONMENT_ECM_ENDOTHELIAL_INTERACTION",
    ]
    ordered = [m for m in preferred if m in modules]
    remaining = [m for m in modules if m not in ordered]
    return ordered + remaining


def save_boxplots(
    score_df,
    meta,
    sample_col,
    label_col,
    positive_label,
    outdir,
    positive_name="Responder",
    negative_name="Non-responder",
):
    merged = meta[[sample_col, label_col]].merge(
        score_df.reset_index(),
        left_on=sample_col,
        right_on="SampleID",
        how="inner"
    )

    modules = [c for c in merged.columns if c not in [sample_col, label_col, "SampleID"]]
    modules = _module_order(modules)

    stats_rows = []
    raw_pvals = []
    cache = {}

    # Compute statistics first
    for mod in modules:
        grp_pos = pd.to_numeric(
            merged.loc[merged[label_col].astype(str) == str(positive_label), mod],
            errors="coerce"
        ).dropna()

        grp_neg = pd.to_numeric(
            merged.loc[merged[label_col].astype(str) != str(positive_label), mod],
            errors="coerce"
        ).dropna()

        pval = float("nan")
        if len(grp_pos) > 0 and len(grp_neg) > 0:
            try:
                pval = mannwhitneyu(grp_pos, grp_neg, alternative="two-sided").pvalue
            except Exception:
                pass

        cache[mod] = (grp_neg, grp_pos, pval)
        raw_pvals.append(pval)

    fdrs = _bh_fdr(raw_pvals)

    for i, mod in enumerate(modules):
        grp_neg, grp_pos, pval = cache[mod]
        stats_rows.append({
            "Module": mod,
            "PrettyName": _module_title(mod),
            "Median_responder": grp_pos.median() if len(grp_pos) else float("nan"),
            "Median_nonresponder": grp_neg.median() if len(grp_neg) else float("nan"),
            "MannWhitneyU_P": pval,
            "BH_FDR": fdrs[i],
            "N_responder": len(grp_pos),
            "N_nonresponder": len(grp_neg),
        })

    pd.DataFrame(stats_rows).sort_values("MannWhitneyU_P", ascending=True).to_csv(
        outdir / "ssgsea_group_comparison.csv", index=False
    )

    # Plot settings
    plt.rcParams["font.family"] = "Times New Roman"
    point_colors = ["#1f77b4", "#ff7f0e"]  # NR / R

    # ---- 4-panel figure ----
    n_mod = len(modules)
    ncols = 2
    nrows = math.ceil(n_mod / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 9.0))
    axes = np.array(axes).reshape(-1)

    for i, mod in enumerate(modules):
        ax = axes[i]
        grp_neg, grp_pos, pval = cache[mod]

        # reproducible jitter
        rng = np.random.default_rng(2025 + i)
        x_neg = rng.normal(0, 0.025, len(grp_neg))
        x_pos = rng.normal(1, 0.025, len(grp_pos))

        ax.boxplot(
            [grp_neg, grp_pos],
            positions=[0, 1],
            widths=0.42,
            patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
            medianprops=dict(color="#d95f02", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

        ax.scatter(x_neg, grp_neg, alpha=0.75, s=24, color=point_colors[0])
        ax.scatter(x_pos, grp_pos, alpha=0.75, s=24, color=point_colors[1])

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"{negative_name} (n={len(grp_neg)})",
                f"{positive_name} (n={len(grp_pos)})",
            ],
            fontsize=10,
        )
        ax.set_ylabel("ssGSEA score", fontsize=11)

        panel = f"({chr(ord('a') + i)})"
        ttl = f"{panel}  {_module_title(mod)}"
        if not math.isnan(pval):
            ttl += f"\nMann-Whitney p={pval:.3g}"
        ax.set_title(ttl, fontsize=12)

        ax.set_facecolor("white")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        ax.set_axisbelow(True)

    for j in range(n_mod, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.savefig(outdir / "ssgsea_modules_4panel.pdf", bbox_inches="tight")
    fig.savefig(outdir / "ssgsea_modules_4panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- individual figures ----
    for i, mod in enumerate(modules):
        grp_neg, grp_pos, pval = cache[mod]

        plt.figure(figsize=(5.2, 4.6))
        rng = np.random.default_rng(3025 + i)
        x_neg = rng.normal(0, 0.025, len(grp_neg))
        x_pos = rng.normal(1, 0.025, len(grp_pos))

        plt.boxplot(
            [grp_neg, grp_pos],
            positions=[0, 1],
            widths=0.42,
            patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
            medianprops=dict(color="#d95f02", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )
        plt.scatter(x_neg, grp_neg, alpha=0.75, s=24, color=point_colors[0])
        plt.scatter(x_pos, grp_pos, alpha=0.75, s=24, color=point_colors[1])

        plt.xticks(
            [0, 1],
            [
                f"{negative_name} (n={len(grp_neg)})",
                f"{positive_name} (n={len(grp_pos)})",
            ],
            fontsize=10,
        )
        plt.ylabel("ssGSEA score", fontsize=11)

        ttl = _module_title(mod)
        if not math.isnan(pval):
            ttl += f"\nMann-Whitney p={pval:.3g}"
        plt.title(ttl, fontsize=12)

        plt.gca().set_facecolor("white")
        plt.grid(axis="y", color="#E5E5E5", linewidth=0.8)
        plt.gca().set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(outdir / f"boxplot_{mod}.pdf", bbox_inches="tight")
        plt.savefig(outdir / f"boxplot_{mod}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_heatmap(score_df, meta, sample_col, label_col, positive_label, outdir):
    m = score_df.copy()
    m = m.apply(pd.to_numeric, errors="coerce")
    m = m.fillna(m.mean(axis=0))
    m = (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-9)

    order_meta = meta[[sample_col, label_col]].copy()
    order_meta = order_meta[order_meta[sample_col].astype(str).isin(m.index)]
    order_meta["__label_bin"] = (order_meta[label_col].astype(str) == str(positive_label)).astype(int)
    order_meta = order_meta.sort_values(["__label_bin", sample_col], ascending=[False, True])

    m = m.loc[order_meta[sample_col].astype(str)]
    mat = m.T.to_numpy(dtype=float)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(max(6, 0.22 * m.shape[0] + 2), max(3.5, 0.55 * m.shape[1] + 1.5)))
    plt.imshow(mat, aspect="auto")
    plt.yticks(range(m.shape[1]), [_module_title(c) for c in m.columns])
    plt.xticks([])
    plt.xlabel("Samples")
    plt.ylabel("Modules")
    plt.title("ssGSEA module score heatmap (z-scored by module)")
    plt.colorbar(label="z-score")
    plt.tight_layout()
    plt.savefig(outdir / "ssgsea_module_heatmap.pdf", bbox_inches="tight")
    plt.savefig(outdir / "ssgsea_module_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_pred_corr(score_df, meta, sample_col, pred_col, outdir):
    merged = meta[[sample_col, pred_col]].merge(
        score_df.reset_index(),
        left_on=sample_col,
        right_on="SampleID",
        how="inner"
    )
    modules = [c for c in merged.columns if c not in [sample_col, pred_col, "SampleID"]]
    modules = _module_order(modules)

    rows = []
    plt.rcParams["font.family"] = "Times New Roman"

    for mod in modules:
        x = pd.to_numeric(merged[pred_col], errors="coerce")
        y = pd.to_numeric(merged[mod], errors="coerce")
        ok = x.notna() & y.notna()

        rho = float("nan")
        pval = float("nan")
        if ok.sum() >= 3:
            rho, pval = spearmanr(x[ok], y[ok])

        rows.append({
            "Module": mod,
            "PrettyName": _module_title(mod),
            "SpearmanR": rho,
            "PValue": pval,
            "N": int(ok.sum())
        })

        plt.figure(figsize=(4.8, 4.2))
        plt.scatter(x[ok], y[ok], alpha=0.8, s=18)
        plt.xlabel(pred_col)
        plt.ylabel("ssGSEA score")

        ttl = _module_title(mod)
        if not math.isnan(rho):
            ttl += f"\nSpearman rho={rho:.3f}, p={pval:.3g}"
        plt.title(ttl)

        plt.tight_layout()
        plt.savefig(outdir / f"predcorr_{mod}.pdf", bbox_inches="tight")
        plt.savefig(outdir / f"predcorr_{mod}.png", dpi=300, bbox_inches="tight")
        plt.close()

    pd.DataFrame(rows).sort_values("PValue", ascending=True).to_csv(
        outdir / "ssgsea_pred_correlation.csv", index=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", required=True, help="Expression matrix csv/tsv")
    ap.add_argument("--meta", required=True, help="Metadata csv/tsv with sample IDs and labels")
    ap.add_argument("--modules", required=True, help="GMT of custom modules")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--sample-col", required=True, help="Sample ID column in metadata")
    ap.add_argument("--label-col", required=True, help="Label column in metadata")
    ap.add_argument("--positive-label", default="1", help="Which label is responder/positive class")

    ap.add_argument("--positive-name", default="Responder", help="Display name for positive/responder group")
    ap.add_argument("--negative-name", default="Non-responder", help="Display name for negative/non-responder group")

    ap.add_argument("--pred-col", default=None, help="Optional prediction/probability column in metadata")
    ap.add_argument("--genes-axis", choices=["rows", "cols"], default="rows",
                    help="Whether genes are rows or columns in expression file")
    ap.add_argument("--gene-col", default=None,
                    help="Gene symbol column when genes_axis=rows; default uses first column")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    expr = load_expression(args.expr, args.genes_axis, args.gene_col)
    meta = load_metadata(args.meta, args.sample_col, args.label_col, args.pred_col)

    common = [s for s in expr.columns if s in set(meta[args.sample_col].astype(str))]
    if len(common) == 0:
        raise SystemExit("No overlapping sample IDs between expression columns and metadata sample column.")

    expr = expr[common]

    scores_long = run_ssgsea(expr, args.modules)
    scores_long.to_csv(outdir / "ssgsea_scores_long.csv", index=False)

    score_df = pivot_scores(scores_long)
    score_df = score_df.loc[[s for s in common if s in score_df.index]]
    score_df.to_csv(outdir / "ssgsea_scores_matrix.csv")

    save_boxplots(
        score_df,
        meta,
        args.sample_col,
        args.label_col,
        args.positive_label,
        outdir,
        positive_name=args.positive_name,
        negative_name=args.negative_name,
    )

    save_heatmap(score_df, meta, args.sample_col, args.label_col, args.positive_label, outdir)

    if args.pred_col is not None:
        save_pred_corr(score_df, meta, args.sample_col, args.pred_col, outdir)

    info = [
        f"Expression file: {args.expr}",
        f"Metadata file: {args.meta}",
        f"Modules GMT: {args.modules}",
        f"Samples used: {len(common)}",
        f"Genes in expression matrix: {expr.shape[0]}",
        f"Modules scored: {score_df.shape[1]}",
        f"Positive label: {args.positive_label}",
        f"Positive group display name: {args.positive_name}",
        f"Negative group display name: {args.negative_name}",
    ]
    (outdir / "run_info.txt").write_text("\n".join(info), encoding="utf-8")
    print("[OK] saved:", outdir)


if __name__ == "__main__":
    main()