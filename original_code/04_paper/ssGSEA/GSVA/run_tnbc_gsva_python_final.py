#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import textwrap
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, rankdata


def parse_gene_sets(txt_path):
    """
    Parse gene-set text blocks.
    Expected format:

    Module name line 1
    Module name line 2 (optional)
    GENE1, GENE2, GENE3

    blank line
    next module ...
    """
    with open(txt_path, encoding="utf-8") as f:
        txt = f.read().strip()

    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    gene_sets = OrderedDict()

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise ValueError(
                f"Invalid gene-set block in {txt_path!r}: each block must contain at least "
                "one name line and one gene line."
            )

        name = " ".join(lines[:-1])
        name = re.sub(r"\s*\(\d+\s+genes\)\s*", "", name)
        name = " ".join(name.split())
        genes = [g.strip() for g in lines[-1].split(",") if g.strip()]

        if not genes:
            raise ValueError(f"Gene set {name!r} has no genes.")
        if name in gene_sets:
            raise ValueError(f"Duplicated gene-set name: {name!r}")

        gene_sets[name] = genes

    return gene_sets


def gaussian_kernel_cdf_per_gene(values):
    n_genes, n_samples = values.shape
    out = np.empty_like(values, dtype=float)

    for i in range(n_genes):
        x = values[i]
        sd = np.std(x, ddof=1) if n_samples > 1 else 0.0
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sigma = min(sd, iqr / 1.349) if (sd > 0 and iqr > 0) else max(sd, iqr / 1.349)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(sd, 1.0)

        h = 0.9 * sigma * (n_samples ** (-1 / 5))
        if not np.isfinite(h) or h <= 0:
            h = 1.0

        diffs = (x[:, None] - x[None, :]) / h
        cdfs = norm.cdf(diffs).mean(axis=1)
        cdfs = np.clip(cdfs, 1e-10, 1 - 1e-10)
        out[i] = np.log(cdfs / (1 - cdfs))

    return out


def direct_ecdf_logit_per_gene(values):
    n_genes, n_samples = values.shape
    out = np.empty_like(values, dtype=float)

    for i in range(n_genes):
        x = values[i]
        ord_idx = np.argsort(x, kind="mergesort")
        ecdf_vals = np.empty(n_samples, dtype=float)
        sorted_x = x[ord_idx]

        pos = 0
        while pos < n_samples:
            end = pos
            while end + 1 < n_samples and sorted_x[end + 1] == sorted_x[pos]:
                end += 1
            ecdf = (end + 1) / n_samples
            ecdf_vals[ord_idx[pos : end + 1]] = ecdf
            pos = end + 1

        ecdf_vals = np.clip(ecdf_vals, 1e-10, 1 - 1e-10)
        out[i] = np.log(ecdf_vals / (1 - ecdf_vals))

    return out


def compute_rank_scores_from_sort(sort_idxs):
    n_genes, n_samples = sort_idxs.shape
    base = np.abs(np.arange(n_genes, 0, -1) - n_genes / 2.0)
    rank_scores = np.empty((n_genes, n_samples), dtype=float)

    for j in range(n_samples):
        tmp = np.zeros(n_genes, dtype=float)
        tmp[sort_idxs[:, j]] = base
        rank_scores[:, j] = tmp

    return rank_scores


def gsva_es_for_geneset(score_vec, sort_idx, gset_idxs, tau=1.0, mx_diff=True, abs_ranking=False):
    n_genes = len(score_vec)
    n_gset = len(gset_idxs)
    if n_gset == 0:
        return np.nan
    if n_gset >= n_genes:
        raise ValueError("Gene set size must be smaller than total number of genes.")

    gset_idxs = np.array(gset_idxs, dtype=int)
    sum_gset = np.sum(np.abs(score_vec[gset_idxs]) ** tau)
    if sum_gset <= 0:
        return 0.0

    dec = 1.0 / (n_genes - n_gset)
    pos_map = np.empty(n_genes, dtype=int)
    pos_map[sort_idx] = np.arange(1, n_genes + 1)
    offsets = np.sort(pos_map[gset_idxs])

    last_idx = 0
    current = 0.0
    vals = []
    for off in offsets:
        gene_index = sort_idx[off - 1]
        current += (abs(score_vec[gene_index]) ** tau / sum_gset) - dec * (off - last_idx - 1)
        vals.append(current)
        last_idx = off

    vals = np.array(vals)
    max_pos = np.max(vals)
    min_neg = np.min(vals)

    if not mx_diff:
        return vals[np.argmax(np.abs(vals))]
    if abs_ranking:
        return max_pos + abs(min_neg)
    return max_pos - abs(min_neg)


def gsva_from_density(expr_df, gene_sets, gene_density, tau=1.0):
    sort_idxs = np.argsort(gene_density, axis=0)[::-1, :]
    rank_scores = compute_rank_scores_from_sort(sort_idxs)
    gene_to_idx = {g: i for i, g in enumerate(expr_df.index)}

    es = np.empty((len(gene_sets), expr_df.shape[1]), dtype=float)
    for a, (_, genes) in enumerate(gene_sets.items()):
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        for j in range(expr_df.shape[1]):
            es[a, j] = gsva_es_for_geneset(rank_scores[:, j], sort_idxs[:, j], idxs, tau=tau)

    return pd.DataFrame(es, index=list(gene_sets.keys()), columns=expr_df.columns)


def ssgsea(expr_df, gene_sets, alpha=0.25, normalization=True):
    X = expr_df.values.astype(float)
    R = np.apply_along_axis(lambda x: rankdata(x, method="average"), 0, X)
    Ra = np.abs(R) ** alpha
    gene_to_idx = {g: i for i, g in enumerate(expr_df.index)}

    es = np.empty((len(gene_sets), expr_df.shape[1]), dtype=float)
    for j in range(expr_df.shape[1]):
        gene_ranking = np.argsort(R[:, j])[::-1]
        n = len(gene_ranking)

        for a, (_, genes) in enumerate(gene_sets.items()):
            gset_idx = np.array([gene_to_idx[g] for g in genes if g in gene_to_idx], dtype=int)
            k = len(gset_idx)
            if k == 0:
                es[a, j] = np.nan
                continue
            if k >= n:
                raise ValueError("Gene set size must be smaller than total number of genes.")

            idxs = np.sort(np.array([np.where(gene_ranking == g)[0][0] for g in gset_idx]))
            denom = np.sum(Ra[gene_ranking[idxs], j])
            step_in = np.sum(Ra[gene_ranking[idxs], j] * (n - idxs)) / denom if denom > 0 else 0.0
            step_out = (n * (n + 1) / 2 - np.sum(n - idxs)) / (n - k)
            es[a, j] = step_in - step_out

    if normalization:
        denom = es.max() - es.min()
        if denom > 0:
            es = es / denom

    return pd.DataFrame(es, index=list(gene_sets.keys()), columns=expr_df.columns)


def bh_adjust(pvals):
    p = np.array(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    out = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = min(prev, p[idx] * n / rank)
        out[idx] = val
        prev = val

    return out


def group_stats(score_df, meta_df, method_name):
    long_df = score_df.T.reset_index().rename(columns={"index": "SampleID"})
    long_df = long_df.merge(meta_df, on="SampleID", how="left")

    rows = []
    for module in score_df.index:
        vals_r = long_df.loc[long_df["Response"] == 1, module].dropna().values.astype(float)
        vals_n = long_df.loc[long_df["Response"] == 0, module].dropna().values.astype(float)

        if len(vals_r) == 0 or len(vals_n) == 0:
            u = np.nan
            p = np.nan
        else:
            u, p = mannwhitneyu(vals_r, vals_n, alternative="two-sided")

        rows.append({
            "method": method_name,
            "module": module,
            "n_responder": len(vals_r),
            "n_nonresponder": len(vals_n),
            "median_responder": float(np.median(vals_r)) if len(vals_r) else np.nan,
            "median_nonresponder": float(np.median(vals_n)) if len(vals_n) else np.nan,
            "delta_median": float(np.median(vals_r) - np.median(vals_n)) if len(vals_r) and len(vals_n) else np.nan,
            "mean_responder": float(np.mean(vals_r)) if len(vals_r) else np.nan,
            "mean_nonresponder": float(np.mean(vals_n)) if len(vals_n) else np.nan,
            "delta_mean": float(np.mean(vals_r) - np.mean(vals_n)) if len(vals_r) and len(vals_n) else np.nan,
            "mannwhitney_u": float(u) if np.isfinite(u) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })

    stats_df = pd.DataFrame(rows)
    valid_mask = stats_df["p_value"].notna()
    stats_df["fdr_bh"] = np.nan
    if valid_mask.any():
        stats_df.loc[valid_mask, "fdr_bh"] = bh_adjust(stats_df.loc[valid_mask, "p_value"])
    stats_df["direction"] = np.where(stats_df["delta_median"] > 0, "Responder_higher", "Nonresponder_higher")
    return stats_df, long_df


def zscore_by_module(score_df):
    arr = score_df.values.astype(float)
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    z = (arr - mean) / std
    return pd.DataFrame(z, index=score_df.index, columns=score_df.columns)


def normalize_response_value(x):
    if pd.isna(x):
        raise ValueError("Response column contains missing values.")

    s = str(x).strip().lower()
    if s in {"1", "r", "responder", "response", "yes", "true"}:
        return 1
    if s in {"0", "nr", "nonresponder", "non-responder", "no", "false"}:
        return 0

    try:
        v = int(float(s))
    except ValueError as exc:
        raise ValueError(
            "Response column must be binary and interpretable as one of: "
            "0/1, NR/R, nonresponder/responder. "
            f"Got: {x!r}"
        ) from exc

    if v not in {0, 1}:
        raise ValueError(f"Response value must be 0 or 1 after conversion. Got: {x!r}")
    return v


MODULE_TITLE_MAP = {
    "Transport & processing": "Transport &\nprocessing",
    "Cytoskeleton, adhesion & microenvironment": "Cytoskeleton,\nadhesion &\nmicroenvironment",
    "Cell cycle, stress & metabolism": "Cell cycle,\nstress &\nmetabolism",
}


MODULE_COLOR_MAP = {
    "Transport & processing": ("#4C9BD3", "#F39A3D"),
    "Cytoskeleton, adhesion & microenvironment": ("#59B35C", "#E86A6A"),
    "Cell cycle, stress & metabolism": ("#A78BDF", "#B9856F"),
}


DEFAULT_COLOR_PAIRS = [
    ("#4C9BD3", "#F39A3D"),
    ("#59B35C", "#E86A6A"),
    ("#A78BDF", "#B9856F"),
    ("#4DB6AC", "#FF8A65"),
    ("#90CAF9", "#FFCC80"),
    ("#81C784", "#EF9A9A"),
]


def wrap_module_title(name, width=18):
    if name in MODULE_TITLE_MAP:
        return MODULE_TITLE_MAP[name]
    return textwrap.fill(name, width=width)


def get_color_pair(module_name, idx):
    if module_name in MODULE_COLOR_MAP:
        return MODULE_COLOR_MAP[module_name]
    return DEFAULT_COLOR_PAIRS[idx % len(DEFAULT_COLOR_PAIRS)]


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })


def plot_module_boxplots(score_df, meta_df, stats_df, out_png, out_pdf=None, y_label="ssGSEA score (z-score)"):
    setup_plot_style()

    response_map = meta_df.set_index("SampleID")["Response"].to_dict()
    modules = list(score_df.index)
    n_panels = len(modules)
    fig_w = max(4.1 * n_panels, 9.0)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.8), constrained_layout=True)

    if n_panels == 1:
        axes = [axes]

    rng = np.random.default_rng(2026)

    for i, (ax, module) in enumerate(zip(axes, modules)):
        nr_samples = [s for s in score_df.columns if response_map.get(s) == 0]
        r_samples = [s for s in score_df.columns if response_map.get(s) == 1]

        vals_nr = score_df.loc[module, nr_samples].dropna().values.astype(float)
        vals_r = score_df.loc[module, r_samples].dropna().values.astype(float)

        ax.boxplot(
            [vals_nr, vals_r],
            positions=[1, 2],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.6),
            capprops=dict(color="black", linewidth=1.6),
            medianprops=dict(color="#D95F02", linewidth=2.0),
        )

        c_nr, c_r = get_color_pair(module, i)
        x_nr = 1 + rng.uniform(-0.11, 0.11, size=len(vals_nr)) if len(vals_nr) else np.array([])
        x_r = 2 + rng.uniform(-0.11, 0.11, size=len(vals_r)) if len(vals_r) else np.array([])

        ax.scatter(x_nr, vals_nr, s=22, color=c_nr, alpha=0.78, zorder=3)
        ax.scatter(x_r, vals_r, s=22, color=c_r, alpha=0.78, zorder=3)

        ax.set_title(wrap_module_title(module), pad=12)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["NR", "R"])
        ax.set_ylabel(y_label if i == 0 else "")

        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        p_series = stats_df.loc[stats_df["module"] == module, "p_value"]
        p = p_series.iloc[0] if len(p_series) else np.nan
        p_text = rf"$P$ = {p:.3f}" if np.isfinite(p) else r"$P$ = NA"
        ax.text(0.5, 0.93, p_text, transform=ax.transAxes, ha="center", va="top", fontsize=15)

        all_vals = np.concatenate([vals_nr, vals_r]) if (len(vals_nr) + len(vals_r)) else np.array([0.0])
        ymin = float(np.nanmin(all_vals))
        ymax = float(np.nanmax(all_vals))
        yrange = ymax - ymin
        if yrange <= 0:
            yrange = 1.0
        ax.set_ylim(ymin - 0.18 * yrange, ymax + 0.28 * yrange)

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def load_expression_matrix(expr_path):
    expr = pd.read_csv(expr_path)
    if expr.empty:
        raise ValueError(f"Expression file is empty: {expr_path}")

    gene_col = expr.columns[0]
    expr = expr.rename(columns={gene_col: "Gene"}).set_index("Gene")
    expr.index = expr.index.astype(str).str.strip()
    expr = expr.loc[expr.index.notna() & (expr.index != "")]
    expr = expr[~expr.index.duplicated(keep="first")]

    for col in expr.columns:
        expr[col] = pd.to_numeric(expr[col], errors="coerce")

    expr = expr.dropna(axis=0, how="all")
    return expr


def load_metadata(meta_path):
    meta = pd.read_csv(meta_path)
    cols_lower = {c.lower(): c for c in meta.columns}

    if "sampleid" not in cols_lower:
        raise ValueError("Metadata file must contain a SampleID column.")
    if "response" not in cols_lower:
        raise ValueError("Metadata file must contain a Response column.")

    sample_col = cols_lower["sampleid"]
    resp_col = cols_lower["response"]

    meta = meta.rename(columns={sample_col: "SampleID", resp_col: "Response"})
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    meta = meta.loc[meta["SampleID"] != ""].copy()
    meta = meta.drop_duplicates(subset=["SampleID"], keep="first")
    meta["Response"] = meta["Response"].map(normalize_response_value)
    return meta


def make_output_tables(outdir, name, score_df, meta_df):
    stats_df, long_df = group_stats(score_df, meta_df, name)
    long_path = os.path.join(outdir, f"{name}_scores_long.csv")
    wide_path = os.path.join(outdir, f"{name}_scores_wide.csv")
    stats_path = os.path.join(outdir, f"{name}_stats.csv")
    long_df.to_csv(long_path, index=False)
    score_df.to_csv(wide_path)
    stats_df.to_csv(stats_path, index=False)
    return stats_df, long_df, long_path, wide_path, stats_path


def plot_and_save(outdir, name, score_df, meta_df, do_zscore=True, y_label_base=None):
    plot_df = zscore_by_module(score_df) if do_zscore else score_df.copy()
    plot_name = f"{name}_zscore" if do_zscore else name
    stats_df, _ = group_stats(plot_df, meta_df, plot_name)

    if y_label_base is None:
        y_label_base = name
    y_label = f"{y_label_base} score (z-score)" if do_zscore else f"{y_label_base} score"

    png_path = os.path.join(outdir, f"{plot_name}_boxplot.png")
    pdf_path = os.path.join(outdir, f"{plot_name}_boxplot.pdf")
    plot_module_boxplots(plot_df, meta_df, stats_df, png_path, pdf_path, y_label=y_label)
    plot_df.to_csv(os.path.join(outdir, f"{plot_name}_scores_wide.csv"))
    stats_df.to_csv(os.path.join(outdir, f"{plot_name}_stats.csv"), index=False)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute GSVA/ssGSEA scores and generate publication-style boxplots."
    )
    parser.add_argument(
        "--expr",
        required=True,
        help="CSV with genes in rows and samples in columns; first column is treated as gene symbol.",
    )
    parser.add_argument(
        "--meta",
        required=True,
        help="CSV with SampleID and Response columns. Response accepts 0/1, NR/R, or nonresponder/responder.",
    )
    parser.add_argument(
        "--gene_sets",
        required=True,
        help="Text file with module names and comma-separated genes.",
    )
    parser.add_argument("--outdir", default="tnbc_gsva_python")
    parser.add_argument(
        "--plot_method",
        default="ssgsea",
        choices=["ssgsea", "gsva_gaussian", "gsva_ecdf", "all"],
        help="Which scoring result to plot. Default: ssgsea",
    )
    parser.add_argument(
        "--no_zscore",
        action="store_true",
        help="Plot raw scores instead of per-module z-score transformed scores.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    expr = load_expression_matrix(args.expr)
    meta = load_metadata(args.meta)
    gene_sets = parse_gene_sets(args.gene_sets)

    samples = [c for c in expr.columns if c in set(meta["SampleID"])]
    if not samples:
        raise ValueError("No overlapping samples were found between expression matrix and metadata.")

    expr = expr[samples].copy()
    meta = meta.set_index("SampleID").loc[samples].reset_index()

    if expr.shape[0] < 2:
        raise ValueError("Expression matrix must contain at least two genes after preprocessing.")

    gd_gauss = gaussian_kernel_cdf_per_gene(expr.values.astype(float))
    gd_ecdf = direct_ecdf_logit_per_gene(expr.values.astype(float))

    gsva_gaussian = gsva_from_density(expr, gene_sets, gd_gauss)
    gsva_ecdf = gsva_from_density(expr, gene_sets, gd_ecdf)
    ssgsea_scores = ssgsea(expr, gene_sets)

    outputs = {}

    outputs["gsva_gaussian"] = make_output_tables(args.outdir, "gsva_gaussian", gsva_gaussian, meta)
    outputs["gsva_ecdf"] = make_output_tables(args.outdir, "gsva_ecdf", gsva_ecdf, meta)
    outputs["ssgsea"] = make_output_tables(args.outdir, "ssgsea", ssgsea_scores, meta)

    summary_tables = []
    for method_name, (stats_df, _, _, _, _) in outputs.items():
        tmp = stats_df.copy()
        tmp.insert(0, "score_type", method_name)
        summary_tables.append(tmp)
    pd.concat(summary_tables, ignore_index=True).to_csv(
        os.path.join(args.outdir, "all_methods_summary_stats.csv"), index=False
    )

    plot_targets = [args.plot_method] if args.plot_method != "all" else ["ssgsea", "gsva_gaussian", "gsva_ecdf"]
    generated_figures = []

    for method in plot_targets:
        score_df = {
            "ssgsea": ssgsea_scores,
            "gsva_gaussian": gsva_gaussian,
            "gsva_ecdf": gsva_ecdf,
        }[method]
        label_base = {
            "ssgsea": "ssGSEA",
            "gsva_gaussian": "GSVA",
            "gsva_ecdf": "GSVA",
        }[method]
        generated_figures.extend(
            plot_and_save(args.outdir, method, score_df, meta, do_zscore=not args.no_zscore, y_label_base=label_base)
        )

    print("Done.")
    print(f"Output directory: {os.path.abspath(args.outdir)}")
    print("Generated figure files:")
    for path in generated_figures:
        print(f"  - {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
