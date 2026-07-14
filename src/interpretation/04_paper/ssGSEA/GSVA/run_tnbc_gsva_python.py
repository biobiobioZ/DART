
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, rankdata, norm

def parse_gene_sets(txt_path):
    txt = open(txt_path, encoding="utf-8").read().strip()
    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    gene_sets = {}
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        name = " ".join(lines[:-1])
        name = re.sub(r"\s*\(\d+\s+genes\)\s*", "", name)
        name = " ".join(name.split())
        genes = [g.strip() for g in lines[-1].split(",") if g.strip()]
        gene_sets[name] = genes
    return gene_sets

def gaussian_kernel_cdf_per_gene(values):
    n_genes, n_samples = values.shape
    out = np.empty_like(values, dtype=float)
    for i in range(n_genes):
        x = values[i]
        sd = np.std(x, ddof=1)
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
            ecdf_vals[ord_idx[pos:end + 1]] = ecdf
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
    gset_idxs = np.array(gset_idxs, dtype=int)
    sum_gset = np.sum(np.abs(score_vec[gset_idxs]) ** tau)
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
    for a, (name, genes) in enumerate(gene_sets.items()):
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
        for a, (name, genes) in enumerate(gene_sets.items()):
            gset_idx = np.array([gene_to_idx[g] for g in genes if g in gene_to_idx], dtype=int)
            k = len(gset_idx)
            idxs = np.sort(np.array([np.where(gene_ranking == g)[0][0] for g in gset_idx]))
            step_in = np.sum(Ra[gene_ranking[idxs], j] * (n - idxs)) / np.sum(Ra[gene_ranking[idxs], j])
            step_out = (n * (n + 1) / 2 - np.sum(n - idxs)) / (n - k)
            es[a, j] = step_in - step_out
    if normalization:
        es = es / (es.max() - es.min())
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
        vals_r = long_df.loc[long_df["Response"] == 1, module]
        vals_n = long_df.loc[long_df["Response"] == 0, module]
        u, p = mannwhitneyu(vals_r, vals_n, alternative="two-sided")
        rows.append({
            "method": method_name,
            "module": module,
            "n_responder": len(vals_r),
            "n_nonresponder": len(vals_n),
            "median_responder": float(np.median(vals_r)),
            "median_nonresponder": float(np.median(vals_n)),
            "delta_median": float(np.median(vals_r) - np.median(vals_n)),
            "mean_responder": float(np.mean(vals_r)),
            "mean_nonresponder": float(np.mean(vals_n)),
            "delta_mean": float(np.mean(vals_r) - np.mean(vals_n)),
            "mannwhitney_u": float(u),
            "p_value": float(p),
        })
    stats_df = pd.DataFrame(rows)
    stats_df["fdr_bh"] = bh_adjust(stats_df["p_value"])
    stats_df["direction"] = np.where(stats_df["delta_median"] > 0, "Responder_higher", "Nonresponder_higher")
    return stats_df, long_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", required=True, help="CSV with genes in rows and samples in columns; first column is gene symbol")
    parser.add_argument("--meta", required=True, help="CSV with SampleID and Response columns")
    parser.add_argument("--gene_sets", required=True, help="Text file with module names and comma-separated genes")
    parser.add_argument("--outdir", default="tnbc_gsva_python")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    expr = pd.read_csv(args.expr).set_index("Gene")
    meta = pd.read_csv(args.meta)
    samples = [c for c in expr.columns if c in meta["SampleID"].tolist()]
    expr = expr[samples]
    meta = meta.set_index("SampleID").loc[samples].reset_index()
    gene_sets = parse_gene_sets(args.gene_sets)

    gd_gauss = gaussian_kernel_cdf_per_gene(expr.values.astype(float))
    gd_ecdf = direct_ecdf_logit_per_gene(expr.values.astype(float))

    gsva_main = gsva_from_density(expr, gene_sets, gd_gauss)
    gsva_ecdf = gsva_from_density(expr, gene_sets, gd_ecdf)
    ssgsea_scores = ssgsea(expr, gene_sets)

    main_stats, main_long = group_stats(gsva_main, meta, "GSVA_gaussianApprox")
    ecdf_stats, ecdf_long = group_stats(gsva_ecdf, meta, "GSVA_directECDF")
    ssgsea_stats, ssgsea_long = group_stats(ssgsea_scores, meta, "ssGSEA_python")

    main_long.to_csv(os.path.join(args.outdir, "tnbc_gsva_scores_gaussian_approx.csv"), index=False)
    ecdf_long.to_csv(os.path.join(args.outdir, "tnbc_gsva_scores_direct_ecdf.csv"), index=False)
    ssgsea_long.to_csv(os.path.join(args.outdir, "tnbc_ssgsea_scores_python.csv"), index=False)
    pd.concat([main_stats, ecdf_stats, ssgsea_stats], ignore_index=True).to_csv(
        os.path.join(args.outdir, "tnbc_gsva_ssgsea_summary.csv"), index=False
    )

if __name__ == "__main__":
    main()
