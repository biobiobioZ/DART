#!/usr/bin/env python3
"""
Chapter 4 Python pipeline for TNBC transfer-learning biological interpretation.

What this script does
---------------------
1. Reads the four/five CSV files you already prepared:
   - candidates_Q1Q3_sorted(母表).csv
   - delta_log_importance.csv
   - TNBC_rows.csv
   - TNBC_meta.csv
   - (optional) topk_selected_genes_with_scores.csv
2. Builds the key gene sets used in the thesis:
   - Top200 / Top100 / Top50
   - MAD-HVG800 background
3. Runs ORA for GO:BP and KEGG:
   - Preferred online mode: g:Profiler (supports custom background)
   - Optional offline mode: local GMT + hypergeometric ORA
4. Runs preranked GSEA for GO/KEGG with delta_log_importance
   - Online mode: GSEApy + Enrichr library
   - Offline mode: GSEApy + user-supplied GMT
5. Runs ssGSEA for Top200 / Top100 / Top50 on the TNBC cohort
6. Saves tables, gene lists, GMT files, and publication-ready figures.

Recommended package installation
--------------------------------
pip install pandas numpy scipy matplotlib gseapy gprofiler-official

Recommended command
-------------------
python run_ch4_python_pipeline.py \
  --candidates "candidates_Q1Q3_sorted(母表).csv" \
  --delta "delta_log_importance.csv" \
  --expr "TNBC_rows.csv" \
  --meta "TNBC_meta.csv" \
  --outdir "ch4_python_results"

Optional offline command (if you already have GMT files)
--------------------------------------------------------
python run_ch4_python_pipeline.py \
  --candidates "candidates_Q1Q3_sorted(母表).csv" \
  --delta "delta_log_importance.csv" \
  --expr "TNBC_rows.csv" \
  --meta "TNBC_meta.csv" \
  --go-gmt "GO_Biological_Process.gmt" \
  --kegg-gmt "KEGG_Human.gmt" \
  --outdir "ch4_python_results"
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, mannwhitneyu, spearmanr

# -------- optional imports checked at runtime --------
try:
    import gseapy as gp
except ImportError:
    gp = None

try:
    from gprofiler import GProfiler
except ImportError:
    GProfiler = None


# ---------------- utility helpers ----------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment without extra dependency."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(np.nan_to_num(pvals, nan=1.0))
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out


def clean_term_name(s: str, max_len: int = 90) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    # relaxed matching
    for col in df.columns:
        col_norm = re.sub(r"[^a-z0-9]+", "", col.lower())
        for cand in candidates:
            cand_norm = re.sub(r"[^a-z0-9]+", "", cand.lower())
            if cand_norm == col_norm:
                return col
    raise KeyError(f"Cannot find required column among {candidates}. Available: {list(df.columns)}")


def save_gene_list(genes: Sequence[str], path: Path) -> None:
    pd.Series(list(genes), name="gene").to_csv(path, index=False)


def write_gmt(gene_sets: Dict[str, Sequence[str]], path: Path, description: str = "NA") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name, genes in gene_sets.items():
            line = [name, description] + list(dict.fromkeys([g for g in genes if pd.notna(g)]))
            f.write("\t".join(map(str, line)) + "\n")


def parse_gmt(path: Path) -> Dict[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = [g for g in parts[2:] if g]
    return gene_sets


def resolve_enrichr_library(prefixes: Sequence[str], organism: str = "Human") -> str:
    if gp is None:
        raise ImportError("gseapy is not installed. Run: pip install gseapy")
    libs = gp.get_library_name(organism=organism)
    matches = []
    for lib in libs:
        lib_lower = lib.lower()
        if all(pref.lower() in lib_lower for pref in prefixes):
            matches.append(lib)

    if not matches:
        raise RuntimeError(
            f"Could not resolve an Enrichr library with prefixes={prefixes}. "
            f"Check gp.get_library_name(organism='{organism}') on your machine."
        )

    def library_sort_key(name: str) -> Tuple[int, str]:
        years = re.findall(r"(19|20)\d{2}", name)
        year = int(years[-1]) if years else -1
        return (year, name)

    matches = sorted(matches, key=library_sort_key, reverse=True)
    return matches[0]


# ---------------- input preparation ----------------
def load_inputs(
    candidates_path: Path,
    delta_path: Path,
    expr_path: Path,
    meta_path: Path,
    topk_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    candidates = pd.read_csv(candidates_path)
    delta = pd.read_csv(delta_path)
    expr = pd.read_csv(expr_path)
    meta = pd.read_csv(meta_path)
    topk = pd.read_csv(topk_path) if topk_path and topk_path.exists() else None
    return {"candidates": candidates, "delta": delta, "expr": expr, "meta": meta, "topk": topk}


def prepare_gene_sets(data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    candidates = data["candidates"].copy()
    delta = data["delta"].copy()
    topk = data["topk"]

    gene_col_candidates = find_col(candidates, ["gene"])
    rank_col = find_col(candidates, ["rank"])
    candidates = candidates.sort_values(rank_col).drop_duplicates(subset=[gene_col_candidates])

    top200 = candidates[gene_col_candidates].astype(str).tolist()
    top100 = top200[:100]
    top50 = top200[:50]

    delta_gene_col = find_col(delta, ["gene"])
    if topk is not None:
        topk_gene_col = find_col(topk, ["gene"])
        if "selected" in topk.columns:
            selected = topk[topk["selected"].astype(bool)][topk_gene_col].astype(str).tolist()
            mad_hvg800 = selected
        else:
            mad_hvg800 = topk[topk_gene_col].astype(str).tolist()
    else:
        mad_hvg800 = delta[delta_gene_col].astype(str).tolist()

    # Keep background unique while preserving order.
    mad_hvg800 = list(dict.fromkeys(mad_hvg800))

    gene_sets = {
        "Top200": top200,
        "Top100": top100,
        "Top50": top50,
        "MAD_HVG800": mad_hvg800,
    }
    return gene_sets


def prepare_ranked_list(delta: pd.DataFrame) -> pd.DataFrame:
    gene_col = find_col(delta, ["gene"])
    score_col = find_col(delta, ["delta_log_importance"])
    rnk = delta[[gene_col, score_col]].copy()
    rnk.columns = ["gene", "score"]
    rnk = rnk.dropna().drop_duplicates(subset=["gene"])
    rnk = rnk.sort_values("score", ascending=False)
    return rnk


def prepare_expression(expr: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_col = find_col(expr, ["Gene", "gene"])
    sample_col = find_col(meta, ["SampleID", "sample_id", "sample"])
    response_col = find_col(meta, ["Response", "response"])

    expr = expr.copy()
    expr[gene_col] = expr[gene_col].astype(str)
    expr = expr.drop_duplicates(subset=[gene_col]).set_index(gene_col)

    # Keep only samples present in metadata and align the order.
    samples = meta[sample_col].astype(str).tolist()
    missing = [s for s in samples if s not in expr.columns]
    if missing:
        raise ValueError(f"These samples appear in metadata but not in expression matrix: {missing[:10]}")
    expr = expr.loc[:, samples]

    meta = meta.copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta[response_col] = meta[response_col].map({1: "R", 0: "NR", "1": "R", "0": "NR"}).fillna(meta[response_col].astype(str))
    return expr, meta


# ---------------- ORA ----------------
def run_gprofiler_ora(
    query_genes: Sequence[str],
    background_genes: Sequence[str],
    source: str,
    organism: str = "hsapiens",
    user_threshold: float = 0.05,
) -> pd.DataFrame:
    if GProfiler is None:
        raise ImportError("gprofiler-official is not installed. Run: pip install gprofiler-official")
    gp_client = GProfiler(return_dataframe=True)
    res = gp_client.profile(
        organism=organism,
        query=list(query_genes),
        sources=[source],
        user_threshold=user_threshold,
        all_results=True,
        ordered=False,
        no_evidences=False,
        combined=False,
        no_iea=False,
        domain_scope="custom",
        significance_threshold_method="fdr",
        background=" ".join(background_genes),
    )
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=["source", "native", "name", "p_value"])
    return res


def run_local_ora(
    query_genes: Sequence[str],
    background_genes: Sequence[str],
    gmt_path: Path,
    source_label: str,
    min_size: int = 5,
    max_size: int = 500,
) -> pd.DataFrame:
    query = set(g for g in query_genes if pd.notna(g))
    bg = set(g for g in background_genes if pd.notna(g))
    gene_sets = parse_gmt(gmt_path)

    records = []
    M = len(bg)
    N = len(query & bg)
    if N == 0 or M == 0:
        return pd.DataFrame(columns=["source", "native", "name", "p_value"])

    for term, genes in gene_sets.items():
        genes_bg = set(genes) & bg
        n = len(genes_bg)
        if n < min_size or n > max_size:
            continue
        overlap = query & genes_bg
        k = len(overlap)
        if k == 0:
            continue
        pval = hypergeom.sf(k - 1, M, n, N)
        records.append(
            {
                "source": source_label,
                "native": term,
                "name": term,
                "p_value": pval,
                "term_size": n,
                "query_size": N,
                "intersection_size": k,
                "effective_domain_size": M,
                "precision": k / N if N else np.nan,
                "recall": k / n if n else np.nan,
                "intersections": ",".join(sorted(overlap)),
            }
        )

    res = pd.DataFrame(records)
    if len(res) == 0:
        return pd.DataFrame(columns=["source", "native", "name", "p_value"])
    res["p_value_adj"] = bh_fdr(res["p_value"].values)
    res = res.sort_values(["p_value_adj", "p_value", "intersection_size"], ascending=[True, True, False])
    return res


def plot_ora_dotplot(df: pd.DataFrame, title: str, outpath: Path, topn: int = 20) -> None:
    if df.empty:
        return
    df = df.copy()

    p_col = "p_value_adj" if "p_value_adj" in df.columns else "p_value"
    name_col = "name" if "name" in df.columns else "native"
    inter_col = "intersection_size" if "intersection_size" in df.columns else None
    qsize_col = "query_size" if "query_size" in df.columns else None

    df = df.sort_values(p_col, ascending=True).head(topn).copy()
    df[name_col] = df[name_col].astype(str).map(lambda x: clean_term_name(x, max_len=75))
    df["minus_log10_p"] = -np.log10(df[p_col].clip(lower=1e-300))
    if inter_col and qsize_col:
        df["gene_ratio"] = df[inter_col] / df[qsize_col].replace(0, np.nan)
        sizes = df[inter_col] * 18
        x = df["gene_ratio"]
        xlabel = "GeneRatio"
    else:
        sizes = np.repeat(80, len(df))
        x = df["minus_log10_p"]
        xlabel = "-log10(adjusted p)"

    fig_h = max(5, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    y = np.arange(len(df))[::-1]
    sc = ax.scatter(x, y, s=sizes, c=df["minus_log10_p"], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df[name_col])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="-log10(adjusted p)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- GSEA ----------------
def run_prerank(
    ranked_df: pd.DataFrame,
    gene_sets: str | Dict[str, Sequence[str]],
    outdir: Path,
    threads: int = 4,
    seed: int = 123,
    permutation_num: int = 1000,
) -> object:
    if gp is None:
        raise ImportError("gseapy is not installed. Run: pip install gseapy")
    return gp.prerank(
        rnk=ranked_df,
        gene_sets=gene_sets,
        outdir=str(outdir),
        min_size=5,
        max_size=500,
        permutation_num=permutation_num,
        ascending=False,
        threads=threads,
        seed=seed,
        no_plot=True,
        verbose=True,
        format="png",
    )


def plot_gsea_nes_bar(res2d: pd.DataFrame, title: str, outpath: Path, top_each: int = 10) -> None:
    if res2d is None or len(res2d) == 0:
        return
    df = res2d.copy()
    term_col = find_col(df, ["Term", "term"])
    nes_col = find_col(df, ["NES", "nes"])
    # FDR column name varies slightly across versions.
    fdr_candidates = ["FDR q-val", "FDR q value", "FDR", "fdr", "FDR q-vals"]
    fdr_col = None
    for cand in fdr_candidates:
        try:
            fdr_col = find_col(df, [cand])
            break
        except KeyError:
            continue
    if fdr_col is None:
        # fallback to nominal p-value if FDR not found
        fdr_col = find_col(df, ["NOM p-val", "pval", "P-value", "p_value"])

    df = df[[term_col, nes_col, fdr_col]].copy()
    df.columns = ["Term", "NES", "FDR"]
    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    df = df.dropna(subset=["NES", "FDR"])

    sig = df[df["FDR"] <= 0.25].copy()
    target = sig if len(sig) > 0 else df.copy()

    pos = target[target["NES"] > 0].sort_values("NES", ascending=False).head(top_each)
    neg = target[target["NES"] < 0].sort_values("NES", ascending=True).head(top_each)
    plot_df = pd.concat([neg, pos], axis=0)
    if plot_df.empty:
        return
    plot_df["Term"] = plot_df["Term"].astype(str).map(lambda x: clean_term_name(x, max_len=75))

    fig_h = max(5, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["NES"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Term"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("NES")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- ssGSEA ----------------
def run_ssgsea(
    expr: pd.DataFrame,
    gene_sets: str | Dict[str, Sequence[str]],
    outdir: Path,
    threads: int = 4,
    seed: int = 123,
) -> object:
    if gp is None:
        raise ImportError("gseapy is not installed. Run: pip install gseapy")
    return gp.ssgsea(
        data=expr,
        gene_sets=gene_sets,
        outdir=str(outdir),
        sample_norm_method="rank",
        correl_norm_type="rank",
        min_size=5,
        max_size=5000,
        permutation_num=0,
        weight=0.25,
        ascending=False,
        threads=threads,
        seed=seed,
        no_plot=True,
        verbose=True,
        format="png",
    )


def reshape_ssgsea_res(res2d: pd.DataFrame) -> pd.DataFrame:
    # GSEApy typically returns a long table with columns like: Name, Term, ES, NES
    name_col = find_col(res2d, ["Name", "Sample", "sample", "name"])
    term_col = find_col(res2d, ["Term", "term"])
    score_col = None
    for cand in ["ES", "es", "NES", "nes"]:
        try:
            score_col = find_col(res2d, [cand])
            break
        except KeyError:
            continue
    if score_col is None:
        raise KeyError(f"Cannot identify ssGSEA score column in {list(res2d.columns)}")
    wide = res2d.pivot(index=term_col, columns=name_col, values=score_col)
    return wide


def compare_groups(score_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    sample_col = find_col(meta, ["SampleID", "sample_id", "sample"])
    resp_col = find_col(meta, ["Response", "response"])
    meta2 = meta[[sample_col, resp_col]].copy()
    meta2.columns = ["SampleID", "Response"]

    rows = []
    for term in score_df.index:
        vals = score_df.loc[term]
        merged = pd.DataFrame({"SampleID": vals.index.astype(str), "score": vals.values}).merge(meta2, on="SampleID", how="inner")
        r = merged.loc[merged["Response"] == "R", "score"].dropna().astype(float)
        nr = merged.loc[merged["Response"] == "NR", "score"].dropna().astype(float)
        if len(r) == 0 or len(nr) == 0:
            continue
        stat, p = mannwhitneyu(r, nr, alternative="two-sided")
        rows.append(
            {
                "Term": term,
                "n_R": len(r),
                "n_NR": len(nr),
                "median_R": np.median(r),
                "median_NR": np.median(nr),
                "delta_median_R_minus_NR": np.median(r) - np.median(nr),
                "u_statistic": stat,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["p_value_adj"] = bh_fdr(out["p_value"].values)
        out = out.sort_values(["p_value_adj", "p_value"])
    return out


def maybe_model_score_col(meta: pd.DataFrame) -> Optional[str]:
    for cand in ["model_score", "pred_score", "pred_prob", "probability", "oos_prob", "crossfit_oos_prob"]:
        if cand in meta.columns:
            return cand
    return None


def correlate_with_model_score(score_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    score_col_name = maybe_model_score_col(meta)
    if score_col_name is None:
        return pd.DataFrame()
    sample_col = find_col(meta, ["SampleID", "sample_id", "sample"])
    rows = []
    for term in score_df.index:
        vals = score_df.loc[term]
        merged = pd.DataFrame({"SampleID": vals.index.astype(str), "score": vals.values}).merge(
            meta[[sample_col, score_col_name]], left_on="SampleID", right_on=sample_col, how="inner"
        )
        if len(merged) < 3:
            continue
        rho, p = spearmanr(merged["score"].astype(float), merged[score_col_name].astype(float))
        rows.append({"Term": term, "spearman_rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["p_value_adj"] = bh_fdr(out["p_value"].values)
        out = out.sort_values(["p_value_adj", "p_value"])
    return out


def plot_signature_boxplot(score_df: pd.DataFrame, meta: pd.DataFrame, term: str, outpath: Path) -> None:
    sample_col = find_col(meta, ["SampleID", "sample_id", "sample"])
    resp_col = find_col(meta, ["Response", "response"])
    if term not in score_df.index:
        return
    vals = score_df.loc[term]
    merged = pd.DataFrame({"SampleID": vals.index.astype(str), "score": vals.values}).merge(
        meta[[sample_col, resp_col]], left_on="SampleID", right_on=sample_col, how="inner"
    )
    merged["Response"] = merged[resp_col].map({1: "R", 0: "NR", "1": "R", "0": "NR"}).fillna(merged[resp_col].astype(str))
    r = merged.loc[merged["Response"] == "R", "score"].astype(float).values
    nr = merged.loc[merged["Response"] == "NR", "score"].astype(float).values
    if len(r) == 0 or len(nr) == 0:
        return
    _, p = mannwhitneyu(r, nr, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.boxplot([nr, r], labels=["NR", "R"], patch_artist=False)
    ax.scatter(np.random.normal(1, 0.03, len(nr)), nr, alpha=0.7, s=18)
    ax.scatter(np.random.normal(2, 0.03, len(r)), r, alpha=0.7, s=18)
    ax.set_ylabel("ssGSEA score")
    ax.set_title(f"{term} (Mann–Whitney p={p:.3g})")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- main workflow ----------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Python final pipeline for Chapter 4 GO/KEGG/GSEA/ssGSEA.")
    parser.add_argument("--candidates", type=Path, required=True, help="candidates_Q1Q3_sorted(母表).csv")
    parser.add_argument("--delta", type=Path, required=True, help="delta_log_importance.csv")
    parser.add_argument("--expr", type=Path, required=True, help="TNBC_rows.csv")
    parser.add_argument("--meta", type=Path, required=True, help="TNBC_meta.csv")
    parser.add_argument("--topk", type=Path, default=None, help="Optional: topk_selected_genes_with_scores.csv")
    parser.add_argument("--outdir", type=Path, default=Path("ch4_python_results"))
    parser.add_argument("--organism", type=str, default="hsapiens", help="g:Profiler organism code, default: hsapiens")
    parser.add_argument("--enrichr-organism", type=str, default="Human", help="GSEApy/Enrichr organism, default: Human")
    parser.add_argument("--go-gmt", type=Path, default=None, help="Optional offline GO BP GMT file")
    parser.add_argument("--kegg-gmt", type=Path, default=None, help="Optional offline KEGG GMT file")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-ora", action="store_true")
    parser.add_argument("--skip-gsea", action="store_true")
    parser.add_argument("--skip-ssgsea", action="store_true")
    args = parser.parse_args()

    outdir = safe_mkdir(args.outdir)
    prepared_dir = safe_mkdir(outdir / "00_prepared_inputs")
    ora_dir = safe_mkdir(outdir / "01_ORA")
    gsea_dir = safe_mkdir(outdir / "02_GSEA")
    ssgsea_dir = safe_mkdir(outdir / "03_ssGSEA")
    fig_dir = safe_mkdir(outdir / "figures")

    eprint("[1/6] Loading input tables ...")
    data = load_inputs(args.candidates, args.delta, args.expr, args.meta, args.topk)

    eprint("[2/6] Preparing Top200 / Top100 / Top50 / MAD-HVG800 ...")
    gene_sets = prepare_gene_sets(data)
    ranked_df = prepare_ranked_list(data["delta"])
    expr, meta = prepare_expression(data["expr"], data["meta"])

    # Save prepared inputs
    save_gene_list(gene_sets["Top200"], prepared_dir / "Top200_genes.csv")
    save_gene_list(gene_sets["Top100"], prepared_dir / "Top100_genes.csv")
    save_gene_list(gene_sets["Top50"], prepared_dir / "Top50_genes.csv")
    save_gene_list(gene_sets["MAD_HVG800"], prepared_dir / "MAD_HVG800_background.csv")
    ranked_df.to_csv(prepared_dir / "MAD_HVG800_ranked_list.csv", index=False)

    signature_sets = {
        "Top200_signature": gene_sets["Top200"],
        "Top100_signature": gene_sets["Top100"],
        "Top50_signature": gene_sets["Top50"],
    }
    signature_gmt = prepared_dir / "candidate_signatures.gmt"
    write_gmt(signature_sets, signature_gmt)

    # ---- ORA ----
    if not args.skip_ora:
        eprint("[3/6] Running ORA ...")
        for level in ["Top200", "Top100"]:
            q = gene_sets[level]
            bg = gene_sets["MAD_HVG800"]

            # GO BP
            if args.go_gmt is not None:
                go_res = run_local_ora(q, bg, args.go_gmt, source_label="GO:BP")
            else:
                go_res = run_gprofiler_ora(q, bg, source="GO:BP", organism=args.organism)
            go_res.to_csv(ora_dir / f"{level}_GO_BP_ORA.csv", index=False)
            plot_ora_dotplot(
                go_res,
                title=f"{level} GO:BP ORA",
                outpath=fig_dir / f"{level}_GO_BP_ORA_dotplot.png",
                topn=20,
            )

            # KEGG
            if args.kegg_gmt is not None:
                kegg_res = run_local_ora(q, bg, args.kegg_gmt, source_label="KEGG")
            else:
                kegg_res = run_gprofiler_ora(q, bg, source="KEGG", organism=args.organism)
            kegg_res.to_csv(ora_dir / f"{level}_KEGG_ORA.csv", index=False)
            plot_ora_dotplot(
                kegg_res,
                title=f"{level} KEGG ORA",
                outpath=fig_dir / f"{level}_KEGG_ORA_dotplot.png",
                topn=20,
            )

    # ---- GSEA ----
    if not args.skip_gsea:
        eprint("[4/6] Running preranked GSEA ...")
        if gp is None:
            raise ImportError("gseapy is required for GSEA. Run: pip install gseapy")

        if args.go_gmt is not None:
            go_gene_sets = str(args.go_gmt)
            go_label = args.go_gmt.stem
        else:
            go_lib = resolve_enrichr_library(["GO", "Biological", "Process"], organism=args.enrichr_organism)
            go_gene_sets = go_lib
            go_label = go_lib

        if args.kegg_gmt is not None:
            kegg_gene_sets = str(args.kegg_gmt)
            kegg_label = args.kegg_gmt.stem
        else:
            kegg_lib = resolve_enrichr_library(["KEGG", "Human"], organism=args.enrichr_organism)
            kegg_gene_sets = kegg_lib
            kegg_label = kegg_lib

        go_pre = run_prerank(ranked_df, go_gene_sets, gsea_dir / "GO_BP_prerank", threads=args.threads, seed=args.seed)
        go_pre.res2d.to_csv(gsea_dir / "GO_BP_prerank_results.csv", index=False)
        plot_gsea_nes_bar(
            go_pre.res2d,
            title=f"Preranked GSEA: {go_label}",
            outpath=fig_dir / "GO_BP_prerank_NES_barplot.png",
            top_each=10,
        )

        kegg_pre = run_prerank(ranked_df, kegg_gene_sets, gsea_dir / "KEGG_prerank", threads=args.threads, seed=args.seed)
        kegg_pre.res2d.to_csv(gsea_dir / "KEGG_prerank_results.csv", index=False)
        plot_gsea_nes_bar(
            kegg_pre.res2d,
            title=f"Preranked GSEA: {kegg_label}",
            outpath=fig_dir / "KEGG_prerank_NES_barplot.png",
            top_each=10,
        )

    # ---- ssGSEA ----
    if not args.skip_ssgsea:
        eprint("[5/6] Running ssGSEA on Top200 / Top100 / Top50 signatures ...")
        ss = run_ssgsea(expr, signature_sets, ssgsea_dir / "candidate_signature_ssgsea", threads=args.threads, seed=args.seed)
        ss.res2d.to_csv(ssgsea_dir / "candidate_signature_ssgsea_long.csv", index=False)

        score_wide = reshape_ssgsea_res(ss.res2d)
        score_wide.to_csv(ssgsea_dir / "candidate_signature_ssgsea_wide.csv")

        comp = compare_groups(score_wide, meta)
        comp.to_csv(ssgsea_dir / "candidate_signature_group_comparison.csv", index=False)

        corr = correlate_with_model_score(score_wide, meta)
        if len(corr) > 0:
            corr.to_csv(ssgsea_dir / "candidate_signature_model_score_correlation.csv", index=False)

        for term in ["Top200_signature", "Top100_signature", "Top50_signature"]:
            if term in score_wide.index:
                plot_signature_boxplot(score_wide, meta, term=term, outpath=fig_dir / f"{term}_ssGSEA_boxplot.png")

    eprint("[6/6] Done.")
    print(f"\nAll outputs saved to: {outdir.resolve()}")
    print("Key files to use in the thesis:")
    print(f"  - {prepared_dir / 'Top200_genes.csv'}")
    print(f"  - {ora_dir / 'Top200_GO_BP_ORA.csv'}")
    print(f"  - {ora_dir / 'Top200_KEGG_ORA.csv'}")
    print(f"  - {gsea_dir / 'GO_BP_prerank_results.csv'}")
    print(f"  - {gsea_dir / 'KEGG_prerank_results.csv'}")
    print(f"  - {ssgsea_dir / 'candidate_signature_group_comparison.csv'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
