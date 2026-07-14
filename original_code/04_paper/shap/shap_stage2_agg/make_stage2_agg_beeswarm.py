#!/usr/bin/env python3
"""
High-resolution aggregated Stage2 (final) SHAP beeswarm.

Why "blurry" happens:
- When there are many points, SHAP/matplotlib often rasterizes the scatter layer
  inside a PDF for speed/file size. If rasterized at low DPI, zooming looks fuzzy.
Fix:
- Save with a higher DPI (e.g., 600 or 900). This increases the raster layer quality.

Usage:
  python make_stage2_agg_beeswarm_hd.py --out_root /path/to/out_root --topn 50 --dpi 900
Output:
  <out_root>/shap_stage2_agg/stage2_final_agg_beeswarm_top{topn}_dpi{dpi}.pdf
"""
import argparse, glob
from pathlib import Path
import numpy as np

def _load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    shap = d["shap"]
    X = d["X"]
    genes = d["genes"].astype(object).tolist()
    if shap.ndim == 3 and shap.shape[-1] == 1:
        shap = shap[:, :, 0]
    return shap.astype(np.float32), X.astype(np.float32), genes

def _align_to_ref(shap, X, genes, ref_genes):
    if genes == ref_genes:
        return shap, X
    ref_pos = {g:i for i,g in enumerate(ref_genes)}
    if any(g not in ref_pos for g in genes):
        missing = [g for g in genes if g not in ref_pos][:10]
        raise RuntimeError(f"Gene mismatch: some genes not found in reference. examples={missing}")
    cur_pos = {g:i for i,g in enumerate(genes)}
    reorder = [cur_pos[g] for g in ref_genes]
    return shap[:, reorder], X[:, reorder]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--dpi", type=int, default=900, help="DPI for rasterized layers inside PDF/PNG.")
    ap.add_argument("--agg_csv", type=str, default="", help="Optional path to stage2_final_agg_shap_meanabs.csv.")
    ap.add_argument("--out", type=str, default="", help="Output path. Default under <out_root>/shap_stage2_agg/")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    # Find per-run NPZ (recursive, robust to naming)
    npz_paths = sorted(glob.glob(str(out_root / "**" / "shap_stage2_final" / "*_shap_values.npz"), recursive=True))
    if not npz_paths:
        raise SystemExit(f"No per-run Stage2 SHAP NPZ found under {out_root}")

    ref_genes = None
    shap_all, X_all = [], []
    for p in npz_paths:
        shap, X, genes = _load_npz(p)
        if ref_genes is None:
            ref_genes = genes
            shap_all.append(shap); X_all.append(X)
        else:
            shap2, X2 = _align_to_ref(shap, X, genes, ref_genes)
            shap_all.append(shap2); X_all.append(X2)

    shap_all = np.concatenate(shap_all, axis=0)
    X_all = np.concatenate(X_all, axis=0)

    # Decide top gene order (prefer agg csv rank)
    gene_order = None
    agg_csv = args.agg_csv.strip()
    if not agg_csv:
        cand = out_root / "shap_stage2_agg" / "stage2_final_agg_shap_meanabs.csv"
        if cand.exists():
            agg_csv = str(cand)

    if agg_csv and Path(agg_csv).exists():
        import pandas as pd
        df = pd.read_csv(agg_csv)
        if "rank" in df.columns:
            df = df.sort_values("rank", ascending=True)
        else:
            # fallback: sort by a plausible importance column
            imp_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else df.columns[-1]
            df = df.sort_values(imp_col, ascending=False)
        gene_order = df["gene"].astype(str).tolist()
    else:
        mean_abs = np.mean(np.abs(shap_all), axis=0)
        idx = np.argsort(-mean_abs)
        gene_order = [ref_genes[i] for i in idx]

    topn = int(args.topn)
    top_genes = gene_order[:topn]
    ref_pos = {g:i for i,g in enumerate(ref_genes)}
    cols = [ref_pos[g] for g in top_genes if g in ref_pos]
    shap_top = shap_all[:, cols]
    X_top = X_all[:, cols]
    names = [ref_genes[i] for i in cols]

    out_dir = out_root / "shap_stage2_agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f"stage2_final_agg_beeswarm_top{topn}_dpi{args.dpi}.pdf")

    # Plot with SHAP if available, otherwise fallback.
    try:
        import shap
        import matplotlib.pyplot as plt
        # Ensure fonts are embedded nicely for papers
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        plt.figure(figsize=(7.4, 10.8), dpi=args.dpi)
        shap.summary_plot(
            shap_top,
            features=X_top,
            feature_names=names,
            show=False,
            max_display=topn
        )
        plt.tight_layout()
        # dpi matters for rasterized artists inside PDF
        plt.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
        plt.close()
    except Exception:
        import matplotlib.pyplot as plt
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        # simple beeswarm-like fallback
        Xc = X_top
        vmin = np.nanpercentile(Xc, 5, axis=0)
        vmax = np.nanpercentile(Xc, 95, axis=0)
        Xn = (Xc - vmin) / (vmax - vmin + 1e-6)
        Xn = np.clip(Xn, 0.0, 1.0)

        order = np.argsort(-np.mean(np.abs(shap_top), axis=0))
        shap_top = shap_top[:, order]
        Xn = Xn[:, order]
        names2 = [names[i] for i in order]

        rng = np.random.default_rng(0)
        plt.figure(figsize=(7.4, 10.8), dpi=args.dpi)
        for j in range(shap_top.shape[1]):
            s = shap_top[:, j]
            y = np.full_like(s, j, dtype=np.float32) + (rng.random(s.shape[0]).astype(np.float32) - 0.5) * 0.6
            plt.scatter(s, y, c=Xn[:, j], s=8, alpha=0.7)
        plt.axvline(0.0, linewidth=1)
        plt.yticks(range(len(names2)), names2)
        plt.gca().invert_yaxis()
        plt.xlabel("Attribution value (impact on model output)")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
        plt.close()

    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
