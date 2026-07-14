#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLICK-RUN: OOS aggregated ROC/PR (top-journal, double-column, high-res)

What it does:
  1) Read outer_test_pred.zip (50 repeats). Each CSV must contain columns: id, y, p
  2) OOS aggregation: for each sample id, average p across repeats where it appears in outer-test
  3) Plot:
     A) OOS ROC: thick mean line + bootstrap 95% CI band + chance dashed line
        In-plot text (ONE line): AUC=... (95% CI ...–...), N=...
     B) OOS PR : thick mean line + bootstrap 95% CI band
        In-plot text (ONE line): AP=... (95% CI ...–...), N=...

Output:
  OUT_DIR/
    A_oos_roc_top_v2_double.pdf / .png
    B_oos_pr_top_v2_double.pdf  / .png
    oos_aggregated_pred_mean.csv

Dependencies:
  pip install numpy pandas matplotlib scikit-learn
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# =========================
# Configurable style knobs（只改这里即可）
# =========================

# --- Input / Output ---
ZIP_PATH = Path("outer_test_pred.zip")       # 放在同目录即可；否则写绝对路径
OUT_DIR  = Path("oos_topfigs_double")        # 输出目录

# --- Columns in each repeat CSV ---
ID_COL = "id"                                # 样本 ID 列名
Y_COL  = "y"                                 # 标签列名（0/1）
P_COL  = "p"                                 # 概率列名（0~1）

# --- Figure size (double-column, bigger & clearer) ---
FIGSIZE_DOUBLE = (7.8, 3.8)                  # 你要更大就改这里，比如 (8.6, 4.2)

# --- Export quality ---
PNG_DPI = 800                                # 更清晰就调大（600~1200 都行）
PDF_FONT_TYPE = 42                           # 42=可编辑 TrueType

# --- Bootstrap settings (for 95% CI band + AUC/AP CI) ---
BOOTSTRAP_B = 2000                           # 更稳就 5000（会更慢）
SEED = 0

# --- Curve grids (for smooth mean+band) ---
FPR_GRID_N = 501
REC_GRID_N = 501

# --- Theme (blue) ---
BLUE = "#1f77b4"
BLUE_DARK = "#0b3c5d"
BLUE_LIGHT = "#8ecae6"
CI_ALPHA = 0.55
CHANCE_ALPHA = 0.75

# --- Optional font (set to .ttf path if needed) ---
TTF_PATH = None  # e.g., "/path/to/times.ttf"


# =========================
# Helpers
# =========================

def _set_mpl_style():
    mpl.rcParams.update({
        "pdf.fonttype": PDF_FONT_TYPE,
        "ps.fonttype": PDF_FONT_TYPE,
        "font.size": 9.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "savefig.dpi": PNG_DPI,
    })
    if TTF_PATH:
        from matplotlib import font_manager as fm
        fp = fm.FontProperties(fname=str(TTF_PATH))
        mpl.rcParams["font.family"] = fp.get_name()


def _stylize_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)


def _read_zip_predictions(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path.resolve()}")

    zf = zipfile.ZipFile(zip_path)
    csv_names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
    if not csv_names:
        raise ValueError(f"No CSV found in zip: {zip_path}")

    dfs = []
    for name in csv_names:
        with zf.open(name) as f:
            d = pd.read_csv(f)
        for col in (ID_COL, Y_COL, P_COL):
            if col not in d.columns:
                raise ValueError(f"{name} missing column '{col}'. Got columns: {d.columns.tolist()}")
        d = d[[ID_COL, Y_COL, P_COL]].copy()
        d.columns = ["id", "y", "p"]
        dfs.append(d)

    all_pred = pd.concat(dfs, ignore_index=True)
    all_pred["y"] = all_pred["y"].astype(int)
    all_pred["p"] = all_pred["p"].astype(float)
    return all_pred


def _oos_aggregate(all_pred: pd.DataFrame) -> pd.DataFrame:
    # Ensure labels per id are consistent across repeats
    nunique = all_pred.groupby("id")["y"].nunique().max()
    if nunique != 1:
        bad = all_pred.groupby("id")["y"].nunique()
        bad_ids = bad[bad > 1].index.tolist()[:10]
        raise ValueError(f"Inconsistent labels across repeats for some ids, e.g. {bad_ids}")

    agg = all_pred.groupby("id", as_index=False).agg(
        y=("y", "first"),
        p=("p", "mean"),
        n_votes=("p", "size"),
    )
    return agg


def _bootstrap_bands(y: np.ndarray, p: np.ndarray, B: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)

    fpr_grid = np.linspace(0, 1, FPR_GRID_N)
    rec_grid = np.linspace(0, 1, REC_GRID_N)

    tpr_bs = np.empty((B, fpr_grid.size), float)
    auc_bs = np.empty(B, float)

    prec_bs = np.empty((B, rec_grid.size), float)
    ap_bs = np.empty(B, float)

    filled = 0
    attempts = 0
    max_attempts = B * 20  # avoid infinite loops

    while filled < B and attempts < max_attempts:
        attempts += 1
        samp = rng.choice(idx, size=n, replace=True)
        yb = y[samp]
        pb = p[samp]

        # Skip if only one class appears in the bootstrap sample
        if np.unique(yb).size < 2:
            continue

        # ROC
        fpr_b, tpr_b, _ = roc_curve(yb, pb)
        auc_bs[filled] = auc(fpr_b, tpr_b)
        tpr_i = np.interp(fpr_grid, fpr_b, tpr_b)
        tpr_i[0] = 0.0
        tpr_bs[filled] = tpr_i

        # PR
        prec_b, rec_b, _ = precision_recall_curve(yb, pb)
        ap_bs[filled] = average_precision_score(yb, pb)
        order = np.argsort(rec_b)
        r = rec_b[order]
        pr = prec_b[order]
        prec_i = np.interp(rec_grid, r, pr, left=pr[0], right=pr[-1])
        prec_bs[filled] = prec_i

        filled += 1

    if filled < B:
        raise RuntimeError(f"Bootstrap failed: filled {filled}/{B} after {attempts} attempts.")

    # bands
    tpr_lo = np.percentile(tpr_bs, 2.5, axis=0)
    tpr_hi = np.percentile(tpr_bs, 97.5, axis=0)
    tpr_mean = tpr_bs.mean(axis=0)

    prec_lo = np.percentile(prec_bs, 2.5, axis=0)
    prec_hi = np.percentile(prec_bs, 97.5, axis=0)
    prec_mean = prec_bs.mean(axis=0)

    auc_ci = (float(np.percentile(auc_bs, 2.5)), float(np.percentile(auc_bs, 97.5)))
    ap_ci = (float(np.percentile(ap_bs, 2.5)), float(np.percentile(ap_bs, 97.5)))

    return (fpr_grid, tpr_lo, tpr_hi, tpr_mean, auc_ci), (rec_grid, prec_lo, prec_hi, prec_mean, ap_ci)


def _save(fig, out_pdf: Path, out_png: Path):
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=PNG_DPI)
    plt.close(fig)


def _plot_roc(out_pdf: Path, out_png: Path, fpr_grid, tpr_lo, tpr_hi, tpr_mean, auc_hat, auc_ci, n: int):
    fig = plt.figure(figsize=FIGSIZE_DOUBLE)
    ax = fig.add_subplot(111)

    ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color=BLUE_LIGHT, alpha=CI_ALPHA, linewidth=0)
    ax.plot(fpr_grid, tpr_mean, color=BLUE_DARK, linewidth=2.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color=BLUE, linewidth=1.3, alpha=CHANCE_ALPHA)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _stylize_axes(ax)

    # ONE line text only
    text = f"AUC={auc_hat:.3f} (95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}), N={n}"
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=BLUE_DARK)

    _save(fig, out_pdf, out_png)


def _plot_pr(out_pdf: Path, out_png: Path, rec_grid, prec_lo, prec_hi, prec_mean, ap_hat, ap_ci, n: int):
    fig = plt.figure(figsize=FIGSIZE_DOUBLE)
    ax = fig.add_subplot(111)

    ax.fill_between(rec_grid, prec_lo, prec_hi, color=BLUE_LIGHT, alpha=CI_ALPHA, linewidth=0)
    ax.plot(rec_grid, prec_mean, color=BLUE_DARK, linewidth=2.8)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _stylize_axes(ax)

    # ONE line text only
    text = f"AP={ap_hat:.3f} (95% CI {ap_ci[0]:.3f}–{ap_ci[1]:.3f}), N={n}"
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=BLUE_DARK)

    _save(fig, out_pdf, out_png)


# =========================
# Main (click-run)
# =========================

def main():
    _set_mpl_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pred = _read_zip_predictions(ZIP_PATH)
    agg = _oos_aggregate(all_pred)
    agg.to_csv(OUT_DIR / "oos_aggregated_pred_mean.csv", index=False)

    y = agg["y"].to_numpy().astype(int)
    p = agg["p"].to_numpy().astype(float)
    n = len(y)

    # point estimates on aggregated OOS predictions
    fpr, tpr, _ = roc_curve(y, p)
    auc_hat = auc(fpr, tpr)
    ap_hat = average_precision_score(y, p)

    roc_pack, pr_pack = _bootstrap_bands(y, p, B=BOOTSTRAP_B, seed=SEED)
    fpr_grid, tpr_lo, tpr_hi, tpr_mean, auc_ci = roc_pack
    rec_grid, prec_lo, prec_hi, prec_mean, ap_ci = pr_pack

    _plot_roc(
        out_pdf=OUT_DIR / "A_oos_roc_top_v2_double.pdf",
        out_png=OUT_DIR / "A_oos_roc_top_v2_double.png",
        fpr_grid=fpr_grid, tpr_lo=tpr_lo, tpr_hi=tpr_hi, tpr_mean=tpr_mean,
        auc_hat=auc_hat, auc_ci=auc_ci, n=n
    )

    _plot_pr(
        out_pdf=OUT_DIR / "B_oos_pr_top_v2_double.pdf",
        out_png=OUT_DIR / "B_oos_pr_top_v2_double.png",
        rec_grid=rec_grid, prec_lo=prec_lo, prec_hi=prec_hi, prec_mean=prec_mean,
        ap_hat=ap_hat, ap_ci=ap_ci, n=n
    )

    print("Done.")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print(f"AUC={auc_hat:.3f} (95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}), N={n}")
    print(f"AP ={ap_hat:.3f} (95% CI {ap_ci[0]:.3f}–{ap_ci[1]:.3f}), N={n}")


if __name__ == "__main__":
    main()