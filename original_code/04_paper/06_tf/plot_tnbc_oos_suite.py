#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TNBC OOS plotting suite for outputs from v4grl_v2.py.

Expected files under --run-root:
  summary.csv
  crossfit_oos_pred.csv
  cv_repXX_fold00/outer_test_pred.csv  (recursively searched)
Optional (if you later log histories):
  stage0_history.csv
  stage1_history.csv
  cv_repXX_fold00/stage2_history.csv

Generated figures:
  1) fig12B_metrics_boxplot.{pdf,png}
  2) oos_score_distribution.{pdf,png}
  3) threshold_distribution.{pdf,png}
  4) threshold_tradeoff.{pdf,png}
  5) calibration_curve.{pdf,png}
  6) representative_stage2_dynamics.{pdf,png}  [only if stage2 histories exist]

Font rule:
  - Chinese: SimSun / Songti style, 10.5 pt (五号)
  - English letters / digits: Times New Roman, 10.5 pt (五号)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================
# Configurable style knobs
# =========================
FONT_SIZE = 10.5  # 五号
DPI = 300
FIGSIZE_BOX = (8.6, 3.6)
FIGSIZE_SCORE = (4.4, 3.5)
FIGSIZE_THR_HIST = (4.2, 3.4)
FIGSIZE_THR_CURVE = (4.6, 3.5)
FIGSIZE_CAL = (4.4, 3.5)
FIGSIZE_STAGE2 = (5.2, 3.8)

WHIS = (5, 95)
SHOW_POINTS = True
SHOWFLIERS = False
BOX_WIDTH = 0.52
JITTER_STD = 0.035
JITTER_CLIP = 0.10
POINT_SIZE = 14
POINT_ALPHA = 0.35
POINT_EDGE = "0.35"
POINT_LW = 0.7
RNG_SEED = 0

GRID_ALPHA = 0.14
GRID_LW = 0.6
SPINE_LW = 0.9
TICK_LW = 0.9
TICK_LEN = 3.5

BOX_COLORS = (
    "#4C72B0",  # AUC
    "#55A868",  # Precision
    "#8172B2",  # Recall
    "#E39C34",  # F1
    "#C44E52",  # ACC
)
SCORE_COLORS = ("#4C72B0", "#C44E52")
THR_LINE_COLOR = "#7A3E9D"
CAL_LINE_COLOR = "#4C72B0"

USE_CHINESE_LABELS = True

# If your environment cannot automatically find SimSun or Times New Roman,
# you can fill these paths manually.
SIMSUN_PATH = None
TNR_PATH = None


# =========================
# Font helpers
# =========================
def _first_existing_font_path(candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        try:
            p = fm.findfont(name, fallback_to_default=False)
            if p and Path(p).exists():
                return p
        except Exception:
            pass
    return None


def get_fontprops() -> Tuple[FontProperties, FontProperties]:
    global SIMSUN_PATH, TNR_PATH

    if SIMSUN_PATH is None:
        SIMSUN_PATH = _first_existing_font_path([
            "SimSun", "Songti SC", "STSong", "Noto Serif CJK SC", "Source Han Serif SC"
        ])
    if TNR_PATH is None:
        TNR_PATH = _first_existing_font_path([
            "Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"
        ])

    if SIMSUN_PATH and Path(SIMSUN_PATH).exists():
        cn = FontProperties(fname=SIMSUN_PATH, size=FONT_SIZE)
    else:
        cn = FontProperties(family="serif", size=FONT_SIZE)

    if TNR_PATH and Path(TNR_PATH).exists():
        en = FontProperties(fname=TNR_PATH, size=FONT_SIZE)
    else:
        en = FontProperties(family="serif", size=FONT_SIZE)

    return cn, en


CN_FP, EN_FP = get_fontprops()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = EN_FP.get_name()
plt.rcParams["font.size"] = FONT_SIZE


# =========================
# Generic helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rgba(c, a):
    r, g, b, _ = mcolors.to_rgba(c)
    return (r, g, b, a)


def set_axis_paper_style(ax):
    ax.grid(axis="y", linestyle="-", linewidth=GRID_LW, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)
    ax.tick_params(axis="both", width=TICK_LW, length=TICK_LEN, direction="out")


def apply_ticklabel_font(ax, axis="both", fontprops: Optional[FontProperties] = None):
    fp = fontprops or EN_FP
    if axis in ("x", "both"):
        for t in ax.get_xticklabels():
            t.set_fontproperties(fp)
    if axis in ("y", "both"):
        for t in ax.get_yticklabels():
            t.set_fontproperties(EN_FP)


def savefig(fig, out_base: Path):
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================
# File readers
# =========================
def read_summary(run_root: Path) -> pd.DataFrame:
    p = run_root / "summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"summary.csv not found under {run_root}")
    return pd.read_csv(p)


def read_crossfit(run_root: Path) -> pd.DataFrame:
    p = run_root / "crossfit_oos_pred.csv"
    if not p.exists():
        raise FileNotFoundError(f"crossfit_oos_pred.csv not found under {run_root}")
    df = pd.read_csv(p)
    rename_map = {}
    if "p_oos_mean" in df.columns:
        rename_map["p_oos_mean"] = "p"
    df = df.rename(columns=rename_map)
    need = {"id", "y", "p"}
    if not need.issubset(df.columns):
        raise ValueError(f"crossfit_oos_pred.csv must contain {need}, got {list(df.columns)}")
    df = df[["id", "y", "p"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df = df.dropna(subset=["y", "p"]).reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    return df


def find_outer_test_pred_files(run_root: Path) -> List[Path]:
    files = sorted(run_root.rglob("outer_test_pred.csv"))
    if not files:
        alt = sorted(run_root.glob("*_outer_test_pred.csv"))
        files.extend(alt)
    return files


def parse_rep_from_path(path: Path) -> Optional[int]:
    m = re.search(r"cv_rep(\d+)_fold", str(path))
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)_", path.name)
    if m:
        return int(m.group(1))
    return None


def read_stage2_history_files(run_root: Path) -> List[Tuple[int, Path]]:
    files = sorted(run_root.rglob("stage2_history.csv"))
    out = []
    for p in files:
        rep = parse_rep_from_path(p.parent) if p.parent.name.startswith("cv_rep") else parse_rep_from_path(p)
        if rep is None:
            rep = parse_rep_from_path(Path(str(p.parent)))
        if rep is None:
            continue
        out.append((rep, p))
    return sorted(out, key=lambda x: x[0])


# =========================
# Metrics per repeat
# =========================
def compute_metrics_from_pred_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    cols = list(df.columns)
    y_col = next((c for c in ["y", "y_true", "label", "target"] if c in cols), None)
    p_col = next((c for c in ["p", "prob", "proba", "y_prob", "score"] if c in cols), None)
    pred_col = next((c for c in ["pred", "y_pred", "pred_label"] if c in cols), None)
    if y_col is None or p_col is None:
        raise ValueError(f"{path}: need y and p columns, got {cols}")

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok].astype(int)
    p = p[ok].astype(float)

    if pred_col is not None:
        pred = pd.to_numeric(df.loc[ok, pred_col], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        pred = (p >= 0.5).astype(int)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else np.nan
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    acc = float(accuracy_score(y, pred))

    return {
        "rep": parse_rep_from_path(path) if parse_rep_from_path(path) is not None else -1,
        "file": str(path),
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "acc": acc,
    }


def build_metrics_df(run_root: Path) -> pd.DataFrame:
    rows = []
    for fp in find_outer_test_pred_files(run_root):
        try:
            rows.append(compute_metrics_from_pred_csv(fp))
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")
    if not rows:
        raise RuntimeError("No valid outer_test_pred.csv files found.")
    df = pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)
    return df


# =========================
# Plot 1: Fig 12B metrics boxplot
# =========================
def plot_metrics_boxplot(metrics_df: pd.DataFrame, out_dir: Path):
    cols = ["auc", "precision", "recall", "f1", "acc"]
    labels = ["AUC", "Precision", "Recall", "F1", "ACC"]
    data = [metrics_df[c].dropna().to_numpy() for c in cols]

    fig, ax = plt.subplots(figsize=FIGSIZE_BOX, dpi=DPI)
    bp = ax.boxplot(
        data,
        labels=labels,
        whis=WHIS,
        showfliers=SHOWFLIERS,
        widths=BOX_WIDTH,
        patch_artist=True,
        medianprops=dict(linewidth=1.2, color="0.15", zorder=3),
        boxprops=dict(linewidth=1.0, zorder=2),
        whiskerprops=dict(linewidth=1.0, zorder=2),
        capprops=dict(linewidth=1.0, zorder=2),
    )

    for i, b in enumerate(bp["boxes"]):
        c = BOX_COLORS[i % len(BOX_COLORS)]
        b.set_edgecolor(rgba(c, 0.90))
        b.set_facecolor(rgba(c, 0.12))

    if SHOW_POINTS:
        rng = np.random.default_rng(RNG_SEED)
        for i, y in enumerate(data, start=1):
            x = rng.normal(loc=i, scale=JITTER_STD, size=len(y))
            x = np.clip(x, i - JITTER_CLIP, i + JITTER_CLIP)
            ax.scatter(
                x, y, s=POINT_SIZE, facecolors="none", edgecolors=POINT_EDGE,
                linewidths=POINT_LW, alpha=POINT_ALPHA, zorder=1, rasterized=True
            )

    stats = []
    for arr in data:
        mu = float(np.mean(arr)) if len(arr) else np.nan
        sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        stats.append((mu, sd))

    xticklabels = []
    for lab, (mu, sd) in zip(labels, stats):
        xticklabels.append(f"{lab}\n{mu:.3f}±{sd:.3f}")
    ax.set_xticklabels(xticklabels)
    apply_ticklabel_font(ax, axis="x", fontprops=EN_FP)
    apply_ticklabel_font(ax, axis="y", fontprops=EN_FP)

    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score", fontproperties=EN_FP)
    set_axis_paper_style(ax)

    plt.tight_layout(pad=0.4)
    savefig(fig, out_dir / "fig12B_metrics_boxplot")

    summ = pd.DataFrame({
        "metric": labels,
        "mean": [s[0] for s in stats],
        "std": [s[1] for s in stats],
    })
    summ.to_csv(out_dir / "fig12B_metrics_summary.csv", index=False)


# =========================
# Plot 2: OOS score distribution by label
# =========================
def plot_oos_score_distribution(crossfit_df: pd.DataFrame, thr_median: float, out_dir: Path):
    groups = [crossfit_df.loc[crossfit_df["y"] == 0, "p"].to_numpy(),
              crossfit_df.loc[crossfit_df["y"] == 1, "p"].to_numpy()]
    labels_cn = ["非响应者", "响应者"]
    labels_en = ["Non-responder", "Responder"]
    labels = labels_cn if USE_CHINESE_LABELS else labels_en
    tick_fp = CN_FP if USE_CHINESE_LABELS else EN_FP

    fig, ax = plt.subplots(figsize=FIGSIZE_SCORE, dpi=DPI)

    vp = ax.violinplot(groups, positions=[1, 2], widths=0.68, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        c = SCORE_COLORS[i]
        body.set_facecolor(rgba(c, 0.18))
        body.set_edgecolor(rgba(c, 0.85))
        body.set_linewidth(0.9)
        body.set_alpha(1.0)

    bp = ax.boxplot(
        groups, positions=[1, 2], widths=0.32, whis=WHIS, showfliers=False, patch_artist=True,
        medianprops=dict(linewidth=1.2, color="0.15", zorder=4),
        boxprops=dict(linewidth=1.0, zorder=3),
        whiskerprops=dict(linewidth=1.0, zorder=3),
        capprops=dict(linewidth=1.0, zorder=3),
    )
    for i, b in enumerate(bp["boxes"]):
        c = SCORE_COLORS[i]
        b.set_edgecolor(rgba(c, 0.90))
        b.set_facecolor(rgba(c, 0.10))

    rng = np.random.default_rng(RNG_SEED)
    for i, y in enumerate(groups, start=1):
        x = rng.normal(loc=i, scale=0.04, size=len(y))
        x = np.clip(x, i - 0.12, i + 0.12)
        ax.scatter(
            x, y, s=15, facecolors="none", edgecolors="0.40",
            linewidths=0.7, alpha=0.5, zorder=2, rasterized=True
        )

    ax.axhline(thr_median, color=THR_LINE_COLOR, linestyle="--", linewidth=1.1)
    ax.text(
        2.35, thr_median + 0.015,
        f"median threshold = {thr_median:.3f}",
        fontproperties=EN_FP, fontsize=FONT_SIZE, color=THR_LINE_COLOR,
        ha="right", va="bottom"
    )

    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    apply_ticklabel_font(ax, axis="x", fontprops=tick_fp)
    apply_ticklabel_font(ax, axis="y", fontprops=EN_FP)

    if USE_CHINESE_LABELS:
        ax.set_ylabel("预测分数", fontproperties=CN_FP)
    else:
        ax.set_ylabel("Predicted score", fontproperties=EN_FP)

    auc = float(roc_auc_score(crossfit_df["y"].to_numpy(), crossfit_df["p"].to_numpy()))
    ax.text(0.03, 0.97, f"OOS AUC = {auc:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontproperties=EN_FP)

    set_axis_paper_style(ax)
    plt.tight_layout(pad=0.5)
    savefig(fig, out_dir / "oos_score_distribution")


# =========================
# Plot 3: Threshold distribution across repeats
# =========================
def plot_threshold_distribution(summary_df: pd.DataFrame, out_dir: Path):
    thr_col = "thr" if "thr" in summary_df.columns else "tnbc_val_thr"
    if thr_col not in summary_df.columns:
        print("[WARN] threshold column not found in summary.csv; skip threshold distribution")
        return None

    thrs = pd.to_numeric(summary_df[thr_col], errors="coerce").dropna().to_numpy()
    if len(thrs) == 0:
        print("[WARN] no valid thresholds; skip threshold distribution")
        return None

    med = float(np.median(thrs))
    mean = float(np.mean(thrs))
    q1, q3 = np.quantile(thrs, [0.25, 0.75])

    fig, ax = plt.subplots(figsize=FIGSIZE_THR_HIST, dpi=DPI)
    bins = min(12, max(6, int(round(math.sqrt(len(thrs))))))
    ax.hist(thrs, bins=bins, edgecolor="0.25", linewidth=0.8, color=rgba("#4C72B0", 0.18))
    ax.axvline(med, color=THR_LINE_COLOR, linestyle="--", linewidth=1.2)
    ax.axvline(mean, color="0.35", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Threshold", fontproperties=EN_FP)
    ax.set_ylabel("Count", fontproperties=EN_FP)
    apply_ticklabel_font(ax, axis="both", fontprops=EN_FP)
    set_axis_paper_style(ax)
    ax.text(0.98, 0.97,
            f"median = {med:.3f}\nmean = {mean:.3f}\nIQR = [{q1:.3f}, {q3:.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontproperties=EN_FP)

    plt.tight_layout(pad=0.5)
    savefig(fig, out_dir / "threshold_distribution")
    return med


# =========================
# Plot 4: Threshold tradeoff curve on OOS predictions
# =========================
def plot_threshold_tradeoff(crossfit_df: pd.DataFrame, thr_median: float, out_dir: Path):
    y = crossfit_df["y"].to_numpy().astype(int)
    p = crossfit_df["p"].to_numpy().astype(float)

    ts = np.unique(np.round(np.linspace(0.0, 1.0, 201), 4))
    precs, recs, f1s = [], [], []
    for t in ts:
        pred = (p >= t).astype(int)
        precs.append(precision_score(y, pred, zero_division=0))
        recs.append(recall_score(y, pred, zero_division=0))
        f1s.append(f1_score(y, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=FIGSIZE_THR_CURVE, dpi=DPI)
    ax.plot(ts, precs, linewidth=1.3, label="Precision")
    ax.plot(ts, recs, linewidth=1.3, label="Recall")
    ax.plot(ts, f1s, linewidth=1.3, label="F1")
    ax.axvline(thr_median, color=THR_LINE_COLOR, linestyle="--", linewidth=1.1)

    # annotate metrics at median threshold
    pred_med = (p >= thr_median).astype(int)
    prec_m = precision_score(y, pred_med, zero_division=0)
    rec_m = recall_score(y, pred_med, zero_division=0)
    f1_m = f1_score(y, pred_med, zero_division=0)

    ax.text(0.98, 0.97,
            f"median threshold = {thr_median:.3f}\nPrecision = {prec_m:.3f}\nRecall = {rec_m:.3f}\nF1 = {f1_m:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontproperties=EN_FP)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Threshold", fontproperties=EN_FP)
    ax.set_ylabel("Metric value", fontproperties=EN_FP)
    apply_ticklabel_font(ax, axis="both", fontprops=EN_FP)
    set_axis_paper_style(ax)
    leg = ax.legend(frameon=False, prop=EN_FP, loc="lower left")
    for t in leg.get_texts():
        t.set_fontproperties(EN_FP)

    plt.tight_layout(pad=0.5)
    savefig(fig, out_dir / "threshold_tradeoff")


# =========================
# Plot 5: Calibration curve + Brier score
# =========================
def plot_calibration(crossfit_df: pd.DataFrame, out_dir: Path, n_bins: int = 8):
    y = crossfit_df["y"].to_numpy().astype(int)
    p = crossfit_df["p"].to_numpy().astype(float)

    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y, p))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else np.nan

    fig, ax = plt.subplots(figsize=FIGSIZE_CAL, dpi=DPI)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.45")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=1.2, markersize=4.0, color=CAL_LINE_COLOR)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability", fontproperties=EN_FP)
    ax.set_ylabel("Observed responder fraction", fontproperties=EN_FP)
    apply_ticklabel_font(ax, axis="both", fontprops=EN_FP)
    set_axis_paper_style(ax)
    ax.text(0.98, 0.05, f"Brier = {brier:.3f}\nOOS AUC = {auc:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontproperties=EN_FP)

    plt.tight_layout(pad=0.5)
    savefig(fig, out_dir / "calibration_curve")


# =========================
# Plot 6: Representative stage2 dynamics (optional)
# =========================
def choose_representative_rep(summary_df: pd.DataFrame) -> Optional[int]:
    auc_col = "outer_auc" if "outer_auc" in summary_df.columns else None
    rep_col = "rep" if "rep" in summary_df.columns else None
    if auc_col is None or rep_col is None:
        return None
    aucs = pd.to_numeric(summary_df[auc_col], errors="coerce")
    reps = pd.to_numeric(summary_df[rep_col], errors="coerce")
    ok = aucs.notna() & reps.notna()
    if ok.sum() == 0:
        return None
    aucs = aucs[ok].to_numpy(dtype=float)
    reps = reps[ok].to_numpy(dtype=int)
    med = float(np.median(aucs))
    idx = int(np.argmin(np.abs(aucs - med)))
    return int(reps[idx])


def plot_representative_stage2(summary_df: pd.DataFrame, run_root: Path, out_dir: Path):
    rep = choose_representative_rep(summary_df)
    if rep is None:
        print("[WARN] cannot determine representative rep; skip stage2 dynamics")
        return

    hist_files = dict(read_stage2_history_files(run_root))
    if rep not in hist_files:
        print("[WARN] stage2_history.csv not found; skip stage2 dynamics")
        return

    df = pd.read_csv(hist_files[rep])
    need_cols = {"epoch", "tnbc_val_auc", "oth_val_auc", "sup_loss", "best_flag"}
    if not need_cols.issubset(df.columns):
        print(f"[WARN] {hist_files[rep]} missing columns {need_cols - set(df.columns)}; skip stage2 dynamics")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_STAGE2, dpi=DPI)
    ax.plot(df["epoch"], df["tnbc_val_auc"], linewidth=1.3, label="TNBC-val AUC")
    ax.plot(df["epoch"], df["oth_val_auc"], linewidth=1.1, label="OTH-val AUC")

    if "sup_loss" in df.columns:
        # rescale to [0,1] only for trend display, avoiding a second y-axis in a small figure
        loss = df["sup_loss"].to_numpy(dtype=float)
        if np.nanmax(loss) > np.nanmin(loss):
            loss_scaled = (loss - np.nanmin(loss)) / (np.nanmax(loss) - np.nanmin(loss))
            ax.plot(df["epoch"], loss_scaled, linewidth=1.0, linestyle=":", label="Train loss (scaled)")

    if "phase" in df.columns and (df["phase"] == 2).any():
        ep_sw = int(df.loc[df["phase"] == 2, "epoch"].min())
        ax.axvline(ep_sw, color="0.45", linestyle="--", linewidth=1.0)
        ax.text(ep_sw, 0.03, "Phase2", ha="left", va="bottom", fontproperties=EN_FP)

    best_rows = df[df["best_flag"] == 1]
    if len(best_rows) > 0:
        best_ep = int(best_rows.iloc[-1]["epoch"])
        best_auc = float(best_rows.iloc[-1]["tnbc_val_auc"])
        ax.scatter([best_ep], [best_auc], s=28, facecolors="white", edgecolors="0.1", zorder=5)
        ax.text(best_ep, best_auc + 0.03, f"best ep = {best_ep}", fontproperties=EN_FP)

    ax.set_xlim(df["epoch"].min(), df["epoch"].max())
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Epoch", fontproperties=EN_FP)
    ax.set_ylabel("Value", fontproperties=EN_FP)
    apply_ticklabel_font(ax, axis="both", fontprops=EN_FP)
    set_axis_paper_style(ax)
    leg = ax.legend(frameon=False, prop=EN_FP, loc="lower right")
    for t in leg.get_texts():
        t.set_fontproperties(EN_FP)

    plt.tight_layout(pad=0.5)
    savefig(fig, out_dir / "representative_stage2_dynamics")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=str, required=True,
                    help="Root directory of one v4grl_v2 experiment, containing summary.csv and crossfit_oos_pred.csv")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory for figures. Default: <run-root>/paper_plots")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_root / "paper_plots")
    ensure_dir(out_dir)

    summary_df = read_summary(run_root)
    crossfit_df = read_crossfit(run_root)
    metrics_df = build_metrics_df(run_root)
    metrics_df.to_csv(out_dir / "per_repeat_outer_metrics.csv", index=False)

    plot_metrics_boxplot(metrics_df, out_dir)
    thr_median = plot_threshold_distribution(summary_df, out_dir)
    if thr_median is None:
        thr_median = float(np.median(summary_df["thr"].dropna())) if "thr" in summary_df.columns else 0.5

    plot_oos_score_distribution(crossfit_df, thr_median, out_dir)
    plot_threshold_tradeoff(crossfit_df, thr_median, out_dir)
    plot_calibration(crossfit_df, out_dir)
    plot_representative_stage2(summary_df, run_root, out_dir)

    # useful scalar summary for paper writing
    y = crossfit_df["y"].to_numpy().astype(int)
    p = crossfit_df["p"].to_numpy().astype(float)
    pred = (p >= thr_median).astype(int)
    summary_json = {
        "oos_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else None,
        "median_threshold": float(thr_median),
        "precision_at_median_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_at_median_threshold": float(recall_score(y, pred, zero_division=0)),
        "f1_at_median_threshold": float(f1_score(y, pred, zero_division=0)),
        "acc_at_median_threshold": float(accuracy_score(y, pred)),
        "brier_score": float(brier_score_loss(y, p)),
        "n_samples": int(len(crossfit_df)),
        "n_responders": int((y == 1).sum()),
        "n_nonresponders": int((y == 0).sum()),
    }
    with open(out_dir / "oos_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"[OK] plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
