#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_refined.py (click-run version)

Paper-ready baseline evaluation (no CLI):
- Train ML baselines on OTH (source domain) with leakage-free CV:
  scaler is inside Pipeline; CV metric = ROC-AUC.
- Evaluate each model on:
  (1) OTH internal TEST split
  (2) TNBC external TEST set
- Save (under OUTDIR):
  - summary_ml_baselines.csv with all metrics
  - per-model pipelines (scaler + model) as joblib pkl
  - ROC curves (OTH-test and TNBC-test)
  - metrics heatmaps + radar charts (paper-friendly)
  - meta (feature columns + label mapping)

Usage: right-click run (or: python model_refined.py)
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix
)

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# =========================
# Configurable knobs (edit here)
# =========================
OTH_CSV   = "OTH.csv"
TNBC_CSV  = "TNBCalcom.csv"   # change to your TNBC csv path/name if needed
LABEL_COL = "Response"

OUTDIR = "out_ml_baselines_clickrun"

SEED = 42
OTH_TEST_SIZE = 0.30   # OTH internal test split
CV_FOLDS = 5           # CV folds on OTH-train
TOPN_PLOT = 6          # plot top-N models (avoid overcrowded paper figures)

# If your labels are strings and you want to FORCE which class is positive (encoded as 1),
# set POSITIVE_CLASS to the exact label string, e.g. "Responder" or "pCR".
# If None, script will auto-encode with LabelEncoder (positive class might be arbitrary).
POSITIVE_CLASS = None


# -------------------------
# Optional deps: lightgbm / xgboost
# -------------------------
def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        return LGBMClassifier
    except Exception as e:
        warnings.warn(f"[WARN] lightgbm not available, skip LGBM. ({e})")
        return None


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier
    except Exception as e:
        warnings.warn(f"[WARN] xgboost not available, skip XGBoost. ({e})")
        return None


def set_seed(seed: int):
    np.random.seed(seed)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf -> NaN, then fill NaN by column mean (numeric columns only)."""
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    return df.fillna(0.0)


def encode_binary_labels(y_oth: pd.Series, y_tnbc: pd.Series, positive_class=None):
    """
    Ensure labels are {0,1} and consistent across OTH/TNBC.

    If labels already numeric {0,1} -> keep.
    Else:
      - if positive_class is provided: map that class to 1, the other to 0
      - otherwise: use LabelEncoder over combined labels (warn: positive class may be arbitrary)
    Returns: (y_oth_enc, y_tnbc_enc, mapping_dict)
    """
    y_all = pd.concat([y_oth, y_tnbc], axis=0)

    # already numeric 0/1?
    try:
        uniq = set(int(v) for v in pd.unique(y_all.dropna()))
        if uniq == {0, 1}:
            mapping = {0: 0, 1: 1}
            return y_oth.astype(int).values, y_tnbc.astype(int).values, mapping
    except Exception:
        pass

    y_all_str = y_all.astype(str)
    uniq_str = list(pd.unique(y_all_str.dropna()))
    if len(uniq_str) != 2:
        warnings.warn(f"[WARN] labels are not binary (unique={uniq_str}); metrics like AUC/ROC may be invalid.")

    if positive_class is not None:
        pos = str(positive_class)
        if pos not in set(uniq_str):
            raise ValueError(f"POSITIVE_CLASS='{pos}' not found in labels {uniq_str}")
        neg = [c for c in uniq_str if c != pos]
        if len(neg) != 1:
            raise ValueError(f"Cannot infer negative class from labels: {uniq_str}")
        neg = neg[0]
        mapping = {neg: 0, pos: 1}
        y_oth_enc = y_oth.astype(str).map(mapping).astype(int).values
        y_tnbc_enc = y_tnbc.astype(str).map(mapping).astype(int).values
        return y_oth_enc, y_tnbc_enc, mapping

    # default: LabelEncoder
    warnings.warn("[WARN] POSITIVE_CLASS is None. LabelEncoder will decide 0/1 order by class name. "
                  "If you care about 'responder=1', set POSITIVE_CLASS at the top of this file.")
    le = LabelEncoder()
    le.fit(y_all_str)
    y_oth_enc = le.transform(y_oth.astype(str))
    y_tnbc_enc = le.transform(y_tnbc.astype(str))
    mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
    return y_oth_enc, y_tnbc_enc, mapping


def get_score(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Score for AUC/ROC:
    - prefer predict_proba[:,1]
    - else decision_function
    - else fallback to predicted label (not ideal, but avoids crash)
    """
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            return pipeline.decision_function(X)
        except Exception:
            pass
    return pipeline.predict(X).astype(float)


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y_true: np.ndarray):
    """Return metrics dict + roc curve arrays (fpr,tpr)."""
    y_score = get_score(pipeline, X)
    y_pred = pipeline.predict(X)

    auc = np.nan
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr = None, None
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)

    metrics = {
        "auc": float(auc) if auc == auc else np.nan,
        "acc": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else None,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else None,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else None,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else None,
    }
    return metrics, (fpr, tpr)


def build_models(seed: int):
    """Return dict[name] = estimator (NOT pipeline)."""
    models = {}
    models["GBDT"] = GradientBoostingClassifier(random_state=seed)
    models["RandomForest"] = RandomForestClassifier(random_state=seed)
    models["AdaBoost"] = AdaBoostClassifier(random_state=seed)

    # SVM: probability=True to enable predict_proba
    models["SVM_linear"] = SVC(kernel="linear", probability=True, random_state=seed)

    models["MLP"] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)

    # Bagging: sklearn API differences
    base_tree = DecisionTreeClassifier(random_state=seed)
    try:
        models["BaggingTree"] = BaggingClassifier(estimator=base_tree, n_estimators=10, random_state=seed)
    except TypeError:
        models["BaggingTree"] = BaggingClassifier(base_estimator=base_tree, n_estimators=10, random_state=seed)

    LGBMClassifier = _try_import_lightgbm()
    if LGBMClassifier is not None:
        models["LGBM"] = LGBMClassifier(random_state=seed)

    XGBClassifier = _try_import_xgboost()
    if XGBClassifier is not None:
        try:
            models["XGBoost"] = XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        except TypeError:
            models["XGBoost"] = XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
            )

    return models


def _plot_roc_curves(roc_dict, title, out_png: Path, out_pdf: Path):
    if not roc_dict:
        return
    plt.figure(figsize=(8.5, 6.5))
    for name, (fpr, tpr, auc) in roc_dict.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


def _plot_metrics_heatmap(df_metrics: pd.DataFrame, title: str, out_png: Path, out_pdf: Path):
    """
    df_metrics: index=model, columns=[auc,acc,precision,recall,f1]
    """
    metrics = ["auc", "acc", "precision", "recall", "f1"]
    df = df_metrics[metrics].copy().fillna(0.0)

    fig = plt.figure(figsize=(9.2, max(3.4, 0.35 * len(df) + 1.7)))
    ax = fig.add_subplot(111)
    im = ax.imshow(df.values, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], rotation=0)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index.tolist())

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


def _plot_radar(df_metrics: pd.DataFrame, title: str, out_png: Path, out_pdf: Path, topn: int = 3):
    """
    Radar chart for top-N models (by AUC) on a given domain.
    df_metrics: index=model, columns include [auc,acc,precision,recall,f1]
    """
    metrics = ["auc", "acc", "precision", "recall", "f1"]
    df = df_metrics[metrics].copy().fillna(0.0)

    df = df.sort_values("auc", ascending=False).head(max(1, topn))
    labels = [m.upper() for m in metrics]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    ax.set_title(title, y=1.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.0)

    for model_name, row in df.iterrows():
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.10)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


def _plot_domain_shift_radar(oth_row: pd.Series, tnbc_row: pd.Series, model_name: str, out_png: Path, out_pdf: Path):
    """
    One model: compare OTH-test vs TNBC-test with two polygons.
    """
    metrics = ["auc", "acc", "precision", "recall", "f1"]
    labels = [m.upper() for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    def _vals(r):
        v = [float(r.get(m, 0.0)) if pd.notna(r.get(m, 0.0)) else 0.0 for m in metrics]
        v += v[:1]
        return v

    v_oth = _vals(oth_row)
    v_tnbc = _vals(tnbc_row)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    ax.set_title(f"Domain Shift (OTH vs TNBC) | {model_name}", y=1.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.0)

    ax.plot(angles, v_oth, linewidth=2, label="OTH-test")
    ax.fill(angles, v_oth, alpha=0.10)

    ax.plot(angles, v_tnbc, linewidth=2, label="TNBC-test")
    ax.fill(angles, v_tnbc, alpha=0.10)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


def main():
    set_seed(SEED)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Step 0: Load OTH + TNBC
    # =========================
    oth_data = pd.read_csv(OTH_CSV)
    tnbc_data = pd.read_csv(TNBC_CSV)

    if LABEL_COL not in oth_data.columns:
        raise ValueError(f"OTH file has no label column '{LABEL_COL}'.")
    if LABEL_COL not in tnbc_data.columns:
        raise ValueError(f"TNBC file has no label column '{LABEL_COL}', cannot compute AUC/ROC.")

    X_oth = oth_data.drop(columns=[LABEL_COL])
    y_oth = oth_data[LABEL_COL]

    X_tnbc = tnbc_data.drop(columns=[LABEL_COL])
    y_tnbc = tnbc_data[LABEL_COL]

    # =========================
    # Step 0.5: Align feature columns
    # =========================
    common_cols = X_oth.columns.intersection(X_tnbc.columns)
    if len(common_cols) == 0:
        raise ValueError("OTH and TNBC share 0 common feature columns. Check gene/feature names.")
    common_cols = [c for c in X_oth.columns if c in set(common_cols)]

    X_oth = clean_numeric(X_oth[common_cols])
    X_tnbc = clean_numeric(X_tnbc[common_cols])

    # =========================
    # Step 0.6: Encode labels to {0,1}
    # =========================
    y_oth_enc, y_tnbc_enc, label_mapping = encode_binary_labels(y_oth, y_tnbc, positive_class=POSITIVE_CLASS)

    meta = {"feature_cols": common_cols, "label_mapping": label_mapping, "label_col": LABEL_COL}
    joblib.dump(meta, outdir / "meta_feature_cols_and_labels.pkl")
    print(f"[OK] saved {outdir/'meta_feature_cols_and_labels.pkl'} | n_features={len(common_cols)}")

    # =========================
    # Step 1: Split OTH train/test
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_oth, y_oth_enc,
        test_size=OTH_TEST_SIZE,
        random_state=SEED,
        stratify=y_oth_enc
    )

    print("\n================= Data =================")
    print(f"[DATA] OTH: n={len(X_oth)} | train={len(X_train)} test={len(X_test)}")
    print(f"[DATA] TNBC external test: n={len(X_tnbc)}")
    print(f"[CV] folds={CV_FOLDS} | metric=roc_auc")

    # =========================
    # Step 2: Define models
    # =========================
    models = build_models(SEED)
    if len(models) == 0:
        raise RuntimeError("No models available. Please check sklearn/lightgbm/xgboost installation.")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    summary_rows = []
    roc_oth = {}
    roc_tnbc = {}

    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("\n================= Training on OTH (source) =================")
    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        # leakage-free CV on OTH-train
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        pipe.fit(X_train, y_train)

        # Evaluate on OTH-test
        oth_metrics, (fpr_o, tpr_o) = evaluate(pipe, X_test, y_test)
        # Evaluate on TNBC-test
        tnbc_metrics, (fpr_t, tpr_t) = evaluate(pipe, X_tnbc, y_tnbc_enc)

        # Save pipeline (scaler + model)
        model_path = outdir / f"{name}_pipeline.pkl"
        joblib.dump(pipe, model_path)

        # Save prediction scores (for calibration/threshold analysis)
        pd.DataFrame({"y_true": y_test, "y_score": get_score(pipe, X_test)}).to_csv(
            pred_dir / f"{name}_OTH_test_scores.csv", index=False
        )
        pd.DataFrame({"y_true": y_tnbc_enc, "y_score": get_score(pipe, X_tnbc)}).to_csv(
            pred_dir / f"{name}_TNBC_test_scores.csv", index=False
        )

        print(f"[{name}] CV_AUC={cv_mean:.4f}±{cv_std:.4f} | "
              f"OTH_TEST_AUC={oth_metrics['auc']:.4f} | TNBC_TEST_AUC={tnbc_metrics['auc']:.4f} | "
              f"saved={model_path.name}")

        if fpr_o is not None and tpr_o is not None:
            roc_oth[name] = (fpr_o, tpr_o, oth_metrics["auc"])
        if fpr_t is not None and tpr_t is not None:
            roc_tnbc[name] = (fpr_t, tpr_t, tnbc_metrics["auc"])

        summary_rows.append({
            "model": name,
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std,

            "oth_test_auc": oth_metrics["auc"],
            "oth_test_acc": oth_metrics["acc"],
            "oth_test_precision": oth_metrics["precision"],
            "oth_test_recall": oth_metrics["recall"],
            "oth_test_f1": oth_metrics["f1"],
            "oth_test_tn": oth_metrics["tn"],
            "oth_test_fp": oth_metrics["fp"],
            "oth_test_fn": oth_metrics["fn"],
            "oth_test_tp": oth_metrics["tp"],

            "tnbc_test_auc": tnbc_metrics["auc"],
            "tnbc_test_acc": tnbc_metrics["acc"],
            "tnbc_test_precision": tnbc_metrics["precision"],
            "tnbc_test_recall": tnbc_metrics["recall"],
            "tnbc_test_f1": tnbc_metrics["f1"],
            "tnbc_test_tn": tnbc_metrics["tn"],
            "tnbc_test_fp": tnbc_metrics["fp"],
            "tnbc_test_fn": tnbc_metrics["fn"],
            "tnbc_test_tp": tnbc_metrics["tp"],
        })

    # =========================
    # Step 3: Save summary
    # =========================
    summary_df = pd.DataFrame(summary_rows).sort_values(by="tnbc_test_auc", ascending=False)
    summary_csv = outdir / "summary_ml_baselines.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[OK] saved summary -> {summary_csv}")

    # =========================
    # Step 4: Paper-friendly plots
    # =========================
    _plot_roc_curves(roc_oth,  "ROC Curves on OTH Internal Test Set",
                     outdir / "roc_oth_test_all_models.png", outdir / "roc_oth_test_all_models.pdf")
    _plot_roc_curves(roc_tnbc, "ROC Curves on TNBC External Test Set",
                     outdir / "roc_tnbc_test_all_models.png", outdir / "roc_tnbc_test_all_models.pdf")
    if roc_oth:
        print("[OK] saved ROC plots -> roc_oth_test_all_models.(png/pdf)")
    if roc_tnbc:
        print("[OK] saved ROC plots -> roc_tnbc_test_all_models.(png/pdf)")

    # Top-N models by TNBC AUC for cleaner plots
    top_models = summary_df["model"].head(max(1, TOPN_PLOT)).tolist()

    oth_mat = summary_df.set_index("model").loc[top_models, [
        "oth_test_auc", "oth_test_acc", "oth_test_precision", "oth_test_recall", "oth_test_f1"
    ]].rename(columns=lambda c: c.replace("oth_test_", ""))

    tnbc_mat = summary_df.set_index("model").loc[top_models, [
        "tnbc_test_auc", "tnbc_test_acc", "tnbc_test_precision", "tnbc_test_recall", "tnbc_test_f1"
    ]].rename(columns=lambda c: c.replace("tnbc_test_", ""))

    _plot_metrics_heatmap(oth_mat,  f"OTH Internal Test Metrics (Top {len(top_models)})",
                          outdir / "metrics_heatmap_oth_test.png", outdir / "metrics_heatmap_oth_test.pdf")
    _plot_metrics_heatmap(tnbc_mat, f"TNBC External Test Metrics (Top {len(top_models)})",
                          outdir / "metrics_heatmap_tnbc_test.png", outdir / "metrics_heatmap_tnbc_test.pdf")
    print("[OK] saved metrics heatmaps -> metrics_heatmap_(oth/tnbc)_test.(png/pdf)")

    _plot_radar(oth_mat,  "Radar (Top models) on OTH Internal Test",
                outdir / "radar_top_models_oth_test.png", outdir / "radar_top_models_oth_test.pdf", topn=3)
    _plot_radar(tnbc_mat, "Radar (Top models) on TNBC External Test",
                outdir / "radar_top_models_tnbc_test.png", outdir / "radar_top_models_tnbc_test.pdf", topn=3)
    print("[OK] saved radar charts -> radar_top_models_(oth/tnbc)_test.(png/pdf)")

    # Domain shift radar for best TNBC model
    best = summary_df.iloc[0]["model"]
    oth_row = summary_df.set_index("model").loc[best, ["oth_test_auc","oth_test_acc","oth_test_precision","oth_test_recall","oth_test_f1"]]
    tnbc_row = summary_df.set_index("model").loc[best, ["tnbc_test_auc","tnbc_test_acc","tnbc_test_precision","tnbc_test_recall","tnbc_test_f1"]]
    oth_row.index = [i.replace("oth_test_","") for i in oth_row.index]
    tnbc_row.index = [i.replace("tnbc_test_","") for i in tnbc_row.index]
    _plot_domain_shift_radar(oth_row, tnbc_row, best,
                             outdir / f"radar_domain_shift_{best}.png",
                             outdir / f"radar_domain_shift_{best}.pdf")
    print(f"[OK] saved domain-shift radar -> radar_domain_shift_{best}.(png/pdf)")

    print("\n================= Done =================")
    print("Top models by TNBC_TEST_AUC:")
    print(summary_df[["model", "cv_auc_mean", "cv_auc_std", "oth_test_auc", "tnbc_test_auc"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
