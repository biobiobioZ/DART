# v4grl_v2.py  (based on v4_3zs_stage1fix_topk_mixup_Aplus_lastlayer_L2SP.py)
# Added: Stage1 GRL domain-adversarial training (DANN-style) on OTH (labeled) vs TNBC_AU (unlabeled)
# Click-Run:
#   Stage0: DAE pretrain on TNBC_AU (unlabeled)  [on selected topK genes]
#   Stage1: Supervised on OTH labeled (inject semantics)
#   Stage2: TNBC closure (two-phase):
#       Phase1: freeze encoder + enc.eval() (stop dropout noise), train head
#       Phase2: unfreeze encoder LAST linear only + L2-SP, small lr (A+)
#   Optional: mixup on TNBC labeled train in Stage2 (very useful for tiny target)
#   Optional: cross-fit ensemble AUC across repeats (out-of-sample aggregation)

import os, json, math, random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# =========================
# Utils
# =========================
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _fmt_tag_float(x: float) -> str:
    """Format float for folder tag: 0.3 -> 0p3, 1.0 -> 1, 0.05 -> 0p05"""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    # 3 significant digits is usually enough for tags
    s = f"{xf:.3g}"
    s = s.replace(".", "p").replace("-", "m")
    return s

def build_outdir_tag(CFG: Dict) -> str:
    """Small experiment tag appended to out_dir to prevent accidental overwrites."""
    if not bool(CFG.get("auto_tag_outdir", True)):
        return ""
    if not bool(CFG.get("stage1_use_grl", False)):
        return "_nogrlv2"
    parts = [
        "grlv2",
        str(CFG.get("stage1_grl_schedule", "sigmoid"))[:3],
        "lm" + _fmt_tag_float(CFG.get("stage1_grl_lambda_max", 1.0)),
        "wu" + _fmt_tag_float(CFG.get("stage1_grl_warmup", 0.0)),
        "dh" + str(int(CFG.get("stage1_dom_hidden", 256))),
        "dd" + _fmt_tag_float(CFG.get("stage1_dom_dropout", CFG.get("dropout", 0.1))),
        "dw" + _fmt_tag_float(CFG.get("stage1_dom_loss_weight", 1.0)),
        "em" + _fmt_tag_float(CFG.get("stage1_enc_lr_mult", 1.0)),
    ]
    return "_" + "_".join(parts)

def dump_cfg_json(CFG: Dict, out_root: Path):
    try:
        with open(out_root / "config.json", "w", encoding="utf-8") as f:
            json.dump(CFG, f, indent=2, default=str)
        print(f"[CFG] saved: {out_root/'config.json'}")
    except Exception as e:
        print(f"[CFG] warn: failed to save config.json: {e}")

def safe_torch_save(obj, path: str):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp, _use_new_zipfile_serialization=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass

def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def read_genes_json(genes_json_path: str) -> List[str]:
    with open(genes_json_path, "r", encoding="utf-8") as f:
        genes = json.load(f)
    if isinstance(genes, dict) and "genes" in genes:
        genes = genes["genes"]
    if not isinstance(genes, list):
        raise ValueError("genes.json must be a list of gene names or {'genes':[...]} dict.")
    return [str(g) for g in genes]

def load_csv(path: str, index_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if index_col is not None and index_col in df.columns:
        df = df.set_index(index_col)
    else:
        if df.columns[0].lower().startswith("unnamed"):
            df = df.set_index(df.columns[0])
    return df

def intersect_genes(genes: List[str], dfs: List[pd.DataFrame], label_col: str) -> List[str]:
    sets = []
    for df in dfs:
        cols = set(df.columns)
        if label_col in cols:
            cols.remove(label_col)
        sets.append(cols)
    inter = set(genes).intersection(*sets)
    return sorted(list(inter))

def impute_nan_inf_inplace(X: np.ndarray) -> np.ndarray:
    X[np.isinf(X)] = np.nan
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(X))
    X[inds] = col_mean[inds[1]]
    return X

def best_threshold_by_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    ts = np.unique(np.clip(p, 0, 1))
    if ts.size > 200:
        ts = np.quantile(ts, np.linspace(0.01, 0.99, 200))
    best_t, best_f1v = 0.5, -1.0
    for t in ts:
        pred = (p >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1v:
            best_f1v, best_t = f1v, float(t)
    return best_t

def median_abs_deviation(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return mad

def select_topk_genes_unsup(
    df_tnbc_au: pd.DataFrame,
    genes: List[str],
    k: int,
    metric: str = "mad",              # "mad" or "var"
    keep_genes: Optional[List[str]] = None,
) -> List[str]:
    """
    Leak-safe: use ONLY TNBC_AU unlabeled to rank genes by dispersion.
    """
    if k <= 0 or k >= len(genes):
        out = sorted(list(set(genes)))
        if keep_genes:
            out = sorted(list(set(out).union(set(keep_genes))))
        return out

    X = np.asarray(df_tnbc_au[genes].values, dtype=np.float32)
    X = impute_nan_inf_inplace(X)

    if metric.lower() == "mad":
        s = median_abs_deviation(X, axis=0)
    elif metric.lower() == "var":
        s = np.var(X, axis=0)
    else:
        raise ValueError("metric must be 'mad' or 'var'")

    idx = np.argsort(-s)[:k]
    top = [genes[i] for i in idx]
    if keep_genes:
        top = sorted(list(set(top).union(set(keep_genes))))
    return top


# =========================
# Datasets
# =========================
class LabeledDS(Dataset):
    def __init__(self, df: pd.DataFrame, gene_list: List[str], label_col="Response"):
        X = np.asarray(df[gene_list].values, dtype=np.float32).copy()
        y = np.asarray(df[label_col].values, dtype=np.int64).copy()
        X = impute_nan_inf_inplace(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.ids = df.index.astype(str).tolist()

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.ids[i]

class UnlabeledDS(Dataset):
    def __init__(self, df: pd.DataFrame, gene_list: List[str]):
        X = np.asarray(df[gene_list].values, dtype=np.float32).copy()
        X = impute_nan_inf_inplace(X)
        self.X = torch.from_numpy(X)
        self.ids = df.index.astype(str).tolist()

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        return self.X[i], self.ids[i]


# =========================
# Models
# =========================
class Encoder(nn.Module):
    def __init__(self, in_dim: int, z_dim: int = 256, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        # net.0: Linear(in->hidden)
        # net.1: ReLU
        # net.2: Dropout
        # net.3: Linear(hidden->z)  (we will unfreeze ONLY this layer in Stage2 phase2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, z): return self.net(z)

class DAE(nn.Module):
    def __init__(self, in_dim: int, z_dim: int = 256, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim=z_dim, hidden=hidden, dropout=dropout)
        self.decoder = Decoder(z_dim=z_dim, out_dim=in_dim, hidden=hidden, dropout=dropout)
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

class Classifier(nn.Module):
    def __init__(self, z_dim: int = 256, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, z): return self.net(z).squeeze(-1)



# =========================
# GRL (Gradient Reversal Layer) + Domain Discriminator (for DANN)
# =========================
class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None

def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    Gradient Reversal Layer (GRL).
    Forward: identity
    Backward: multiplies gradient by -lambd
    """
    return _GRLFn.apply(x, lambd)

class DomainDiscriminator(nn.Module):
    def __init__(self, z_dim: int = 256, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

def dann_lambda(ep: int, epochs: int, schedule: str = "sigmoid", warmup: float = 0.0, gamma: float = 10.0, lam_max: float = 1.0) -> float:
    """
    Returns lambda in [0, lam_max] for domain-adversarial training.
    - schedule="linear": ramps from 0 to lam_max after warmup fraction.
    - schedule="sigmoid": DANN-style 2/(1+exp(-gamma*p))-1, after warmup.
    warmup: fraction in [0,1), epochs before ramp begins.
    """
    if epochs <= 1:
        return float(lam_max)
    warm_ep = int(round(float(warmup) * float(epochs)))
    if ep < warm_ep:
        return 0.0
    # progress after warmup
    denom = max(1, epochs - warm_ep)
    p = float(ep - warm_ep) / float(denom)  # in [0,1]
    p = max(0.0, min(1.0, p))
    schedule = str(schedule).lower()
    if schedule == "linear":
        lam = p
    else:
        # sigmoid (default)
        lam = 2.0 / (1.0 + math.exp(-float(gamma) * p)) - 1.0
    return float(lam_max) * float(lam)

# =========================
# Eval
# =========================
@torch.no_grad()
def eval_auc(model_enc: nn.Module, model_clf: nn.Module, loader: DataLoader, device: str):
    model_enc.eval(); model_clf.eval()
    ys, ps, ids = [], [], []
    for xb, yb, idb in loader:
        xb = xb.to(device)
        z = model_enc(xb)
        logit = model_clf(z)
        p = torch.sigmoid(logit).detach().cpu().numpy()
        ys.append(yb.numpy())
        ps.append(p)
        ids.extend(list(idb))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    auc = float("nan") if len(np.unique(y)) < 2 else roc_auc_score(y, p)
    return float(auc), y, p, ids



# =========================
# SHAP (feature attribution) utilities
# =========================
class JointModel(nn.Module):
    """Wrap Encoder+Classifier into a single module for SHAP explainers."""
    def __init__(self, enc: nn.Module, clf: nn.Module):
        super().__init__()
        self.enc = enc
        self.clf = clf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        logit = self.clf(z)
        # SHAP's torch Gradient/Deep explainers expect a 2D output: [B, n_outputs].
        # Some binary heads return [B]; normalize to [B, 1] to avoid indexing errors.
        if isinstance(logit, (tuple, list)):
            logit = logit[0]
        if logit.dim() == 1:
            logit = logit.unsqueeze(1)
        # keep raw logit for SHAP (stable, monotonic w.r.t. prob)
        return logit


def _sample_df(df: pd.DataFrame, n: int, seed: int = 0, stratify_col: Optional[str] = None) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df
    if stratify_col is None or stratify_col not in df.columns:
        return df.sample(n=n, random_state=seed, replace=False)
    # stratified sample
    rs = np.random.RandomState(seed)
    out_idx = []
    y = df[stratify_col].astype(int).values
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        k = int(round(n * (len(idx) / len(y))))
        k = max(1, min(len(idx), k))
        pick = rs.choice(idx, size=k, replace=False)
        out_idx.extend(pick.tolist())
    out = df.iloc[out_idx].copy()
    if len(out) > n:
        out = out.sample(n=n, random_state=seed, replace=False)
    return out


def _df_to_X_y_ids(df: pd.DataFrame, genes: List[str], label_col: Optional[str] = None):
    X = np.asarray(df[genes].values, dtype=np.float32).copy()
    X = impute_nan_inf_inplace(X)
    y = None
    if label_col is not None and label_col in df.columns:
        y = np.asarray(df[label_col].values, dtype=np.int64).copy()
    ids = df.index.astype(str).tolist()
    return X, y, ids


def compute_and_save_shap(
    *,
    enc: nn.Module,
    clf: nn.Module,
    genes: List[str],
    df_background: pd.DataFrame,
    df_explain: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    prefix: str,
    seed: int,
    device: str,
    background_n: int = 128,
    explain_n: Optional[int] = None,
    bar_topn: int = 50,
    beeswarm_topn: int = 50,
    fig_fmt: str = "pdf",
    method: str = "gradient",  # "gradient" is robust for generic MLPs
):
    """Compute SHAP on (Encoder+Classifier) and save tables + plots."""
    ensure_dir(out_dir)
    try:
        import shap
    except Exception as e:
        print(f"[SHAP] shap import failed: {e}")
        return None

    df_bg = _sample_df(df_background, background_n, seed=seed, stratify_col=label_col if label_col in df_background.columns else None)
    df_ev = _sample_df(df_explain, explain_n, seed=seed + 17, stratify_col=label_col if label_col in df_explain.columns else None)

    X_bg, _, _ = _df_to_X_y_ids(df_bg, genes, label_col=None)
    X_ev, y_ev, ids_ev = _df_to_X_y_ids(df_ev, genes, label_col=label_col)

    xt_bg = torch.from_numpy(X_bg).to(device)
    xt_ev = torch.from_numpy(X_ev).to(device)

    enc = enc.to(device); clf = clf.to(device)
    enc.eval(); clf.eval()
    model = JointModel(enc, clf).to(device)
    model.eval()

    if str(method).lower() == "deep":
        explainer = shap.DeepExplainer(model, xt_bg)
    else:
        explainer = shap.GradientExplainer(model, xt_bg)

    shap_vals = explainer.shap_values(xt_ev)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals, dtype=np.float32)
    # Depending on SHAP version and output shape, we may get [N, F, 1].
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals[:, :, 0]

    np.savez_compressed(
        out_dir / f"{prefix}_shap_values.npz",
        shap=shap_vals,
        X=X_ev,
        y=(np.asarray(y_ev) if y_ev is not None else np.asarray([])),
        ids=np.asarray(ids_ev, dtype=object),
        genes=np.asarray(genes, dtype=object),
    )

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_val = np.mean(shap_vals, axis=0)

    imp = pd.DataFrame(
        {
            "gene": genes,
            "mean_abs_shap": mean_abs.astype(float),
            "mean_shap": mean_val.astype(float),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)

    csv_path = out_dir / f"{prefix}_shap_meanabs.csv"
    imp.to_csv(csv_path, index=False)

    try:
        top = imp.head(int(bar_topn)).iloc[::-1]
        plt.figure(figsize=(9.0, max(4.0, 0.22 * len(top) + 1.0)))
        plt.barh(top["gene"].tolist(), top["mean_abs_shap"].tolist())
        plt.xlabel("mean(|SHAP|) on logit")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_bar_top{int(bar_topn)}.{fig_fmt}", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[SHAP] bar plot failed: {e}")

    try:
        shap.summary_plot(shap_vals, X_ev, feature_names=genes, show=False, max_display=int(beeswarm_topn))
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_beeswarm_top{int(beeswarm_topn)}.{fig_fmt}", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[SHAP] beeswarm plot failed: {e}")

    meta = {
        "prefix": prefix,
        "device": device,
        "method": method,
        "background_n": int(len(df_bg)),
        "explain_n": int(len(df_ev)),
        "bar_topn": int(bar_topn),
        "beeswarm_topn": int(beeswarm_topn),
        "csv": str(csv_path),
    }
    with open(out_dir / f"{prefix}_shap_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SHAP] saved: {csv_path}")
    return imp


# =========================
# Stage0: DAE pretrain
# =========================
def train_stage0_ae(
    df_tnbc_au: pd.DataFrame,
    genes: List[str],
    out_dir: Path,
    seed: int,
    z_dim: int,
    ae_hidden: int,
    ae_dropout: float,
    ae_lr: float,
    ae_epochs: int,
    ae_batch: int,
    noise_std: float,
    weight_decay: float,
    device: str,
) -> str:
    seed_all(seed)
    ds = UnlabeledDS(df_tnbc_au, genes)
    dl = DataLoader(ds, batch_size=ae_batch, shuffle=True, drop_last=False, num_workers=0)

    ae = DAE(in_dim=len(genes), z_dim=z_dim, hidden=ae_hidden, dropout=ae_dropout).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=ae_lr, weight_decay=weight_decay)

    best_mse, best_state = float("inf"), None

    for ep in range(ae_epochs):
        ae.train()
        mses = []
        for xb, _ in dl:
            xb = xb.to(device)
            xn = xb + noise_std * torch.randn_like(xb) if noise_std > 0 else xb
            xhat, _ = ae(xn)
            loss = F.mse_loss(xhat, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            mses.append(loss.item())

        recon = float(np.mean(mses))
        if ep % 5 == 0 or ep == ae_epochs - 1:
            print(f"[Stage0-ONCE] ep={ep} reconMSE={recon:.4f}")

        if recon < best_mse:
            best_mse = recon
            best_state = {k: v.detach().cpu().clone() for k, v in ae.encoder.state_dict().items()}

    ckpt = out_dir / "_stage0_encoder_pretrained_TNBC_AU.pt"
    safe_torch_save(best_state, str(ckpt))
    return str(ckpt)


# =========================
# Stage1: OTH supervised
# =========================
def train_stage1_oth_supervised(
    df_oth: pd.DataFrame,
    genes: List[str],
    label_col: str,
    out_dir: Path,
    seed: int,
    stage0_encoder_ckpt: str,
    z_dim: int,
    enc_hidden: int,
    clf_hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch: int,
    wd: float,
    enc_lr_mult: float,
    device: str,
) -> Tuple[str, str]:
    seed_all(seed)

    y = df_oth[label_col].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))
    df_tr = df_oth.iloc[tr_idx].copy()
    df_va = df_oth.iloc[va_idx].copy()

    ds_tr = LabeledDS(df_tr, genes, label_col)
    ds_va = LabeledDS(df_va, genes, label_col)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=False, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    enc = Encoder(in_dim=len(genes), z_dim=z_dim, hidden=enc_hidden, dropout=dropout).to(device)
    clf = Classifier(z_dim=z_dim, hidden=clf_hidden, dropout=dropout).to(device)

    enc.load_state_dict(safe_torch_load(stage0_encoder_ckpt, map_location="cpu"), strict=True)

    opt = torch.optim.AdamW(
        [{"params": enc.parameters(), "lr": lr * enc_lr_mult},
         {"params": clf.parameters(), "lr": lr}],
        weight_decay=wd
    )

    best_auc, best_enc_state, best_clf_state = -1.0, None, None

    for ep in range(epochs):
        enc.train(); clf.train()
        losses = []
        for xb, yb, _ in dl_tr:
            xb = xb.to(device)
            yb = yb.float().to(device)
            z = enc(xb)
            logit = clf(z)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        auc, _, _, _ = eval_auc(enc, clf, dl_va, device)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[Stage1] ep={ep:03d} loss={np.mean(losses):.4f} OTH-val AUC={auc:.4f}")

        if (not np.isnan(auc)) and auc > best_auc:
            best_auc = float(auc)
            best_enc_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            best_clf_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    enc_ckpt = out_dir / "_stage1_encoder_OTHsup.pt"
    clf_ckpt = out_dir / "_stage1_clf_OTHsup.pt"
    safe_torch_save(best_enc_state, str(enc_ckpt))
    safe_torch_save(best_clf_state, str(clf_ckpt))
    return str(enc_ckpt), str(clf_ckpt)


def train_stage1_oth_supervised_grl_v2(
    df_oth: pd.DataFrame,
    df_tnbc_au: pd.DataFrame,
    genes: List[str],
    label_col: str,
    out_dir: Path,
    seed: int,
    stage0_encoder_ckpt: str,
    z_dim: int,
    enc_hidden: int,
    clf_hidden: int,
    dom_hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch: int,
    wd: float,
    enc_lr_mult: float,
    dom_lr_mult: float,
    dom_dropout: float,
    dom_schedule: str,
    dom_warmup: float,
    dom_gamma: float,
    dom_lambda_max: float,
    dom_loss_weight: float,
    device: str,
) -> Tuple[str, str, str]:
    """
    Stage1 (GRL-DANN): supervised on OTH labeled + domain-adversarial alignment
    between OTH (source) and TNBC_AU (target, unlabeled).

    Loss:
      L = L_task(OTH) + w * λ(ep) * L_dom(OTH vs TNBC_AU)
    with GRL applied on features fed to the domain discriminator, and λ(ep) scaling the domain loss (DANN-style ramp; avoids λ^2).
    """
    seed_all(seed)

    # OTH split (train/val) for stable tie-break validation
    y = df_oth[label_col].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))
    df_tr = df_oth.iloc[tr_idx].copy()
    df_va = df_oth.iloc[va_idx].copy()

    ds_tr = LabeledDS(df_tr, genes, label_col)
    ds_va = LabeledDS(df_va, genes, label_col)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

    # TNBC_AU (unlabeled) loader for domain alignment
    ds_tu = UnlabeledDS(df_tnbc_au, genes)
    dl_tu = DataLoader(ds_tu, batch_size=batch, shuffle=True, drop_last=True, num_workers=0)

    enc = Encoder(in_dim=len(genes), z_dim=z_dim, hidden=enc_hidden, dropout=dropout).to(device)
    clf = Classifier(z_dim=z_dim, hidden=clf_hidden, dropout=dropout).to(device)
    dom = DomainDiscriminator(z_dim=z_dim, hidden=dom_hidden, dropout=dom_dropout).to(device)

    enc.load_state_dict(safe_torch_load(stage0_encoder_ckpt, map_location="cpu"), strict=True)

    opt = torch.optim.AdamW(
        [{"params": enc.parameters(), "lr": lr * enc_lr_mult},
         {"params": clf.parameters(), "lr": lr},
         {"params": dom.parameters(), "lr": lr * dom_lr_mult}],
        weight_decay=wd
    )

    best_auc, best_enc_state, best_clf_state, best_dom_state = -1.0, None, None, None

    # target iterator (cycled)
    it_tu = iter(dl_tu)

    for ep in range(epochs):
        enc.train(); clf.train(); dom.train()
        losses_task, losses_dom, acc_dom = [], [], []

        lambd = dann_lambda(
            ep=ep, epochs=epochs,
            schedule=dom_schedule,
            warmup=dom_warmup,
            gamma=dom_gamma,
            lam_max=dom_lambda_max,
        )

        for xb_s, yb_s, _ in dl_tr:
            try:
                xb_t, _ = next(it_tu)
            except StopIteration:
                it_tu = iter(dl_tu)
                xb_t, _ = next(it_tu)

            xb_s = xb_s.to(device)
            yb_s = yb_s.float().to(device)
            xb_t = xb_t.to(device)

            # task (source labeled)
            z_s = enc(xb_s)
            logit_y = clf(z_s)
            loss_task = F.binary_cross_entropy_with_logits(logit_y, yb_s)

            # domain (source vs target)
            z_t = enc(xb_t)
            dz_s = dom(grl(z_s, 1.0))  # λ applied via loss scaling (avoid λ^2)
            dz_t = dom(grl(z_t, 1.0))  # λ applied via loss scaling (avoid λ^2)
            dlog = torch.cat([dz_s, dz_t], dim=0)
            dlab = torch.cat([torch.zeros_like(dz_s), torch.ones_like(dz_t)], dim=0)
            loss_dom = F.binary_cross_entropy_with_logits(dlog, dlab)

            loss = loss_task + float(dom_loss_weight) * float(lambd) * loss_dom

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses_task.append(loss_task.item())
            losses_dom.append(loss_dom.item())

            with torch.no_grad():
                pred = (torch.sigmoid(dlog) > 0.5).float()
                acc = (pred == dlab).float().mean().item()
                acc_dom.append(acc)

        auc, _, _, _ = eval_auc(enc, clf, dl_va, device)
        if ep % 5 == 0 or ep == epochs - 1:
            print(
                f"[Stage1-GRLv2] ep={ep:03d} "
                f"λ={lambd:.3f} "
                f"task={np.mean(losses_task):.4f} "
                f"dom={np.mean(losses_dom):.4f} "
                f"domAcc={np.mean(acc_dom):.3f} "
                f"OTH-val AUC={auc:.4f}"
            )

        if (not np.isnan(auc)) and auc > best_auc:
            best_auc = float(auc)
            best_enc_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            best_clf_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
            best_dom_state = {k: v.detach().cpu().clone() for k, v in dom.state_dict().items()}

    enc_ckpt = out_dir / "_stage1_encoder_OTHsup_GRL.pt"
    clf_ckpt = out_dir / "_stage1_clf_OTHsup_GRL.pt"
    dom_ckpt = out_dir / "_stage1_domdisc_OTHvsTNBC_AU.pt"
    safe_torch_save(best_enc_state, str(enc_ckpt))
    safe_torch_save(best_clf_state, str(clf_ckpt))
    safe_torch_save(best_dom_state, str(dom_ckpt))
    return str(enc_ckpt), str(clf_ckpt), str(dom_ckpt)


# =========================
# Stage2 helpers
# =========================
def freeze_all_encoder(enc: nn.Module):
    for p in enc.parameters():
        p.requires_grad = False

def unfreeze_last_layer(enc: nn.Module):
    """
    Only unfreeze enc.net.3.* (last Linear)
    """
    freeze_all_encoder(enc)
    trainable = []
    for name, p in enc.named_parameters():
        if name.startswith("net.3."):
            p.requires_grad = True
            trainable.append(p)
    return trainable

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y2 = y[idx]
    xm = lam * x + (1 - lam) * x2
    ym = lam * y + (1 - lam) * y2
    return xm, ym


# =========================
# Stage2: TNBC closure (A+ lastlayer + L2SP) + optional mixup
# =========================
def train_stage2_tnbc_Aplus(
    df_tnbc_train: pd.DataFrame,
    df_tnbc_val: pd.DataFrame,
    df_oth_val: pd.DataFrame,
    genes: List[str],
    label_col: str,
    out_dir: Path,
    seed: int,
    init_enc_ckpt: str,
    init_clf_ckpt: str,
    z_dim: int,
    enc_hidden: int,
    clf_hidden: int,
    dropout: float,

    stage2_lr_head: float,
    stage2_lr_enc_mult: float,   # lastlayer lr = head_lr * mult
    wd: float,
    lambda_l2sp: float,
    phase1_ratio: float,
    epochs: int,
    batch_lab: int,
    tie_break_weight: float,

    use_mixup: bool,
    mixup_alpha: float,

    device: str,
) -> Tuple[nn.Module, nn.Module, Dict]:
    seed_all(seed)

    ds_lab_tr = LabeledDS(df_tnbc_train, genes, label_col)
    ds_lab_va = LabeledDS(df_tnbc_val, genes, label_col)
    ds_oth_va = LabeledDS(df_oth_val, genes, label_col)

    dl_lab_tr = DataLoader(ds_lab_tr, batch_size=batch_lab, shuffle=True, drop_last=True, num_workers=0)
    dl_lab_va = DataLoader(ds_lab_va, batch_size=batch_lab, shuffle=False, drop_last=False, num_workers=0)
    dl_oth_va = DataLoader(ds_oth_va, batch_size=batch_lab, shuffle=False, drop_last=False, num_workers=0)

    enc = Encoder(in_dim=len(genes), z_dim=z_dim, hidden=enc_hidden, dropout=dropout).to(device)
    clf = Classifier(z_dim=z_dim, hidden=clf_hidden, dropout=dropout).to(device)
    enc.load_state_dict(safe_torch_load(init_enc_ckpt, map_location="cpu"), strict=True)
    clf.load_state_dict(safe_torch_load(init_clf_ckpt, map_location="cpu"), strict=True)

    phase1_epochs = int(round(epochs * float(phase1_ratio)))
    phase1_epochs = max(1, min(epochs - 1, phase1_epochs))
    print(f"[Stage2] Phase1(head-only)={phase1_epochs} epochs, Phase2(unfreeze lastlayer + L2SP)={epochs-phase1_epochs} epochs")

    # Phase1: freeze encoder, BUT IMPORTANT: enc.eval() to disable Dropout noise
    freeze_all_encoder(enc)
    enc.eval()
    clf.train()

    opt = torch.optim.AdamW([{"params": clf.parameters(), "lr": stage2_lr_head}], weight_decay=wd)

    ref_last = None  # L2SP reference for last layer weights
    best_score, best_state, best_log = -1e18, None, {}

    for ep in range(epochs):
        # switch to phase2
        if ep == phase1_epochs:
            train_last = unfreeze_last_layer(enc)
            enc.train()    # now allow Dropout because last layer adapts; still mild because only last layer trainable
            clf.train()

            # L2SP reference snapshot for trainable params only (last layer)
            ref_last = {}
            for name, p in enc.named_parameters():
                if p.requires_grad:
                    ref_last[name] = p.detach().clone()

            opt = torch.optim.AdamW(
                [
                    {"params": clf.parameters(), "lr": stage2_lr_head},
                    {"params": train_last, "lr": stage2_lr_head * stage2_lr_enc_mult},
                ],
                weight_decay=wd
            )
            print(f"[Stage2] ---- switch to phase2 at ep={ep}: unfreeze enc.net.3; L2SP lambda={lambda_l2sp} lr_enc_mult={stage2_lr_enc_mult} ----")

        losses = []
        l2sps = []
        for xb, yb, _ in dl_lab_tr:
            xb = xb.to(device)
            yb = yb.float().to(device)

            # mixup on labeled TNBC train
            if use_mixup:
                xb2, yb2 = mixup_batch(xb, yb, mixup_alpha)
            else:
                xb2, yb2 = xb, yb

            # Phase1 encoder frozen: safe to run no_grad to avoid useless graph
            if ep < phase1_epochs:
                with torch.no_grad():
                    z = enc(xb2)
                logit = clf(z)
                loss_sup = F.binary_cross_entropy_with_logits(logit, yb2)
                loss_l2sp = torch.tensor(0.0, device=device)
            else:
                z = enc(xb2)
                logit = clf(z)
                loss_sup = F.binary_cross_entropy_with_logits(logit, yb2)

                # L2-SP on last layer
                loss_l2sp = torch.tensor(0.0, device=device)
                for name, p in enc.named_parameters():
                    if p.requires_grad:
                        loss_l2sp = loss_l2sp + F.mse_loss(p, ref_last[name])

            loss = loss_sup + (lambda_l2sp * loss_l2sp if ep >= phase1_epochs else 0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss_sup.item()))
            l2sps.append(float(loss_l2sp.item()))

        auc_tnbc, yv, pv, _ = eval_auc(enc, clf, dl_lab_va, device)
        auc_oth, _, _, _ = eval_auc(enc, clf, dl_oth_va, device)
        score = (0.0 if np.isnan(auc_tnbc) else auc_tnbc) + tie_break_weight * (0.0 if np.isnan(auc_oth) else auc_oth)

        print(
            f"[Stage2] ep={ep:03d} TNBC-val AUC={auc_tnbc:.4f} | OTH-val AUC={auc_oth:.4f} "
            f"| sup={np.mean(losses):.4f} l2sp={np.mean(l2sps):.6f} | best_score={best_score:.4f}"
        )

        if score > best_score:
            best_score = float(score)
            best_state = {
                "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                "clf": {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()},
            }
            thr = best_threshold_by_f1(yv, pv)
            pred = (pv >= thr).astype(int)
            best_log = {
                "best_ep": int(ep),
                "tnbc_val_auc": float(auc_tnbc),
                "oth_val_auc": float(auc_oth),
                "tnbc_val_thr": float(thr),
                "tnbc_val_f1": float(f1_score(yv, pred, zero_division=0)),
                "tnbc_val_acc": float(accuracy_score(yv, pred)),
                "lambda_l2sp": float(lambda_l2sp),
                "phase1_ratio": float(phase1_ratio),
                "use_mixup": bool(use_mixup),
                "mixup_alpha": float(mixup_alpha),
                "stage2_lr_head": float(stage2_lr_head),
                "stage2_lr_enc_mult": float(stage2_lr_enc_mult),
            }

    enc.load_state_dict(best_state["enc"], strict=True)
    clf.load_state_dict(best_state["clf"], strict=True)

    enc_ckpt = out_dir / "_stage2_best_encoder.pt"
    clf_ckpt = out_dir / "_stage2_best_clf.pt"
    safe_torch_save(best_state["enc"], str(enc_ckpt))
    safe_torch_save(best_state["clf"], str(clf_ckpt))

    best_log["stage2_encoder_ckpt"] = str(enc_ckpt)
    best_log["stage2_clf_ckpt"] = str(clf_ckpt)
    best_log["best_score"] = float(best_score)
    return enc, clf, best_log


def split_tnbc_labeled_repeated_holdout(
    df_lab: pd.DataFrame,
    label_col: str,
    seed: int,
    test_size: float,
    val_size: float,
):
    y = df_lab[label_col].astype(int).values
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trva, idx_te = next(sss1.split(np.zeros(len(y)), y))
    df_trva = df_lab.iloc[idx_trva].copy()
    df_te = df_lab.iloc[idx_te].copy()

    y_trva = df_trva[label_col].astype(int).values
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 999)
    idx_tr, idx_va = next(sss2.split(np.zeros(len(y_trva)), y_trva))
    df_tr = df_trva.iloc[idx_tr].copy()
    df_va = df_trva.iloc[idx_va].copy()
    return df_tr, df_va, df_te


# =========================
# Main
# =========================
def main(CFG: Dict):
    # Build an experiment-specific output folder to avoid accidental overwrites.
    out_root = Path(CFG["out_dir"])
    tag = build_outdir_tag(CFG)
    if tag:
        out_root = out_root.with_name(out_root.name + tag)
    ensure_dir(out_root)

    device = pick_device()
    print(f"[OUT] {out_root}")
    print(f"[DEVICE] {device}")

    # Echo critical knobs for reproducibility / sanity-checking
    print("[CFG] stage1_use_grl=", bool(CFG.get("stage1_use_grl", False)))
    if bool(CFG.get("stage1_use_grl", False)):
        print(
            "[CFG] GRL schedule={sch} warmup={wu} lambda_max={lm} dom_hidden={dh} dom_dropout={dd} dom_w={dw} enc_lr_mult={em}".format(
                sch=CFG.get("stage1_grl_schedule", "sigmoid"),
                wu=CFG.get("stage1_grl_warmup", 0.0),
                lm=CFG.get("stage1_grl_lambda_max", 1.0),
                dh=CFG.get("stage1_dom_hidden", 256),
                dd=CFG.get("stage1_dom_dropout", CFG.get("dropout", 0.1)),
                dw=CFG.get("stage1_dom_loss_weight", 1.0),
                em=CFG.get("stage1_enc_lr_mult", 1.0),
            )
        )

    dump_cfg_json(CFG, out_root)

    genes0 = read_genes_json(CFG["genes_json"])
    df_oth = load_csv(CFG["oth_labeled_csv"], index_col=CFG.get("index_col"))
    df_tnbc_au = load_csv(CFG["tnbc_array_unlabeled_csv"], index_col=CFG.get("index_col"))
    df_tnbc_al = load_csv(CFG["tnbc_array_labeled_csv"], index_col=CFG.get("index_col"))
    label_col = CFG["label_col"]

    genes_all = intersect_genes(genes0, [df_oth, df_tnbc_au, df_tnbc_al], label_col)
    print(f"[GENES] intersection = {len(genes_all)}")

    keep_genes = None
    if CFG.get("topk_keep_genes_json"):
        keep_genes = read_genes_json(CFG["topk_keep_genes_json"])
        keep_genes = [g for g in keep_genes if g in set(genes_all)]
        print(f"[TOPK] keep_genes from json (in intersection): {len(keep_genes)}")

    # Leak-safe gene selection on TNBC_AU only
    if CFG["topk_enable"]:
        genes = select_topk_genes_unsup(
            df_tnbc_au=df_tnbc_au,
            genes=genes_all,
            k=int(CFG["topk_k"]),
            metric=str(CFG["topk_metric"]),
            keep_genes=keep_genes,
        )
        print(f"[TOPK] metric={CFG['topk_metric']} k={CFG['topk_k']} -> selected genes={len(genes)}")
    else:
        genes = genes_all
        print(f"[TOPK] disabled -> genes={len(genes)}")

    print(f"[DATA] OTH={len(df_oth)} | TNBC_AU={len(df_tnbc_au)} | TNBC_AL={len(df_tnbc_al)}")

    # Stage0 once
    stage0_ckpt = out_root / "_stage0_encoder_pretrained_TNBC_AU.pt"
    if stage0_ckpt.exists() and CFG.get("reuse_stage0", False):
        print(f"[Stage0-ONCE] Reuse existing: {stage0_ckpt}")
        stage0_path = str(stage0_ckpt)
    else:
        print("[Stage0-ONCE] Pretrain DAE on TNBC_AU only.")
        stage0_path = train_stage0_ae(
            df_tnbc_au=df_tnbc_au,
            genes=genes,
            out_dir=out_root,
            seed=CFG["seed"],
            z_dim=CFG["z_dim"],
            ae_hidden=CFG["ae_hidden"],
            ae_dropout=CFG["dropout"],
            ae_lr=CFG["ae_lr"],
            ae_epochs=CFG["ae_epochs"],
            ae_batch=CFG["ae_batch"],
            noise_std=CFG["ae_noise_std"],
            weight_decay=CFG["wd"],
            device=device,
        )
        print(f"[Stage0-ONCE] Saved: {stage0_path}")

    # Stage1 once
    use_grl = bool(CFG.get("stage1_use_grl", True))
    if use_grl:
        stage1_enc_ckpt = out_root / "_stage1_encoder_OTHsup_GRL.pt"
        stage1_clf_ckpt = out_root / "_stage1_clf_OTHsup_GRL.pt"
        stage1_dom_ckpt = out_root / "_stage1_domdisc_OTHvsTNBC_AU.pt"
        reuse_ok = stage1_enc_ckpt.exists() and stage1_clf_ckpt.exists() and stage1_dom_ckpt.exists()
    else:
        stage1_enc_ckpt = out_root / "_stage1_encoder_OTHsup.pt"
        stage1_clf_ckpt = out_root / "_stage1_clf_OTHsup.pt"
        stage1_dom_ckpt = None
        reuse_ok = stage1_enc_ckpt.exists() and stage1_clf_ckpt.exists()

    if reuse_ok and CFG.get("reuse_stage1", False):
        print(f"[Stage1-ONCE] Reuse existing stage1 ckpts. use_grl={use_grl}")
        stage1_enc_path = str(stage1_enc_ckpt)
        stage1_clf_path = str(stage1_clf_ckpt)
    else:
        if use_grl:
            print("\n[Stage1] OTH supervised + GRL domain-adversarial alignment (OTH vs TNBC_AU).")
            stage1_enc_path, stage1_clf_path, _ = train_stage1_oth_supervised_grl_v2(
                df_oth=df_oth,
                df_tnbc_au=df_tnbc_au,
                genes=genes,
                label_col=label_col,
                out_dir=out_root,
                seed=CFG["seed"],
                stage0_encoder_ckpt=stage0_path,
                z_dim=CFG["z_dim"],
                enc_hidden=CFG["enc_hidden"],
                clf_hidden=CFG["clf_hidden"],
                dom_hidden=CFG.get("stage1_dom_hidden", 256),
                dropout=CFG["dropout"],
                lr=CFG["stage1_lr"],
                epochs=CFG["stage1_epochs"],
                batch=CFG["stage1_batch"],
                wd=CFG["wd"],
                enc_lr_mult=CFG["stage1_enc_lr_mult"],
                dom_lr_mult=CFG.get("stage1_dom_lr_mult", 1.0),
                dom_dropout=CFG.get("stage1_dom_dropout", CFG["dropout"]),
                dom_schedule=CFG.get("stage1_grl_schedule", "sigmoid"),
                dom_warmup=CFG.get("stage1_grl_warmup", 0.30),
                dom_gamma=CFG.get("stage1_grl_gamma", 10.0),
                dom_lambda_max=CFG.get("stage1_grl_lambda_max", 0.5),
                dom_loss_weight=CFG.get("stage1_dom_loss_weight", 1.0),
                device=device,
            )
        else:
            print("\n[Stage1] Supervised on OTH labeled to inject task semantics.")
            stage1_enc_path, stage1_clf_path = train_stage1_oth_supervised(
                df_oth=df_oth,
                genes=genes,
                label_col=label_col,
                out_dir=out_root,
                seed=CFG["seed"],
                stage0_encoder_ckpt=stage0_path,
                z_dim=CFG["z_dim"],
                enc_hidden=CFG["enc_hidden"],
                clf_hidden=CFG["clf_hidden"],
                dropout=CFG["dropout"],
                lr=CFG["stage1_lr"],
                epochs=CFG["stage1_epochs"],
                batch=CFG["stage1_batch"],
                wd=CFG["wd"],
                enc_lr_mult=CFG["stage1_enc_lr_mult"],
                device=device,
            )
    # stable OTH val for tie-break
    y_oth = df_oth[label_col].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG["seed"])
    _, oth_val_idx = next(sss.split(np.zeros(len(y_oth)), y_oth))
    df_oth_val = df_oth.iloc[oth_val_idx].copy()


    # =========================
    # SHAP evidence-1: Source-only Stage1 model (OTH-supervised, no GRL)
    # =========================
    stage1_sourceonly_enc_path = stage1_enc_path
    stage1_sourceonly_clf_path = stage1_clf_path
    if bool(CFG.get("shap_enable", False)):
        # If Stage1 uses GRL for training, optionally train an extra source-only Stage1 for SHAP attribution.
        if use_grl and bool(CFG.get("shap_source_force_source_only", True)):
            so_enc = out_root / "_stage1_encoder_OTHsup.pt"
            so_clf = out_root / "_stage1_clf_OTHsup.pt"
            so_reuse_ok = so_enc.exists() and so_clf.exists()
            if so_reuse_ok and bool(CFG.get("reuse_stage1_sourceonly_for_shap", True)):
                print("[SHAP] Reuse source-only Stage1 ckpts for SHAP.")
                stage1_sourceonly_enc_path = str(so_enc)
                stage1_sourceonly_clf_path = str(so_clf)
            else:
                print("[SHAP] Train extra source-only Stage1 model for SHAP (OTH supervised, no GRL).")
                stage1_sourceonly_enc_path, stage1_sourceonly_clf_path = train_stage1_oth_supervised(
                    df_oth=df_oth,
                    genes=genes,
                    label_col=label_col,
                    out_dir=out_root,
                    seed=CFG["seed"] + 2026,
                    stage0_encoder_ckpt=stage0_path,
                    z_dim=CFG["z_dim"],
                    enc_hidden=CFG["enc_hidden"],
                    clf_hidden=CFG["clf_hidden"],
                    dropout=CFG["dropout"],
                    lr=CFG["stage1_lr"],
                    epochs=CFG["stage1_epochs"],
                    batch=CFG["stage1_batch"],
                    wd=CFG["wd"],
                    enc_lr_mult=CFG["stage1_enc_lr_mult"],
                    device=device,
                )

        print("[SHAP] Evidence-1: Stage1(source-only) SHAP on OTH-val.")
        shap_dir1 = out_root / "shap_stage1_sourceonly"
        shap_dev = str(CFG.get("shap_device", device))
        if shap_dev.startswith("cuda") and (not torch.cuda.is_available()):
            shap_dev = "cpu"

        enc1 = Encoder(in_dim=len(genes), z_dim=CFG["z_dim"], hidden=CFG["enc_hidden"], dropout=CFG["dropout"])
        clf1 = Classifier(z_dim=CFG["z_dim"], hidden=CFG["clf_hidden"], dropout=CFG["dropout"])
        enc1.load_state_dict(safe_torch_load(stage1_sourceonly_enc_path, map_location="cpu"), strict=True)
        clf1.load_state_dict(safe_torch_load(stage1_sourceonly_clf_path, map_location="cpu"), strict=True)

        shap_stage1_imp = compute_and_save_shap(
            enc=enc1, clf=clf1,
            genes=genes,
            df_background=df_oth,
            df_explain=df_oth_val,
            label_col=label_col,
            out_dir=shap_dir1,
            prefix="stage1_sourceonly_OTHval",
            seed=CFG["seed"],
            device=shap_dev,
            background_n=int(CFG.get("shap_background_n", 128)),
            explain_n=CFG.get("shap_explain_n_stage1", None),
            bar_topn=int(CFG.get("shap_bar_topn", 50)),
            beeswarm_topn=int(CFG.get("shap_beeswarm_topn", 50)),
            fig_fmt=str(CFG.get("shap_fig_fmt", "pdf")),
            method=str(CFG.get("shap_method", "gradient")),
        )
    else:
        shap_stage1_imp = None

    # For Stage2 aggregation (Evidence-2)
    stage2_shap_meanabs_list: List[np.ndarray] = []
    stage2_shap_mean_list: List[np.ndarray] = []
    stage2_topk_freq: DefaultDict[str, int] = defaultdict(int)

    # Cross-fit aggregation store: id -> list of probs (out-of-sample only)
    oos_probs: DefaultDict[str, List[float]] = defaultdict(list)

    # repeated holdout
    results = []
    for r in range(CFG["repeats"]):
        seed = CFG["seed"] + r * 100
        run_dir = out_root / f"cv_rep{r:02d}_fold00"
        ensure_dir(run_dir)

        df_tr, df_va, df_te = split_tnbc_labeled_repeated_holdout(
            df_tnbc_al, label_col=label_col, seed=seed,
            test_size=CFG["outer_test_size"], val_size=CFG["inner_val_size"]
        )

        print("\n" + "=" * 80)
        print(f"[RUN] cv_rep{r:02d}_fold00  TNBC train={len(df_tr)} val={len(df_va)} outerTest={len(df_te)} seed={seed}")
        print("=" * 80)

        enc2, clf2, log = train_stage2_tnbc_Aplus(
            df_tnbc_train=df_tr,
            df_tnbc_val=df_va,
            df_oth_val=df_oth_val,
            genes=genes,
            label_col=label_col,
            out_dir=run_dir,
            seed=seed,
            init_enc_ckpt=stage1_enc_path,
            init_clf_ckpt=stage1_clf_path,
            z_dim=CFG["z_dim"],
            enc_hidden=CFG["enc_hidden"],
            clf_hidden=CFG["clf_hidden"],
            dropout=CFG["dropout"],
            stage2_lr_head=CFG["stage2_lr_head"],
            stage2_lr_enc_mult=CFG["stage2_lr_enc_mult"],
            wd=CFG["wd"],
            lambda_l2sp=CFG["lambda_l2sp"],
            phase1_ratio=CFG["phase1_ratio"],
            epochs=CFG["stage2_epochs"],
            batch_lab=CFG["stage2_batch_lab"],
            tie_break_weight=CFG["tie_break_weight"],
            use_mixup=CFG["use_mixup"],
            mixup_alpha=CFG["mixup_alpha"],
            device=device,
        )

        # outer test eval
        ds_te = LabeledDS(df_te, genes, label_col)
        dl_te = DataLoader(ds_te, batch_size=CFG["stage2_batch_lab"], shuffle=False, drop_last=False, num_workers=0)
        auc_te, yte, pte, ids_te = eval_auc(enc2, clf2, dl_te, device)

        # store OOS probs for cross-fit
        if CFG.get("do_crossfit_ensemble", True):
            for sid, pp in zip(ids_te, pte.tolist()):
                oos_probs[str(sid)].append(float(pp))

        thr = float(log.get("tnbc_val_thr", 0.5))
        pred = (pte >= thr).astype(int)
        f1v = f1_score(yte, pred, zero_division=0)
        accv = accuracy_score(yte, pred)

        pred_df = pd.DataFrame({"id": ids_te, "y": yte, "p": pte, "pred": pred})
        pred_df.to_csv(run_dir / "outer_test_pred.csv", index=False)

        run_res = {
            "rep": r,
            "seed": seed,
            "outer_auc": float(auc_te),
            "outer_f1": float(f1v),
            "outer_acc": float(accv),
            "thr": float(thr),
            **log
        }
        results.append(run_res)
        with open(run_dir / "run_result.json", "w", encoding="utf-8") as f:
            json.dump(run_res, f, indent=2)

        # =========================
        # SHAP evidence-2: Stage2(final) attribution on TNBC outer-test (OOS)
        # =========================
        if bool(CFG.get("shap_enable", False)) and bool(CFG.get("shap_stage2_per_run", True)):
            print(f"[SHAP] Evidence-2: Stage2(final) SHAP on TNBC outer-test for rep={r:02d}.")
            shap_dir2 = run_dir / "shap_stage2_final"
            shap_dev2 = str(CFG.get("shap_device", device))
            if shap_dev2.startswith("cuda") and (not torch.cuda.is_available()):
                shap_dev2 = "cpu"

            imp2 = compute_and_save_shap(
                enc=enc2, clf=clf2,
                genes=genes,
                df_background=df_tr,
                df_explain=df_te,
                label_col=label_col,
                out_dir=shap_dir2,
                prefix=f"stage2_final_rep{r:02d}_outertest",
                seed=seed,
                device=shap_dev2,
                background_n=int(CFG.get("shap_background_n", 128)),
                explain_n=CFG.get("shap_explain_n_stage2", None),
                bar_topn=int(CFG.get("shap_bar_topn", 50)),
                beeswarm_topn=int(CFG.get("shap_beeswarm_topn", 50)),
                fig_fmt=str(CFG.get("shap_fig_fmt", "pdf")),
                method=str(CFG.get("shap_method", "gradient")),
            )
            if imp2 is not None:
                imp2_idx = imp2.set_index("gene")
                v_abs = imp2_idx.loc[genes, "mean_abs_shap"].values.astype(float)
                v_mean = imp2_idx.loc[genes, "mean_shap"].values.astype(float)
                stage2_shap_meanabs_list.append(v_abs)
                stage2_shap_mean_list.append(v_mean)

                topk_freq_K = int(CFG.get("shap_topk_freq_k", 50))
                for g in imp2.head(topk_freq_K)["gene"].tolist():
                    stage2_topk_freq[str(g)] += 1

        print(f"[cv_rep{r:02d}_fold00] OUTER auc={auc_te:.4f} | F1@thr={f1v:.4f} ACC@thr={accv:.4f} thr={thr:.3f}")

    summ = pd.DataFrame(results)
    summ_path = out_root / "summary.csv"
    summ.to_csv(summ_path, index=False)

    aucs = summ["outer_auc"].values.astype(float)
    print("\n" + "=" * 80)
    print(f"[SUMMARY] repeats={CFG['repeats']}")
    print(f"[SUMMARY] outer_auc mean={np.nanmean(aucs):.4f} std={np.nanstd(aucs):.4f} min={np.nanmin(aucs):.4f} max={np.nanmax(aucs):.4f}")
    print(f"[SUMMARY] saved: {summ_path}")

    # cross-fit ensemble AUC (out-of-sample aggregation)
    if CFG.get("do_crossfit_ensemble", True):
        ids_all = df_tnbc_al.index.astype(str).tolist()
        y_all = df_tnbc_al[label_col].astype(int).values
        p_all = []
        miss = 0
        for sid in ids_all:
            lst = oos_probs.get(str(sid), [])
            if len(lst) == 0:
                p_all.append(np.nan)
                miss += 1
            else:
                p_all.append(float(np.mean(lst)))
        p_all = np.asarray(p_all, dtype=float)

        ok = ~np.isnan(p_all)
        if ok.sum() >= 10 and len(np.unique(y_all[ok])) == 2:
            auc_cf = roc_auc_score(y_all[ok], p_all[ok])
            print(f"[CROSSFIT] OOS-ensemble AUC on TNBC_AL (n_ok={int(ok.sum())}, miss={miss}) = {auc_cf:.4f}")
            pd.DataFrame({"id": ids_all, "y": y_all, "p_oos_mean": p_all}).to_csv(out_root / "crossfit_oos_pred.csv", index=False)
            print(f"[CROSSFIT] saved: {out_root/'crossfit_oos_pred.csv'}")
        else:
            print(f"[CROSSFIT] not enough oos coverage or label variety. miss={miss} ok={int(ok.sum())}")

    
    # =========================
    # SHAP aggregation + Stage1 vs Stage2 comparison (for Figure 5.2/5.3 style)
    # =========================
    if bool(CFG.get("shap_enable", False)) and len(stage2_shap_meanabs_list) > 0:
        R = len(stage2_shap_meanabs_list)
        M_abs = np.vstack(stage2_shap_meanabs_list)  # [R, G]
        M_mean = np.vstack(stage2_shap_mean_list)    # [R, G]

        mean_abs = np.mean(M_abs, axis=0)
        std_abs = np.std(M_abs, axis=0)
        mean_sh = np.mean(M_mean, axis=0)

        freq = np.asarray([stage2_topk_freq.get(g, 0) / float(R) for g in genes], dtype=float)

        df_stage2 = pd.DataFrame(
            {
                "gene": genes,
                "mean_abs_shap": mean_abs.astype(float),
                "std_abs_shap": std_abs.astype(float),
                "mean_shap": mean_sh.astype(float),
                "topk_freq": freq.astype(float),
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df_stage2["rank"] = np.arange(1, len(df_stage2) + 1)

        out_shap2 = out_root / "shap_stage2_agg"
        ensure_dir(out_shap2)
        df_stage2.to_csv(out_shap2 / "stage2_final_agg_shap_meanabs.csv", index=False)

        # bar plot for aggregated Stage2 importance
        try:
            topn = int(CFG.get("shap_bar_topn", 50))
            fig_fmt = str(CFG.get("shap_fig_fmt", "pdf"))
            top = df_stage2.head(topn).iloc[::-1]
            plt.figure(figsize=(9.0, max(4.0, 0.22 * len(top) + 1.0)))
            plt.barh(top["gene"].tolist(), top["mean_abs_shap"].tolist())
            plt.xlabel("mean(|SHAP|) on logit (aggregated across repeats)")
            plt.tight_layout()
            plt.savefig(out_shap2 / f"stage2_final_agg_bar_top{topn}.{fig_fmt}", bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[SHAP] Stage2-agg bar plot failed: {e}")

        # compare Stage1(source-only) vs Stage2(final)
        if shap_stage1_imp is not None:
            try:
                df1 = shap_stage1_imp.set_index("gene")
                v1 = df1.loc[genes, "mean_abs_shap"].values.astype(float)
                v2 = df_stage2.set_index("gene").loc[genes, "mean_abs_shap"].values.astype(float)

                # overlap@N
                Ns = [10, 20, 30, 50, 100]
                top1 = shap_stage1_imp["gene"].tolist()
                top2 = df_stage2["gene"].tolist()
                rows = []
                for N in Ns:
                    s1 = set(top1[:N]); s2 = set(top2[:N])
                    rows.append({"N": int(N), "overlap": float(len(s1 & s2) / float(N)), "jaccard": float(len(s1 & s2) / float(len(s1 | s2)))})
                df_ov = pd.DataFrame(rows)
                out_cmp = out_root / "shap_compare_stage1_vs_stage2"
                ensure_dir(out_cmp)
                df_ov.to_csv(out_cmp / "overlap_at_N.csv", index=False)

                # overlap curve plot
                fig_fmt = str(CFG.get("shap_fig_fmt", "pdf"))
                plt.figure(figsize=(6.0, 3.8))
                plt.plot(df_ov["N"].values, df_ov["overlap"].values, marker="o")
                plt.xlabel("Top-N")
                plt.ylabel("Overlap@N")
                plt.tight_layout()
                plt.savefig(out_cmp / f"overlap_curve.{fig_fmt}", bbox_inches="tight")
                plt.close()

                # spearman-like correlation (rank corr)
                ra = pd.Series(v1).rank(method="average").values
                rb = pd.Series(v2).rank(method="average").values
                spearman = float(np.corrcoef(ra, rb)[0, 1])

                # delta (log scale) for "importance shift"
                eps = 1e-12
                delta = np.log(v2 + eps) - np.log(v1 + eps)
                df_delta = pd.DataFrame({"gene": genes, "delta_log_importance": delta.astype(float), "stage1_mean_abs": v1.astype(float), "stage2_mean_abs": v2.astype(float)})
                df_delta = df_delta.sort_values("delta_log_importance", ascending=False).reset_index(drop=True)
                df_delta.to_csv(out_cmp / "delta_log_importance.csv", index=False)

                # delta plots (top up / top down)
                k = int(CFG.get("shap_delta_topn", 20))
                up = df_delta.head(k).iloc[::-1]
                dn = df_delta.tail(k)  # already ascending at tail
                plt.figure(figsize=(9.0, max(4.0, 0.22 * len(up) + 1.0)))
                plt.barh(up["gene"].tolist(), up["delta_log_importance"].tolist())
                plt.xlabel("log(mean_abs_stage2) - log(mean_abs_stage1)")
                plt.tight_layout()
                plt.savefig(out_cmp / f"delta_top{int(k)}_up.{fig_fmt}", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(9.0, max(4.0, 0.22 * len(dn) + 1.0)))
                plt.barh(dn["gene"].tolist(), dn["delta_log_importance"].tolist())
                plt.xlabel("log(mean_abs_stage2) - log(mean_abs_stage1)")
                plt.tight_layout()
                plt.savefig(out_cmp / f"delta_top{int(k)}_down.{fig_fmt}", bbox_inches="tight")
                plt.close()

                # scatter (log1p)
                plt.figure(figsize=(5.5, 5.0))
                plt.scatter(np.log1p(v1), np.log1p(v2), s=8)
                plt.xlabel("log(1+Stage1 mean(|SHAP|))")
                plt.ylabel("log(1+Stage2 mean(|SHAP|))")
                plt.tight_layout()
                plt.savefig(out_cmp / f"scatter_log1p.{fig_fmt}", bbox_inches="tight")
                plt.close()

                with open(out_cmp / "compare_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"spearman_rankcorr": spearman, "runs_stage2": int(R)}, f, indent=2)

                print(f"[SHAP] Compare saved under: {out_cmp}")
            except Exception as e:
                print(f"[SHAP] Stage1 vs Stage2 comparison failed: {e}")

        print(f"[SHAP] Stage2 agg saved under: {out_shap2}")

print("=" * 80)


if __name__ == "__main__":
    CFG = dict(
        # ---- output ----
        out_dir="/root/grl/out__topk1000_Aplus_lastlayer_L2SP_mixupOFF_800",
        auto_tag_outdir=True,  # append a short tag to out_dir for safety
        seed=42,

        # ---- data ----
        genes_json="/root/grl/data/genes.json",
        oth_labeled_csv="/root/grl/data/OTH_labeled_std.csv",
        tnbc_array_unlabeled_csv="/root/grl/data/BRA_unlabeled_std.csv",
        tnbc_array_labeled_csv="/root/grl/data/TNBC_labeled_std.csv",
        label_col="Response",
        index_col=None,

        # ---- leak-safe topK gene selection (ONLY uses TNBC_AU) ----
        topk_enable=True,
        topk_k=800,               # try 800 / 1000 / 1500
        topk_metric="mad",         # "mad" is usually more robust than var
        topk_keep_genes_json=None, # optional: put your immune prior genes here (json list). Keep leak-safe.

        # ---- reuse (recommend False when changing topK) ----
        reuse_stage0=False,
        reuse_stage1=False,

        # ---- evaluation protocol ----
        repeats=50,
        outer_test_size=14/69,
        inner_val_size=14/55,
        tie_break_weight=0.2,
        do_crossfit_ensemble=True,

        # ---- SHAP (feature attribution) ----
        shap_enable=True,                 # enable SHAP outputs (tables + plots)
        shap_device="cuda",               # "cuda" or "cpu" (auto-fallback to cpu if cuda unavailable)
        shap_method="gradient",           # "gradient" (robust) or "deep" (faster but pickier)
        shap_fig_fmt="pdf",               # "pdf" or "png"
        shap_background_n=128,            # number of background samples for SHAP
        shap_explain_n_stage1=None,       # None = use all OTH-val
        shap_explain_n_stage2=None,       # None = use all TNBC outer-test
        shap_bar_topn=50,                 # Top-N genes in bar plot
        shap_beeswarm_topn=50,            # Top-N genes in beeswarm plot
        shap_topk_freq_k=50,              # K used to count topK frequency across repeats (Stage2)
        shap_stage2_per_run=True,         # compute and save Stage2 SHAP per repeat run
        shap_source_force_source_only=True,        # when stage1_use_grl=True, also train a source-only Stage1 for SHAP
        reuse_stage1_sourceonly_for_shap=True,     # reuse the extra source-only Stage1 ckpts if exist
        shap_delta_topn=20,               # Top-N for delta plots (Stage1 vs Stage2)

        # ---- model ----
        z_dim=256,
        enc_hidden=1024,  # keep your proven setting first
        clf_hidden=256,   # keep your proven setting first (don't shrink until topK validated)
        dropout=0.1,
        wd=1e-4,

        # ---- Stage0 (DAE) ----
        ae_hidden=1024,
        ae_lr=1e-3,
        ae_epochs=50,
        ae_batch=128,
        ae_noise_std=0.05,

        # ---- Stage1 ----
        stage1_lr=5e-4,
        stage1_epochs=30,
        stage1_batch=128,
        stage1_enc_lr_mult=0.2,

        # ---- Stage1 (GRL / DANN) ----
        stage1_use_grl=True,
        stage1_dom_hidden=256,
        stage1_dom_dropout=0.1,
        stage1_dom_lr_mult=1.0,
        stage1_dom_loss_weight=1.0,
        stage1_grl_schedule="sigmoid",   # "sigmoid"(DANN) or "linear"
        stage1_grl_warmup=0.30,          # fraction of stage1 epochs with λ=0
        stage1_grl_gamma=10.0,           # only for "sigmoid"
        stage1_grl_lambda_max=0.5,       # keep small first; tune in {0.1,0.3,0.5,1.0}

        # ---- Stage2 (A+ lastlayer + L2SP) ----
        stage2_epochs=80,
        stage2_batch_lab=32,
        phase1_ratio=0.20,         # head-only first 20% epochs
        stage2_lr_head=0.0003,
        stage2_lr_enc_mult=0.02,   # your best-known stable region
        lambda_l2sp=0.001,          # your best-known stable region

        # ---- Mixup (Stage2 labeled only) ----
        use_mixup=False,           # VersionA: False(0.7433); VersionB: True(0.7380)
        mixup_alpha=0.4,           # 0.2~0.8 are common
    )

    main(CFG)
