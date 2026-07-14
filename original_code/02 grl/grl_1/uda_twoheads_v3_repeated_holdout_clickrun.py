# uda_twoheads_v3_repeated_holdout_clickrun.py
# Click-Run: v3 + Repeated Holdout on TNBC microarray labeled
#
# What it does:
# - Keep your v3 training pipeline (Stage0 DAE -> Stage1a/1b -> Stage2 UDA(2 heads)+SSDA anchor)
# - Run repeated stratified holdout splits on TNBC_array_labeled:
#     for each repeat:
#         TNBC labeled -> train/val/test (test NEVER used for checkpoint selection)
#         Stage2 anchor train uses TNBC_train, checkpoint selection uses TNBC_val
#         Final report on TNBC_test AUC
# - Aggregate AUC over repeats and save summary CSV/JSON.

import json, math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------- Datasets -----------------
class LabeledDS(Dataset):
    """Return: x, y, id, platform_id, cancer_id"""
    def __init__(self, df: pd.DataFrame, gene_list: List[str], label_col: str, platform_id: int, cancer_id: int):
        X = df[gene_list].astype(np.float32).values
        y = df[label_col].astype(np.int64).values
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.ids = df.index.astype(str).tolist()
        self.platform_id = int(platform_id)
        self.cancer_id = int(cancer_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.ids[i], self.platform_id, self.cancer_id


class UnlabeledDS(Dataset):
    """Return: x, id, platform_id, cancer_id"""
    def __init__(self, df: pd.DataFrame, gene_list: List[str], platform_id: int, cancer_id: int):
        X = df[gene_list].astype(np.float32).values
        self.X = torch.from_numpy(X)
        self.ids = df.index.astype(str).tolist()
        self.platform_id = int(platform_id)
        self.cancer_id = int(cancer_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.X[i], self.ids[i], self.platform_id, self.cancer_id


class TensorOnlyDS(Dataset):
    """Return: x only (for AE pretrain pool)"""
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return self.X[i]


# ----------------- Losses / Tools -----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.g = float(gamma)
        self.a = float(alpha)

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        return (self.a * (1 - pt) ** self.g * ce).mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.1)

    def forward(self, z, y):
        return ((z - self.centers[y]) ** 2).sum(dim=1).mean()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grl(x, lambd):
    return GradReverse.apply(x, lambd)


def orth_loss(z_inv, z_bc):
    """decorrelate z_inv and z_bc"""
    if z_inv.size(0) <= 1:
        return z_inv.new_tensor(0.0)
    z_inv_c = z_inv - z_inv.mean(dim=0, keepdim=True)
    z_bc_c  = z_bc  - z_bc.mean(dim=0, keepdim=True)
    C = torch.matmul(z_inv_c.t(), z_bc_c) / (z_inv_c.size(0) - 1)
    return torch.norm(C, p="fro")


# ----------------- Model -----------------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, z_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, z_dim)
        )

    def forward(self, x):
        return self.net(x)


class DenoisingAE(nn.Module):
    def __init__(self, in_dim, z_dim=256, dropout=0.1):
        super().__init__()
        self.enc = MLPEncoder(in_dim, z_dim, dropout)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 512), nn.ReLU(),
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, in_dim)
        )

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return z, xhat


class CrossAttentionHead(nn.Module):
    def __init__(self, num_genes, h):
        super().__init__()
        self.val = nn.Linear(1, h)
        self.gene_id = nn.Embedding(num_genes, h)
        self.Wq = nn.Linear(h, h)
        self.Wk = nn.Linear(h, h)
        self.Wv = nn.Linear(h, h)

    def forward(self, x, z):
        B, N = x.shape
        device = x.device
        gene_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        t = self.val(x.unsqueeze(-1)) + self.gene_id(gene_ids)  # [B,N,h]
        Q = self.Wq(z).unsqueeze(1)                             # [B,1,h]
        K = self.Wk(t)                                          # [B,N,h]
        V = self.Wv(t)                                          # [B,N,h]
        logits = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))  # [B,1,N]
        w = torch.softmax(logits, dim=-1).squeeze(1)            # [B,N]
        pooled = torch.matmul(w.unsqueeze(1), V).squeeze(1)     # [B,h]
        return pooled, w


class FeatureDecomposer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_inv = nn.Linear(dim, dim)
        self.to_bc  = nn.Linear(dim, dim)

    def forward(self, h):
        return self.to_inv(h), self.to_bc(h)


class UDAClassifierTwoHeads(nn.Module):
    def __init__(self, in_dim, num_genes, z_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, z_dim, dropout)
        self.attn = CrossAttentionHead(num_genes, z_dim)
        self.decomp = FeatureDecomposer(z_dim)

        # task head (uses z_inv ONLY)
        self.head = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(z_dim // 2, 2)
        )

        # platform head (RNAseq vs microarray) -> use z_bc
        self.platform = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(),
            nn.Linear(z_dim // 2, 2)
        )

        # cancer head (OTH vs TNBC) -> use z_inv
        self.cancer = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(),
            nn.Linear(z_dim // 2, 2)
        )

    def forward(self, x, grl_platform=None, grl_cancer=None):
        z_enc = self.encoder(x)
        pooled, w = self.attn(x, z_enc)
        z_inv, z_bc = self.decomp(pooled)

        logits = self.head(z_inv)  # ✅ task uses invariant only
        z_all = z_inv

        plat_logits = None
        cancer_logits = None
        if grl_platform is not None:
            plat_logits = self.platform(grl(z_bc, grl_platform))
        if grl_cancer is not None:
            cancer_logits = self.cancer(grl(z_inv, grl_cancer))

        return logits, plat_logits, cancer_logits, z_all, w, pooled, z_inv, z_bc


# ----------------- Utils -----------------
def seed_all(sd=42):
    import random
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_line_plot(values, ylabel, title, out_png):
    plt.figure()
    xs = np.arange(1, len(values) + 1)
    plt.plot(xs, values)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_curve_csv(values, out_csv, colname):
    df = pd.DataFrame({"epoch": np.arange(1, len(values) + 1), colname: values})
    df.to_csv(out_csv, index=False)


def _save_roc(y_true, y_score, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    if len(np.unique(y_true)) > 1:
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_score):.4f}")
    else:
        plt.plot(fpr, tpr, label="AUC=nan")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _bin_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ACC@0.5": float(accuracy_score(y_true, y_pred)),
        "F1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "N": int(len(y_true))
    }


@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        xb, yb, *_ = batch
        xb, yb = xb.to(device), yb.to(device)
        logits, _, _, _, _, _, _, _ = model(xb, None, None)
        p = torch.softmax(logits, 1)[:, 1]
        probs.append(p.detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())
    y = np.concatenate(labels) if labels else np.array([])
    p = np.concatenate(probs) if probs else np.array([])
    auc = roc_auc_score(y, p) if (len(y) > 0 and len(np.unique(y)) > 1) else float("nan")
    return auc, p, y


def as_long_label_batch(v, B: int, device):
    if torch.is_tensor(v):
        v = v.to(device)
        if v.dtype != torch.long:
            v = v.long()
        if v.ndim == 0:
            return v.view(1).expand(B)
        if v.ndim == 1:
            if v.numel() == B:
                return v
            if v.numel() == 1:
                return v.expand(B)
        raise ValueError(f"Label tensor has unexpected shape: {tuple(v.shape)} (B={B})")
    return torch.full((B,), int(v), dtype=torch.long, device=device)


def read_genes_json(genes_json: Optional[str]):
    if not genes_json:
        return None
    obj = json.load(open(genes_json, "r", encoding="utf-8"))
    if isinstance(obj, dict) and "genes" in obj:
        return list(obj["genes"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f"genes.json format not recognized: {genes_json}")


def align_to_genes(df: pd.DataFrame, genes: List[str], label_col: Optional[str] = None):
    df = df.copy()
    keep_label = (label_col is not None and label_col in df.columns)

    cols = [c for c in df.columns if (c in genes) or (keep_label and c == label_col)]
    df = df[cols].copy()

    missing = [g for g in genes if g not in df.columns]
    for g in missing:
        df[g] = 0.0

    if keep_label:
        df = df[genes + [label_col]]
    else:
        df = df[genes]

    X = df[genes].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        print(f"[WARN] NaN={n_nan}, Inf={n_inf} -> impute NaN by col mean, Inf->NaN")
        X = np.where(np.isinf(X), np.nan, X)
        col_mean = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])
        df[genes] = X

    return df


def sanitize_labels(df: pd.DataFrame, label_col: str, name: str):
    """Ensure label_col is numeric 0/1 without NaN; drop NaN labels with warning."""
    if label_col not in df.columns:
        raise ValueError(f"[{name}] Missing label column: {label_col}")
    y = pd.to_numeric(df[label_col], errors="coerce")
    n_nan = int(y.isna().sum())
    if n_nan > 0:
        print(f"[WARN] [{name}] label has NaN={n_nan}. Dropping those rows.")
        df = df.loc[~y.isna()].copy()
        y = pd.to_numeric(df[label_col], errors="coerce")
    # map possible floats to ints safely
    df[label_col] = y.astype(int)
    # sanity
    vc = df[label_col].value_counts().to_dict()
    print(f"[LABEL] [{name}] label counts: {vc}")
    return df


def ramp(p: float, warm: float = 0.2) -> float:
    """0..warm => 0; warm..1 => linear to 1"""
    if p < warm:
        return 0.0
    if warm >= 1.0:
        return 1.0
    return float(min(1.0, (p - warm) / (1.0 - warm)))


def cycle_loader(loader):
    while True:
        for b in loader:
            yield b


def pick_balanced_pseudolabels(pt, tau, max_per_class=8, min_per_class=2):
    conf, pseudo = pt.max(dim=1)
    base = conf >= tau
    B = pt.size(0)
    m = torch.zeros(B, dtype=torch.bool, device=pt.device)

    for cls in [0, 1]:
        idx = torch.where(base & (pseudo == cls))[0]
        if idx.numel() > 0:
            order = torch.argsort(conf[idx], descending=True)
            keep = idx[order[:max_per_class]]
            m[keep] = True

    for cls in [0, 1]:
        has_cls = int(((pseudo[m] == cls).sum().item())) if m.any() else 0
        if has_cls == 0:
            un = torch.where(~m)[0]
            if un.numel() > 0:
                scores = pt[un, cls]
                order = torch.argsort(scores, descending=True)
                take = un[order[:min_per_class]]
                m[take] = True
                pseudo[take] = cls

    if m.sum() == 0:
        k = min(max(1, B // 4), B)
        order = torch.argsort(conf, descending=True)
        m[order[:k]] = True

    mask_rate = float(m.float().mean().item())
    pos_rate  = float((pseudo[m] == 1).float().mean().item()) if m.any() else 0.0
    return m, pseudo, conf, mask_rate, pos_rate


def t_confidence_interval(xs: List[float], alpha=0.05) -> Tuple[float, float]:
    """Return (low, high) approx CI using t-dist; fallback to normal if scipy not available."""
    xs = [float(x) for x in xs if (x is not None and not np.isnan(x))]
    n = len(xs)
    if n <= 1:
        m = float(xs[0]) if n == 1 else float("nan")
        return m, m
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1))
    se = s / math.sqrt(n)
    # try scipy
    try:
        from scipy.stats import t
        tval = float(t.ppf(1 - alpha/2, df=n-1))
    except Exception:
        # normal approximation
        tval = 1.96
    return m - tval * se, m + tval * se


# ----------------- Training core: one repeat -----------------
def run_one_repeat(cfg: Dict[str, Any], repeat_id: int, out_dir: Path,
                   oth: pd.DataFrame, tnbc_arr_ul: pd.DataFrame,
                   tnbc_rna_ul: Optional[pd.DataFrame], tnbc_arr_lab: pd.DataFrame,
                   gene_list: List[str], device: torch.device,
                   cached_pretrained_encoder: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    """
    One repeated holdout run:
    - TNBC labeled -> train/val/test (stratified, repeat-specific seed)
    - Stage0 optional (pretrain encoder; strict no-peek by default)
    - Stage1a/b on OTH split
    - Stage2 UDA+anchor where anchor=train, selection=val
    - Final eval on TNBC test
    """
    run_out = out_dir / f"rep_{repeat_id:02d}"
    run_out.mkdir(parents=True, exist_ok=True)

    # fixed labels
    PLAT_RNASEQ = 0
    PLAT_ARRAY  = 1
    CANC_OTH  = 0
    CANC_TNBC = 1

    # ---------------- OTH split (keep consistent or per-repeat) ----------------
    y_oth = oth[cfg["label_col"]].astype(int).values
    idx_all = np.arange(len(oth))
    tr_idx, va_idx = train_test_split(
        idx_all, test_size=cfg["val_ratio"], random_state=cfg["seed"],
        stratify=y_oth if len(np.unique(y_oth)) == 2 else None
    )
    oth_tr = oth.iloc[tr_idx].copy()
    oth_va = oth.iloc[va_idx].copy()

    trL = DataLoader(
        LabeledDS(oth_tr, gene_list, cfg["label_col"], platform_id=PLAT_RNASEQ, cancer_id=CANC_OTH),
        batch_size=cfg["batch"], shuffle=True, drop_last=True
    )
    vaL = DataLoader(
        LabeledDS(oth_va, gene_list, cfg["label_col"], platform_id=PLAT_RNASEQ, cancer_id=CANC_OTH),
        batch_size=cfg["batch"], shuffle=False, drop_last=False
    )

    # ---------------- TNBC labeled repeated holdout split ----------------
    y_t = tnbc_arr_lab[cfg["label_col"]].astype(int).values
    idx_t = np.arange(len(tnbc_arr_lab))

    # First split train_val vs test
    trva_idx, te_idx = train_test_split(
        idx_t,
        test_size=cfg["tnbc_test_ratio"],
        random_state=cfg["seed"] + 1000 + repeat_id * 13,
        stratify=y_t if len(np.unique(y_t)) == 2 else None
    )

    t_trva = tnbc_arr_lab.iloc[trva_idx].copy()
    t_test = tnbc_arr_lab.iloc[te_idx].copy()

    # Then split train vs val (for checkpoint selection)
    y_trva = t_trva[cfg["label_col"]].astype(int).values
    idx_trva = np.arange(len(t_trva))
    tr2_idx, va2_idx = train_test_split(
        idx_trva,
        test_size=cfg["tnbc_val_ratio_within_trainval"],
        random_state=cfg["seed"] + 2000 + repeat_id * 17,
        stratify=y_trva if len(np.unique(y_trva)) == 2 else None
    )

    t_train = t_trva.iloc[tr2_idx].copy()
    t_val   = t_trva.iloc[va2_idx].copy()

    print(f"[REP {repeat_id:02d}] TNBC labeled split: train={len(t_train)} val={len(t_val)} test={len(t_test)}")

    tnbcArrLabTrain = DataLoader(
        LabeledDS(t_train, gene_list, cfg["label_col"], platform_id=PLAT_ARRAY, cancer_id=CANC_TNBC),
        batch_size=cfg["batch"], shuffle=True, drop_last=True
    )
    tnbcArrLabVal = DataLoader(
        LabeledDS(t_val, gene_list, cfg["label_col"], platform_id=PLAT_ARRAY, cancer_id=CANC_TNBC),
        batch_size=cfg["batch"], shuffle=False, drop_last=False
    )
    tnbcArrLabTest = DataLoader(
        LabeledDS(t_test, gene_list, cfg["label_col"], platform_id=PLAT_ARRAY, cancer_id=CANC_TNBC),
        batch_size=cfg["batch"], shuffle=False, drop_last=False
    )

    # ---------------- unlabeled loaders ----------------
    tnbcArrUL = DataLoader(
        UnlabeledDS(tnbc_arr_ul, gene_list, platform_id=PLAT_ARRAY, cancer_id=CANC_TNBC),
        batch_size=cfg["batch"], shuffle=True, drop_last=True
    )

    tnbcRnaUL = None
    if tnbc_rna_ul is not None:
        tnbcRnaUL = DataLoader(
            UnlabeledDS(tnbc_rna_ul, gene_list, platform_id=PLAT_RNASEQ, cancer_id=CANC_TNBC),
            batch_size=cfg["batch"], shuffle=True, drop_last=True
        )

    # ---------------- model ----------------
    in_dim = len(gene_list)
    model = UDAClassifierTwoHeads(in_dim, num_genes=in_dim, z_dim=cfg["z_dim"], dropout=cfg["dropout"]).to(device)
    focal = FocalLoss(cfg["focal_gamma"], cfg["focal_alpha"])
    center = CenterLoss(2, cfg["z_dim"]).to(device)

    # =========================
    # Stage0: DAE pretrain
    # =========================
    # strict_no_peek:
    # - If True: AE pool must NOT include TNBC labeled val/test samples
    # - If False: can reuse cached encoder (trained on full pool) to save time
    pretrain_mse_log = []
    enc_state = None

    if cfg["reuse_pretrain_encoder"] and (cached_pretrained_encoder is not None):
        model.encoder.load_state_dict(cached_pretrained_encoder, strict=True)
        print(f"[REP {repeat_id:02d}] [Stage0] Reused cached pretrained encoder.")
    else:
        if cfg["pretrain_epochs"] > 0:
            pool = []
            # OTH train (drop labels)
            pool.append(oth_tr[gene_list].values.astype(np.float32))
            # target unlabeled
            pool.append(tnbc_arr_ul[gene_list].values.astype(np.float32))
            # TNBC RNA unlabeled
            if tnbc_rna_ul is not None:
                pool.append(tnbc_rna_ul[gene_list].values.astype(np.float32))
            # optionally include TNBC labeled TRAIN ONLY (no-peek)
            if cfg["use_tnbc_train_in_ae_pool"]:
                pool.append(t_train[gene_list].values.astype(np.float32))

            X_pool = np.concatenate(pool, axis=0)
            print(f"[REP {repeat_id:02d}] [AE] pool size={X_pool.shape[0]} (strict_no_peek={cfg['strict_no_peek']})")

            ae = DenoisingAE(in_dim, cfg["z_dim"], cfg["dropout"]).to(device)
            opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg["lr_pre"])
            ae.train()

            dl_pool = DataLoader(TensorOnlyDS(X_pool), batch_size=cfg["batch"], shuffle=True, drop_last=True)

            for ep in range(cfg["pretrain_epochs"]):
                losses = []
                for xb in dl_pool:
                    xb = xb.to(device)
                    mask = (torch.rand_like(xb) > cfg["mask_ratio"]).float()
                    noisy = xb * mask + torch.randn_like(xb) * cfg["noise_std"]
                    _, xhat = ae(noisy)
                    loss = F.mse_loss(xhat, xb)

                    opt_ae.zero_grad()
                    loss.backward()
                    opt_ae.step()
                    losses.append(loss.item())

                ep_mse = float(np.mean(losses)) if losses else float("nan")
                pretrain_mse_log.append(ep_mse)
                if (ep % max(1, cfg["pretrain_log_every"])) == 0:
                    print(f"[REP {repeat_id:02d}] [Pretrain] epoch {ep} reconMSE={ep_mse:.4f}")

            enc_state = ae.enc.state_dict()
            model.encoder.load_state_dict(enc_state, strict=True)

            # save pretrain curves
            _save_curve_csv(pretrain_mse_log, run_out / "pretrain_reconMSE.csv", "reconMSE")
            _save_line_plot(pretrain_mse_log, "recon MSE", f"Pretrain MSE (rep {repeat_id})", run_out / "pretrain_reconMSE.png")
            torch.save(enc_state, run_out / "encoder_pretrained.pt")
            del ae

    # =========================
    # Stage1a: source supervised (freeze encoder)
    # =========================
    for p in model.encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
            {"params": center.parameters(), "lr": cfg["lr"] * 0.5},
        ],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_auc, bad = -1, 0
    auc_log_stage1a = []
    for ep in range(cfg["src_head_epochs"]):
        model.train()
        for xb, yb, _, _, _ in trL:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, _, z_all, _, _, z_inv, z_bc = model(xb, None, None)
            L_cls = focal(logits, yb) + cfg["lambda_center"] * center(z_all, yb)
            L_ort = orth_loss(z_inv, z_bc) * cfg["lambda_ortho"]
            loss = L_cls + L_ort

            opt.zero_grad()
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        auc, _, _ = evaluate_auc(model, vaL, device)
        auc_log_stage1a.append(float(auc))
        print(f"[REP {repeat_id:02d}] [Stage1a] epoch {ep} OTH-val AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save(model.state_dict(), run_out / "stage1a_best.pt")
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break

    if auc_log_stage1a:
        _save_curve_csv(auc_log_stage1a, run_out / "stage1a_auc.csv", "AUC")
        _save_line_plot(auc_log_stage1a, "AUC", f"Stage1a OTH-val AUC (rep {repeat_id})", run_out / "stage1a_auc.png")

    # =========================
    # Stage1b: source supervised (unfreeze last layers lightly)
    # =========================
    model.load_state_dict(safe_torch_load(run_out / "stage1a_best.pt", device), strict=True)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for n, p in model.encoder.named_parameters():
        if ("net.3" in n) or ("net.6" in n):
            p.requires_grad = True

    opt = torch.optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
            {"params": center.parameters(), "lr": cfg["lr"] * 0.5},
        ],
        lr=cfg["lr"] * 0.3,
        weight_decay=cfg["weight_decay"],
    )

    best_auc, bad = -1, 0
    auc_log_stage1b = []
    for ep in range(cfg["src_ft_epochs"]):
        model.train()
        for xb, yb, _, _, _ in trL:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, _, z_all, _, _, z_inv, z_bc = model(xb, None, None)
            L_cls = focal(logits, yb) + cfg["lambda_center"] * center(z_all, yb)
            L_ort = orth_loss(z_inv, z_bc) * cfg["lambda_ortho"]
            loss = L_cls + L_ort

            opt.zero_grad()
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        auc, _, _ = evaluate_auc(model, vaL, device)
        auc_log_stage1b.append(float(auc))
        print(f"[REP {repeat_id:02d}] [Stage1b] epoch {ep} OTH-val AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save(model.state_dict(), run_out / "stage1b_best.pt")
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break

    if auc_log_stage1b:
        _save_curve_csv(auc_log_stage1b, run_out / "stage1b_auc.csv", "AUC")
        _save_line_plot(auc_log_stage1b, "AUC", f"Stage1b OTH-val AUC (rep {repeat_id})", run_out / "stage1b_auc.png")

    model.load_state_dict(safe_torch_load(run_out / "stage1b_best.pt", device), strict=True)

    # =========================
    # Stage2: UDA with TWO heads + SSDA target anchor + warmup schedule
    # =========================
    it_src = cycle_loader(trL)
    it_arr = cycle_loader(tnbcArrUL)
    it_rna = cycle_loader(tnbcRnaUL) if tnbcRnaUL is not None else None
    it_tlab = cycle_loader(tnbcArrLabTrain)

    steps_per_epoch = cfg["steps_per_epoch"]
    if steps_per_epoch <= 0:
        steps_per_epoch = min(len(trL), len(tnbcArrUL))
        if tnbcRnaUL is not None:
            steps_per_epoch = min(steps_per_epoch, len(tnbcRnaUL))
        steps_per_epoch = max(1, steps_per_epoch)

    total_steps = max(1, steps_per_epoch * cfg["uda_epochs"])
    gstep = 0

    opt = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": center.parameters(), "lr": cfg["lr"] * 0.5},
        ],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_oth_auc = -1.0
    best_oth_epoch = -1
    best_val_auc = -1.0
    best_val_epoch = -1

    auc_log_uda_oth = []
    auc_log_uda_val = []
    tau_last = None

    warned_platform_fallback = False
    warned_cancer_disabled = False

    for ep in range(cfg["uda_epochs"]):
        model.train()
        ep_mask_rate = []
        ep_pos_rate = []

        for _ in range(steps_per_epoch):
            prog = gstep / (total_steps - 1) if total_steps > 1 else 1.0
            r_all = ramp(prog, warm=cfg["warmup"])
            grl_plat = cfg["lambda_platform"] * r_all
            grl_canc = cfg["lambda_cancer"] * r_all

            # source batch (OTH RNAseq)
            xb_s, yb_s, _, plat_s, canc_s = next(it_src)
            xb_s, yb_s = xb_s.to(device), yb_s.to(device)
            plat_s = as_long_label_batch(plat_s, xb_s.size(0), device)
            canc_s = as_long_label_batch(canc_s, xb_s.size(0), device)

            # target microarray batch (TNBC array unlabeled)
            xb_a, _, plat_a, canc_a = next(it_arr)
            xb_a = xb_a.to(device)
            plat_a = as_long_label_batch(plat_a, xb_a.size(0), device)
            canc_a = as_long_label_batch(canc_a, xb_a.size(0), device)

            # optional TNBC RNAseq batch
            xb_r = None
            plat_r = None
            canc_r = None
            if it_rna is not None:
                xb_r, _, plat_r0, canc_r0 = next(it_rna)
                xb_r = xb_r.to(device)
                plat_r = as_long_label_batch(plat_r0, xb_r.size(0), device)
                canc_r = as_long_label_batch(canc_r0, xb_r.size(0), device)

            # forward (with scheduled GRL)
            logits_s, platlog_s, canclog_s, z_all_s, _, _, z_inv_s, z_bc_s = model(xb_s, grl_plat, grl_canc)
            logits_a, platlog_a, canclog_a, z_all_a, _, _, z_inv_a, z_bc_a = model(xb_a, grl_plat, grl_canc)
            if xb_r is not None:
                logits_r, platlog_r, canclog_r, z_all_r, _, _, z_inv_r, z_bc_r = model(xb_r, grl_plat, grl_canc)

            # source task loss
            L_src = focal(logits_s, yb_s) + cfg["lambda_center"] * center(z_all_s, yb_s)

            # SSDA target anchor (TNBC train)
            xb_t, yb_t, _, _, _ = next(it_tlab)
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            logits_t, _, _, z_all_t, _, _, z_inv_t, z_bc_t = model(xb_t, None, None)
            L_tlab = (focal(logits_t, yb_t) + cfg["lambda_center"] * center(z_all_t, yb_t)) * cfg["lambda_tlab"]

            # pseudo-label on microarray target (scheduled)
            pt_a = torch.softmax(logits_a, 1)
            tau = cfg["pl_tau_start"] + (cfg["pl_tau_end"] - cfg["pl_tau_start"]) * prog
            tau_last = float(tau)

            mask_pl, pseudo, conf, mask_rate, pos_rate = pick_balanced_pseudolabels(
                pt_a, tau=tau,
                max_per_class=cfg["pl_max_per_class"],
                min_per_class=cfg["pl_min_per_class"]
            )
            ep_mask_rate.append(mask_rate)
            ep_pos_rate.append(pos_rate)

            if cfg["lambda_entropy"] > 0:
                if cfg["ent_on_conf"] and mask_pl.any():
                    pt_sel = pt_a[mask_pl]
                else:
                    pt_sel = pt_a
                L_ent = (-(pt_sel * (pt_sel.clamp_min(1e-8)).log()).sum(dim=1)).mean()
                L_ent = L_ent * (cfg["lambda_entropy"] * r_all)
            else:
                L_ent = 0.0

            if cfg["lambda_pl"] > 0 and mask_pl.any():
                L_pl = F.cross_entropy(logits_a[mask_pl], pseudo[mask_pl]) * (cfg["lambda_pl"] * r_all)
            else:
                L_pl = 0.0

            # platform adversarial loss (scheduled)
            L_plat = 0.0
            if cfg["lambda_platform_loss"] > 0 and r_all > 0:
                if xb_r is not None and cfg["platform_within_tnbc"]:
                    L_plat = F.cross_entropy(platlog_a, plat_a) + F.cross_entropy(platlog_r, plat_r)
                else:
                    if cfg["fallback_use_oth_for_platform_if_no_tnbc_rna"]:
                        if not warned_platform_fallback:
                            print("[WARN] No TNBC RNA UL for within-TNBC platform loss -> fallback to (OTH RNA) vs (TNBC array). "
                                  "This may re-introduce platform/cancer collinearity.")
                            warned_platform_fallback = True
                        L_plat = F.cross_entropy(platlog_a, plat_a) + F.cross_entropy(platlog_s, plat_s)
                    else:
                        L_plat = 0.0
                L_plat = L_plat * (cfg["lambda_platform_loss"] * r_all)

            # cancer adversarial loss (scheduled)
            L_canc = 0.0
            if cfg["lambda_cancer_loss"] > 0 and r_all > 0:
                if xb_r is not None and cfg["cancer_within_rnaseq"]:
                    L_canc = F.cross_entropy(canclog_s, canc_s) + F.cross_entropy(canclog_r, canc_r)
                    L_canc = L_canc * (cfg["lambda_cancer_loss"] * r_all)
                else:
                    if cfg["disable_cancer_if_no_tnbc_rna"]:
                        if not warned_cancer_disabled:
                            print("[WARN] No TNBC RNA UL -> cancer loss disabled (to avoid cancer/platform collinearity via microarray).")
                            warned_cancer_disabled = True
                        L_canc = 0.0
                    else:
                        L_canc = (F.cross_entropy(canclog_s, canc_s) + F.cross_entropy(canclog_a, canc_a)) * (cfg["lambda_cancer_loss"] * r_all)

            # orth regularization
            L_ortho = 0.5 * cfg["lambda_ortho"] * (orth_loss(z_inv_s, z_bc_s) + orth_loss(z_inv_a, z_bc_a))
            if xb_r is not None:
                L_ortho = L_ortho + (cfg["lambda_ortho"] * 0.25 * orth_loss(z_inv_r, z_bc_r))

            loss = L_src + L_tlab + L_ent + L_pl + L_plat + L_canc + L_ortho

            opt.zero_grad()
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            gstep += 1

        # ---- monitor: OTH-val + TNBC-val (selection) ----
        auc_oth, _, _ = evaluate_auc(model, vaL, device)
        auc_val, _, _ = evaluate_auc(model, tnbcArrLabVal, device)
        auc_log_uda_oth.append(float(auc_oth))
        auc_log_uda_val.append(float(auc_val))

        mr = float(np.mean(ep_mask_rate)) if ep_mask_rate else 0.0
        pr = float(np.mean(ep_pos_rate)) if ep_pos_rate else 0.0
        print(f"[REP {repeat_id:02d}] [UDA] epoch {ep} OTH-val AUC={auc_oth:.4f} | TNBC-val AUC={auc_val:.4f} "
              f"(tau~{tau_last:.2f}) warm_r={ramp((gstep-1)/max(1,total_steps-1), cfg['warmup']):.2f} mask={mr:.3f} mask&1={pr:.3f}")

        # save best by OTH-val
        if float(auc_oth) > best_oth_auc:
            best_oth_auc = float(auc_oth)
            best_oth_epoch = ep + 1
            torch.save(model.state_dict(), run_out / "checkpoint_uda_best_othval.pt")

        # save best by TNBC-val (this is the "best_target" for this repeat)
        if (not np.isnan(auc_val)) and float(auc_val) > best_val_auc:
            best_val_auc = float(auc_val)
            best_val_epoch = ep + 1
            torch.save(model.state_dict(), run_out / "checkpoint_uda_best_target.pt")

    # save curves
    if auc_log_uda_oth:
        _save_curve_csv(auc_log_uda_oth, run_out / "uda_auc_othval.csv", "AUC")
        _save_line_plot(auc_log_uda_oth, "AUC", f"UDA OTH-val AUC (rep {repeat_id})", run_out / "uda_auc_othval.png")
    if auc_log_uda_val:
        _save_curve_csv(auc_log_uda_val, run_out / "uda_auc_tnbc_val.csv", "AUC")
        _save_line_plot(auc_log_uda_val, "AUC", f"UDA TNBC-val AUC (rep {repeat_id})", run_out / "uda_auc_tnbc_val.png")

    # =========================
    # Final eval on TNBC test
    # =========================
    ckpt_target = run_out / "checkpoint_uda_best_target.pt"
    ckpt_othval = run_out / "checkpoint_uda_best_othval.pt"

    chosen = "best_target" if ckpt_target.exists() else ("best_othval" if ckpt_othval.exists() else "last")
    if chosen == "best_target":
        model.load_state_dict(safe_torch_load(ckpt_target, device), strict=True)
    elif chosen == "best_othval":
        model.load_state_dict(safe_torch_load(ckpt_othval, device), strict=True)

    auc_oth, p_oth, y_othv = evaluate_auc(model, vaL, device)
    auc_val, p_val, y_val = evaluate_auc(model, tnbcArrLabVal, device)
    auc_test, p_test, y_test = evaluate_auc(model, tnbcArrLabTest, device)

    # save predictions
    pd.DataFrame({"Prob": p_val, "True": y_val}).to_csv(run_out / "pred_tnbc_val.csv", index=False)
    pd.DataFrame({"Prob": p_test, "True": y_test}).to_csv(run_out / "pred_tnbc_test.csv", index=False)
    pd.DataFrame({"Prob": p_oth, "True": y_othv}).to_csv(run_out / "pred_oth_val.csv", index=False)

    _save_roc(y_test, p_test, run_out / "roc_tnbc_test.png")
    _save_roc(y_val, p_val, run_out / "roc_tnbc_val.png")
    _save_roc(y_othv, p_oth, run_out / "roc_oth_val.png")

    m_oth = _bin_metrics(y_othv, p_oth, thr=0.5)
    m_val = _bin_metrics(y_val, p_val, thr=0.5)
    m_test = _bin_metrics(y_test, p_test, thr=0.5)

    metrics = {
        "repeat": int(repeat_id),
        "chosen_ckpt": chosen,
        "best_oth_epoch": int(best_oth_epoch),
        "best_oth_auc": float(best_oth_auc),
        "best_val_epoch": int(best_val_epoch),
        "best_val_auc": float(best_val_auc),
        "OTH_val": m_oth,
        "TNBC_val": m_val,
        "TNBC_test": m_test,
        "tau_last": float(tau_last) if tau_last is not None else None,
        "cfg_snapshot": {
            "lambda_tlab": float(cfg["lambda_tlab"]),
            "lambda_pl": float(cfg["lambda_pl"]),
            "lambda_entropy": float(cfg["lambda_entropy"]),
            "lambda_platform_loss": float(cfg["lambda_platform_loss"]),
            "lambda_cancer_loss": float(cfg["lambda_cancer_loss"]),
            "warmup": float(cfg["warmup"]),
            "z_dim": int(cfg["z_dim"]),
            "dropout": float(cfg["dropout"]),
        }
    }

    with open(run_out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[REP {repeat_id:02d}] FINAL: TNBC-test AUC={auc_test:.4f} | TNBC-val AUC={auc_val:.4f} | OTH-val AUC={auc_oth:.4f} | chosen={chosen}")
    return metrics, enc_state


# ----------------- Main -----------------
def main():
    cfg = CFG
    seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and (not cfg["cpu"]) else "cpu")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    oth = pd.read_csv(cfg["oth_labeled_csv"], index_col=0)
    tnbc_arr_ul = pd.read_csv(cfg["tnbc_array_unlabeled_csv"], index_col=0)

    tnbc_rna_ul = None
    if cfg["tnbc_rnaseq_unlabeled_csv"]:
        tnbc_rna_ul = pd.read_csv(cfg["tnbc_rnaseq_unlabeled_csv"], index_col=0)

    tnbc_arr_lab = pd.read_csv(cfg["tnbc_array_labeled_csv"], index_col=0)

    # ---- sanitize labels ----
    oth = sanitize_labels(oth, cfg["label_col"], "OTH_labeled")
    tnbc_arr_lab = sanitize_labels(tnbc_arr_lab, cfg["label_col"], "TNBC_array_labeled")

    # ---- decide gene list (once) ----
    genes_json_list = read_genes_json(cfg["genes_json"]) if cfg["genes_json"] else None
    if genes_json_list is not None:
        gene_list = [g for g in genes_json_list if g in oth.columns and g in tnbc_arr_ul.columns and g in tnbc_arr_lab.columns]
        if tnbc_rna_ul is not None:
            gene_list = [g for g in gene_list if g in tnbc_rna_ul.columns]
        print(f"[GENES] from genes.json -> after intersection = {len(gene_list)}")
    else:
        gene_list = [g for g in oth.columns if g != cfg["label_col"] and g in tnbc_arr_ul.columns and g in tnbc_arr_lab.columns]
        if tnbc_rna_ul is not None:
            gene_list = [g for g in gene_list if g in tnbc_rna_ul.columns]
        print(f"[GENES] intersection from csvs = {len(gene_list)}")

    assert len(gene_list) > 0, "No intersected genes found."

    # align all to same gene_list
    oth = align_to_genes(oth, gene_list, label_col=cfg["label_col"])
    tnbc_arr_ul = align_to_genes(tnbc_arr_ul, gene_list, label_col=None)
    if tnbc_rna_ul is not None:
        tnbc_rna_ul = align_to_genes(tnbc_rna_ul, gene_list, label_col=None)
    tnbc_arr_lab = align_to_genes(tnbc_arr_lab, gene_list, label_col=cfg["label_col"])

    print(f"[DATA] genes={len(gene_list)} | OTH={len(oth)} | TNBC_arr_ul={len(tnbc_arr_ul)} | "
          f"TNBC_rna_ul={(len(tnbc_rna_ul) if tnbc_rna_ul is not None else 0)} | TNBC_arr_lab={len(tnbc_arr_lab)}")

    # save genes.json for this whole repeated run
    with open(out_dir / "genes.json", "w", encoding="utf-8") as f:
        json.dump({"genes": gene_list}, f, ensure_ascii=False, indent=2)

    # Optionally: pretrain encoder once on "full pool" and reuse for all repeats (fast but less strict)
    cached_encoder = None
    if cfg["reuse_pretrain_encoder"]:
        # If strict_no_peek=True, we still pretrain inside each repeat using only train portion
        # so reuse_pretrain_encoder will be ignored unless strict_no_peek is False.
        if not cfg["strict_no_peek"]:
            print("[Stage0] Pretraining one shared encoder for all repeats (strict_no_peek=False).")
            pool = []
            pool.append(oth[gene_list].values.astype(np.float32))
            pool.append(tnbc_arr_ul[gene_list].values.astype(np.float32))
            if tnbc_rna_ul is not None:
                pool.append(tnbc_rna_ul[gene_list].values.astype(np.float32))
            # include all TNBC labeled (WITHOUT labels)
            pool.append(tnbc_arr_lab[gene_list].values.astype(np.float32))
            X_pool = np.concatenate(pool, axis=0)

            ae = DenoisingAE(len(gene_list), cfg["z_dim"], cfg["dropout"]).to(device)
            opt_ae = torch.optim.Adam(ae.parameters(), lr=cfg["lr_pre"])
            dl_pool = DataLoader(TensorOnlyDS(X_pool), batch_size=cfg["batch"], shuffle=True, drop_last=True)

            pretrain_mse_log = []
            for ep in range(cfg["pretrain_epochs"]):
                losses = []
                ae.train()
                for xb in dl_pool:
                    xb = xb.to(device)
                    mask = (torch.rand_like(xb) > cfg["mask_ratio"]).float()
                    noisy = xb * mask + torch.randn_like(xb) * cfg["noise_std"]
                    _, xhat = ae(noisy)
                    loss = F.mse_loss(xhat, xb)
                    opt_ae.zero_grad()
                    loss.backward()
                    opt_ae.step()
                    losses.append(loss.item())
                pretrain_mse_log.append(float(np.mean(losses)) if losses else float("nan"))
                if (ep % max(1, cfg["pretrain_log_every"])) == 0:
                    print(f"[Shared Pretrain] epoch {ep} reconMSE={pretrain_mse_log[-1]:.4f}")

            cached_encoder = ae.enc.state_dict()
            torch.save(cached_encoder, out_dir / "encoder_pretrained_shared.pt")
            _save_curve_csv(pretrain_mse_log, out_dir / "pretrain_shared_reconMSE.csv", "reconMSE")
            _save_line_plot(pretrain_mse_log, "recon MSE", "Shared Pretrain Reconstruction MSE", out_dir / "pretrain_shared_reconMSE.png")
            del ae
        else:
            print("[INFO] strict_no_peek=True -> encoder will be pretrained per repeat using TNBC-train only (no reuse).")
            cached_encoder = None

    # ---------------- Repeated holdout loop ----------------
    all_metrics = []
    per_rep_test_auc = []
    per_rep_val_auc = []
    per_rep_oth_auc = []

    for r in range(cfg["repeats"]):
        seed_all(cfg["seed"] + r * 999)
        metrics, enc_state = run_one_repeat(
            cfg=cfg,
            repeat_id=r,
            out_dir=out_dir,
            oth=oth,
            tnbc_arr_ul=tnbc_arr_ul,
            tnbc_rna_ul=tnbc_rna_ul,
            tnbc_arr_lab=tnbc_arr_lab,
            gene_list=gene_list,
            device=device,
            cached_pretrained_encoder=cached_encoder
        )
        all_metrics.append(metrics)
        per_rep_test_auc.append(metrics["TNBC_test"]["AUC"])
        per_rep_val_auc.append(metrics["TNBC_val"]["AUC"])
        per_rep_oth_auc.append(metrics["OTH_val"]["AUC"])

    # ---------------- Aggregate ----------------
    df = pd.DataFrame([{
        "repeat": m["repeat"],
        "chosen": m["chosen_ckpt"],
        "tnbc_val_auc": m["TNBC_val"]["AUC"],
        "tnbc_test_auc": m["TNBC_test"]["AUC"],
        "oth_val_auc": m["OTH_val"]["AUC"],
        "best_val_epoch": m["best_val_epoch"],
        "best_val_auc": m["best_val_auc"],
        "best_oth_epoch": m["best_oth_epoch"],
        "best_oth_auc": m["best_oth_auc"],
    } for m in all_metrics])
    df.to_csv(out_dir / "repeated_holdout_each_repeat.csv", index=False)

    test_mean = float(np.nanmean(per_rep_test_auc))
    test_std  = float(np.nanstd(per_rep_test_auc, ddof=1)) if len(per_rep_test_auc) > 1 else 0.0
    test_med  = float(np.nanmedian(per_rep_test_auc))
    ci_low, ci_high = t_confidence_interval(per_rep_test_auc, alpha=0.05)

    summary = {
        "repeats": int(cfg["repeats"]),
        "TNBC_test_auc": {
            "mean": test_mean,
            "std": test_std,
            "median": test_med,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "all": [float(x) for x in per_rep_test_auc],
        },
        "TNBC_val_auc": {
            "mean": float(np.nanmean(per_rep_val_auc)),
            "std": float(np.nanstd(per_rep_val_auc, ddof=1)) if len(per_rep_val_auc) > 1 else 0.0,
            "all": [float(x) for x in per_rep_val_auc],
        },
        "OTH_val_auc": {
            "mean": float(np.nanmean(per_rep_oth_auc)),
            "std": float(np.nanstd(per_rep_oth_auc, ddof=1)) if len(per_rep_oth_auc) > 1 else 0.0,
            "all": [float(x) for x in per_rep_oth_auc],
        },
        "splits": {
            "tnbc_test_ratio": float(cfg["tnbc_test_ratio"]),
            "tnbc_val_ratio_within_trainval": float(cfg["tnbc_val_ratio_within_trainval"]),
        },
        "notes": {
            "strict_no_peek": bool(cfg["strict_no_peek"]),
            "reuse_pretrain_encoder": bool(cfg["reuse_pretrain_encoder"]),
        }
    }

    with open(out_dir / "repeated_holdout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # A small one-line CSV for quick view
    pd.DataFrame([{
        "repeats": cfg["repeats"],
        "tnbc_test_auc_mean": test_mean,
        "tnbc_test_auc_std": test_std,
        "tnbc_test_auc_median": test_med,
        "tnbc_test_auc_ci95_low": ci_low,
        "tnbc_test_auc_ci95_high": ci_high,
        "oth_val_auc_mean": float(np.nanmean(per_rep_oth_auc)),
    }]).to_csv(out_dir / "repeated_holdout_summary.csv", index=False)

    print("\n[SUMMARY] Repeated Holdout DONE.")
    print(f"  TNBC-test AUC: mean={test_mean:.4f} std={test_std:.4f} median={test_med:.4f} 95%CI=[{ci_low:.4f},{ci_high:.4f}]")
    print(f"  saved: {out_dir / 'repeated_holdout_each_repeat.csv'}")
    print(f"  saved: {out_dir / 'repeated_holdout_summary.json'}")
    print(f"  saved: {out_dir / 'repeated_holdout_summary.csv'}")


# =========================
# ✅ 只改这里！点 Run！
# =========================
CFG = dict(
    # ---- Paths (sample×gene; labeled has Response col) ----
    oth_labeled_csv=r"/grl/data/OTH_labeled_std.csv",
    tnbc_array_unlabeled_csv=r"/grl/data/BRA_unlabeled_std.csv",
    tnbc_rnaseq_unlabeled_csv=r"/grl/data/TNBC_ULR_std.csv",  # can be None/"" to disable
    tnbc_array_labeled_csv=r"/grl/data/TNBC_labeled_std.csv",
    label_col="Response",

    # genes list (recommended)
    genes_json=r"/grl/data/genes.json",

    # outputs
    out_dir=r"/grl/out_repeated_holdout_v3",

    # device
    cpu=False,

    # repeated holdout settings
    repeats=10,  # e.g. 10
    tnbc_test_ratio=0.20,                 # TNBC labeled -> test ratio
    tnbc_val_ratio_within_trainval=0.20,  # remaining train_val -> val ratio

    # strictness & speed
    strict_no_peek=True,            # True: AE pool does NOT include TNBC val/test (recommended)
    reuse_pretrain_encoder=False,   # True only meaningful if strict_no_peek=False (fast but less strict)
    use_tnbc_train_in_ae_pool=True, # If strict_no_peek=True, only TNBC_train enters AE pool

    # model/training
    z_dim=256,
    dropout=0.1,
    batch=32,
    val_ratio=0.2,      # OTH stratified split
    seed=42,

    # regularization
    weight_decay=1e-4,
    grad_clip=5.0,

    # stage0 (DAE)
    pretrain_epochs=30,      # suggest smaller for repeated runs
    pretrain_log_every=5,
    lr_pre=1e-3,
    mask_ratio=0.1,
    noise_std=0.05,

    # stage1
    src_head_epochs=20,
    src_ft_epochs=35,
    lr=1.5e-4,
    patience=5,
    focal_gamma=2.0,
    focal_alpha=0.5,
    lambda_center=0.05,
    lambda_ortho=0.02,

    # stage2 (UDA)
    uda_epochs=70,       # suggest smaller first; scale up if stable
    steps_per_epoch=0,   # 0=auto

    # warmup
    warmup=0.10,

    # GRL strengths (scheduled by warmup)
    lambda_platform=0.12,
    lambda_cancer=0.04,

    # loss weights (scheduled by warmup)
    lambda_platform_loss=0.7,
    lambda_cancer_loss=0.25,

    # SSDA anchor
    lambda_tlab=3.0,

    # pseudo-label/entropy on microarray target (scheduled by warmup)
    lambda_entropy=0.01,
    lambda_pl=0.2,
    pl_tau_start=0.9,
    pl_tau_end=0.75,
    pl_max_per_class=12,
    pl_min_per_class=2,
    ent_on_conf=True,

    # de-collinearity switches
    platform_within_tnbc=True,
    cancer_within_rnaseq=True,
    fallback_use_oth_for_platform_if_no_tnbc_rna=True,
    disable_cancer_if_no_tnbc_rna=True,
)

if __name__ == "__main__":
    main()
