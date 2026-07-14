import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 读取上一阶段保存的 h5ad
# -----------------------------
adata = sc.read_h5ad("GSE118389_raw_counts.h5ad")

print("loaded:", adata)
print("cells x genes:", adata.n_obs, adata.n_vars)

# 基因名去重（以防后面报错）
adata.var_names_make_unique()

# -----------------------------
# 2. 计算基础 QC 指标
# -----------------------------
# 线粒体基因：人类通常以 MT- 开头
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True
)

print("\nQC summary before filtering:")
print(adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())

# 保存 QC 指标表
adata.obs.to_csv("qc_metrics_before_filter.csv")

# 画 QC 图
sc.pl.violin(
    adata,
    ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
    show=False
)
plt.savefig("qc_violin_before_filter.png", dpi=300, bbox_inches="tight")
plt.close()

sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt", show=False)
plt.savefig("qc_scatter_counts_mt.png", dpi=300, bbox_inches="tight")
plt.close()

sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", show=False)
plt.savefig("qc_scatter_counts_genes.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 3. 基础过滤（先用温和阈值）
# -----------------------------
# 基因：至少在 3 个细胞中表达
sc.pp.filter_genes(adata, min_cells=3)

# 细胞：至少检测到 200 个基因
sc.pp.filter_cells(adata, min_genes=200)

# 去掉线粒体比例过高的细胞
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

print("\nAfter filtering:")
print(adata)
print("cells x genes:", adata.n_obs, adata.n_vars)

# -----------------------------
# 4. 标准化 + log1p
# -----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 保存一份原始归一化表达，后面画图会方便
adata.raw = adata

# -----------------------------
# 5. 高变基因 / PCA / 邻居图 / UMAP / Leiden
# -----------------------------
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
print("HVGs:", int(adata.var["highly_variable"].sum()))

adata_hvg = adata[:, adata.var["highly_variable"]].copy()

sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver="arpack")
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_hvg)
sc.tl.leiden(adata_hvg, resolution=0.5)

print("\nCluster counts:")
print(adata_hvg.obs["leiden"].value_counts().sort_index())

# -----------------------------
# 6. UMAP 可视化
# -----------------------------
sc.pl.umap(adata_hvg, color=["patient"], show=False)
plt.savefig("umap_by_patient.png", dpi=300, bbox_inches="tight")
plt.close()

sc.pl.umap(adata_hvg, color=["leiden"], legend_loc="on data", show=False)
plt.savefig("umap_by_leiden.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 7. 保存结果
# -----------------------------
adata_hvg.write("GSE118389_processed_leiden.h5ad")
print("\nsaved: GSE118389_processed_leiden.h5ad")
print("saved figures: qc_violin_before_filter.png, qc_scatter_counts_mt.png, qc_scatter_counts_genes.png, umap_by_patient.png, umap_by_leiden.png")