import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# =========================================================
# 0. 文件路径
# =========================================================
RAW_H5AD = "GSE118389_raw_counts.h5ad"
ANNOT_H5AD = "GSE118389_celltype_annotated.h5ad"
MODULE_CSV = "custom_route1_modules.csv"

# =========================================================
# 1. 读取对象
# =========================================================
adata_raw = sc.read_h5ad(RAW_H5AD)                 # 全基因原始计数对象
adata_annot = sc.read_h5ad(ANNOT_H5AD)             # 已有 leiden / celltype / UMAP 的对象

print("raw:", adata_raw)
print("annot:", adata_annot)

# 按 cell 顺序对齐
common_cells = [c for c in adata_annot.obs_names if c in adata_raw.obs_names]
adata_raw = adata_raw[common_cells].copy()
adata_annot = adata_annot[common_cells].copy()

print("aligned raw:", adata_raw)
print("aligned annot:", adata_annot)

# 把注释信息和 UMAP 坐标拷到全基因对象
for col in ["patient", "leiden", "celltype"]:
    if col in adata_annot.obs.columns:
        adata_raw.obs[col] = adata_annot.obs[col].astype(str)

if "X_umap" in adata_annot.obsm:
    adata_raw.obsm["X_umap"] = adata_annot.obsm["X_umap"].copy()
else:
    raise ValueError("annot 对象里没有 X_umap，请确认 step2/step4 是否正确保存。")

# =========================================================
# 2. 在全基因对象上做 normalize + log1p
# =========================================================
adata = adata_raw.copy()
adata.var_names_make_unique()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

print("normalized full-gene adata:", adata)

# =========================================================
# 3. 构建 major_celltype（用于后面更稳地讲故事）
# =========================================================
major_map = {
    "cytotoxic_lymphoid_NK_like": "Lymphoid",
    "lymphoid_immune": "Lymphoid",
    "myeloid_macrophage": "Myeloid",
    "plasma_plasmablast_like": "B_Plasma",
    "fibroblast_CAF_like_1": "Fibro_CAF",
    "fibroblast_matrix_CAF_like_2": "Fibro_CAF",
    "stromal_inflammatory_fibroblast_like": "Fibro_CAF",
    "endothelial": "Endothelial",
    "mural_pericyte_like": "Pericyte_Mural",
    "myoepithelial_smooth_muscle_like": "Pericyte_Mural",
    "perivascular_mural_like": "Pericyte_Mural",
    "epithelial_tumor_like_1": "Epithelial_Tumor",
    "epithelial_tumor_like_2": "Epithelial_Tumor",
    "epithelial_tumor_like_3": "Epithelial_Tumor",
    "secretory_epithelial_tumor_like": "Epithelial_Tumor",
    "secretory_glandular_epithelial_like": "Epithelial_Tumor",
}

adata.obs["major_celltype"] = adata.obs["celltype"].map(
    lambda x: major_map.get(str(x), str(x))
)

major_order = [
    "Epithelial_Tumor",
    "Lymphoid",
    "Myeloid",
    "B_Plasma",
    "Fibro_CAF",
    "Endothelial",
    "Pericyte_Mural",
]
adata.obs["major_celltype"] = pd.Categorical(
    adata.obs["major_celltype"],
    categories=[x for x in major_order if x in adata.obs["major_celltype"].unique()],
    ordered=True
)

# =========================================================
# 4. 画图参数
# =========================================================
sc.set_figure_params(
    dpi=300,
    dpi_save=300,
    facecolor="white",
    frameon=False,
    vector_friendly=True
)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["font.size"] = 12

n_cells = adata.n_obs
if n_cells <= 1500:
    point_size = 10
elif n_cells <= 5000:
    point_size = 6
else:
    point_size = 3

print("point_size =", point_size)

# =========================================================
# 5. 通用 UMAP 重绘函数
# =========================================================
def draw_umap(adata, color_key, out_png, out_pdf=None, legend_loc="right margin"):
    if color_key not in adata.obs.columns:
        print(f"[skip] {color_key} not found")
        return

    if adata.obs[color_key].dtype.name != "category":
        adata.obs[color_key] = adata.obs[color_key].astype("category")

    sc.pl.umap(
        adata,
        color=color_key,
        size=point_size,
        alpha=0.9,
        frameon=False,
        legend_loc=legend_loc,
        legend_fontsize=10,
        legend_fontoutline=2,
        title="",
        show=False
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")
    if out_pdf:
        print(f"[saved] {out_pdf}")

# =========================================================
# 6. 重绘 celltype / major_celltype UMAP
# =========================================================
draw_umap(
    adata,
    color_key="celltype",
    out_png="umap_by_celltype_pretty.png",
    out_pdf="umap_by_celltype_pretty.pdf",
    legend_loc="right margin"
)

draw_umap(
    adata,
    color_key="major_celltype",
    out_png="umap_by_major_celltype_pretty.png",
    out_pdf="umap_by_major_celltype_pretty.pdf",
    legend_loc="right margin"
)

# =========================================================
# 7. 定义候选基因（主名单 12 个；主展示 8 个）
# =========================================================
priority_genes_12 = [
    "CLTA", "COPB2", "CHMP5", "IGF2R",
    "ALG6", "GPN1", "DNAJC2", "DERL1",
    "CX3CL1", "COL4A1", "CD93",
    "ATP6V0C"
]

focus_genes_8 = [
    "CLTA", "COPB2", "CHMP5", "IGF2R",
    "ALG6", "DERL1",
    "CX3CL1", "COL4A1"
]

priority_genes_12 = [g for g in priority_genes_12 if g in adata.var_names]
focus_genes_8 = [g for g in focus_genes_8 if g in adata.var_names]

print("priority_genes_12 found:", priority_genes_12)
print("focus_genes_8 found:", focus_genes_8)

# =========================================================
# 8. UMAP 上画重点候选基因
# =========================================================
if len(focus_genes_8) > 0:
    sc.pl.umap(
        adata,
        color=focus_genes_8,
        ncols=4,
        size=point_size,
        frameon=False,
        cmap="Reds",
        use_raw=False,
        show=False
    )
    plt.savefig("umap_featureplots_priority_genes.png", dpi=300, bbox_inches="tight")
    plt.savefig("umap_featureplots_priority_genes.pdf", bbox_inches="tight")
    plt.close()
    print("[saved] umap_featureplots_priority_genes.png/pdf")

# =========================================================
# 9. 候选基因在 major_celltype 上做 DotPlot
# =========================================================
if len(priority_genes_12) > 0:
    dp = sc.pl.dotplot(
        adata,
        var_names=priority_genes_12,
        groupby="major_celltype",
        standard_scale="var",
        figsize=(10, 5),
        show=False,
        return_fig=True,
        use_raw=False
    )
    dp.savefig("dotplot_priority_genes_by_major_celltype.png", dpi=300)
    dp.savefig("dotplot_priority_genes_by_major_celltype.pdf")
    plt.close("all")
    print("[saved] dotplot_priority_genes_by_major_celltype.png/pdf")

# =========================================================
# 10. 导出候选基因在 major_celltype 中的平均表达
# =========================================================
def to_dense(x):
    return x.toarray() if sparse.issparse(x) else x

if len(priority_genes_12) > 0:
    X = pd.DataFrame(
        to_dense(adata[:, priority_genes_12].X),
        index=adata.obs_names,
        columns=priority_genes_12
    )
    gene_mean_major = X.groupby(adata.obs["major_celltype"], observed=True).mean()
    gene_mean_major.to_csv("candidate_gene_mean_by_major_celltype.csv")
    print("[saved] candidate_gene_mean_by_major_celltype.csv")

# =========================================================
# 11. 读取 4 个模块，并计算 module score
# =========================================================
mod_df = pd.read_csv(MODULE_CSV)

module_alias = {
    "MODULE_MEMBRANE_VESICLE_TRAFFICKING": "score_membrane_vesicle",
    "MODULE_PROTEIN_PROCESSING_ER_GOLGI_GLYCOSYLATION": "score_er_golgi",
    "MODULE_AUTOPHAGY_STRESS_METABOLIC_ADAPTATION": "score_autophagy_stress",
    "MODULE_MICROENVIRONMENT_ECM_ENDOTHELIAL_INTERACTION": "score_microenv_ecm",
}

module_to_genes = {}
for module_name, subdf in mod_df.groupby("Module"):
    genes = [g for g in subdf["Gene"].astype(str).tolist() if g in adata.var_names]
    module_to_genes[module_name] = genes
    print(module_name, "=>", len(genes), "genes found")

for module_name, genes in module_to_genes.items():
    if len(genes) == 0:
        print(f"[skip] {module_name}: no genes found")
        continue

    score_name = module_alias.get(module_name, module_name.lower())
    sc.tl.score_genes(
        adata,
        gene_list=genes,
        score_name=score_name,
        use_raw=False
    )
    print(f"[done] module score: {score_name}")

score_cols = [v for v in module_alias.values() if v in adata.obs.columns]
print("score_cols:", score_cols)

# =========================================================
# 12. UMAP 上画模块 score
# =========================================================
if len(score_cols) > 0:
    sc.pl.umap(
        adata,
        color=score_cols,
        ncols=2,
        size=point_size,
        frameon=False,
        cmap="Reds",
        show=False
    )
    plt.savefig("umap_module_scores.png", dpi=300, bbox_inches="tight")
    plt.savefig("umap_module_scores.pdf", bbox_inches="tight")
    plt.close()
    print("[saved] umap_module_scores.png/pdf")

# =========================================================
# 13. 模块 score 在 major_celltype 上的均值表
# =========================================================
if len(score_cols) > 0:
    module_mean_major = adata.obs.groupby("major_celltype", observed=True)[score_cols].mean()
    module_mean_major.to_csv("module_score_mean_by_major_celltype.csv")
    print("[saved] module_score_mean_by_major_celltype.csv")

    # 做 z-score 热图（每个模块在不同 celltype 间标准化）
    heat = module_mean_major.T.copy()

    def zscore_row(row):
        std = row.std(ddof=0)
        if std == 0:
            return row * 0
        return (row - row.mean()) / std

    heat_z = heat.apply(zscore_row, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heat_z.values, aspect="auto", cmap="RdBu_r")

    ax.set_xticks(np.arange(heat_z.shape[1]))
    ax.set_xticklabels(heat_z.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(heat_z.shape[0]))
    ax.set_yticklabels(heat_z.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Z-score across cell types")

    ax.set_title("Module scores by major cell type")
    plt.tight_layout()
    plt.savefig("heatmap_module_scores_by_major_celltype.png", dpi=300, bbox_inches="tight")
    plt.savefig("heatmap_module_scores_by_major_celltype.pdf", bbox_inches="tight")
    plt.close()
    print("[saved] heatmap_module_scores_by_major_celltype.png/pdf")

# =========================================================
# 14. 每个模块再单独画 violin
# =========================================================
for col in score_cols:
    sc.pl.violin(
        adata,
        keys=col,
        groupby="major_celltype",
        rotation=45,
        stripplot=False,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"violin_{col}_by_major_celltype.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"violin_{col}_by_major_celltype.pdf", bbox_inches="tight")
    plt.close()
    print(f"[saved] violin_{col}_by_major_celltype.png/pdf")

# =========================================================
# 15. 保存整合好的对象
# =========================================================
adata.write("GSE118389_fullgenes_celltype_modules.h5ad")
print("[saved] GSE118389_fullgenes_celltype_modules.h5ad")

print("done.")