
# ============================================================
# Chapter 4.2 redo: GO/KEGG ORA + GSEA + ssGSEA pipeline
# Input files are generated from the uploaded tables.
# Recommended packages:
#   install.packages(c("tidyverse", "patchwork", "pheatmap"))
#   if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
   BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "enrichplot", "fgsea", "GSVA"))
# ============================================================

R.version.string
BiocManager::version()

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

options(pkgType = "binary")
options(repos = BiocManager::repositories())

BiocManager::install(c("rhdf5", "HDF5Array", "DelayedArray", "GSVA"),
                     ask = FALSE, update = FALSE)


library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(fgsea)
library(GSVA)
library(pheatmap)
library(patchwork)

# -----------------------------
# 0. Read inputs
# -----------------------------
base_dir <- "ch4_redo"

top200 <- read_csv(file.path(base_dir, "top200_main_candidates.csv"), show_col_types = FALSE)
top100 <- read_csv(file.path(base_dir, "top100_focus_candidates.csv"), show_col_types = FALSE)
top50  <- read_csv(file.path(base_dir, "top50_display_candidates.csv"), show_col_types = FALSE)
bg800  <- read_csv(file.path(base_dir, "MAD_HVG800_background.csv"), show_col_types = FALSE)
ranked <- read_csv(file.path(base_dir, "ranked_genes_for_GSEA_delta_log_importance.csv"), show_col_types = FALSE)
expr   <- read_csv(file.path(base_dir, "TNBC_expression_matrix_rows_genes.csv"), show_col_types = FALSE)
meta   <- read_csv(file.path(base_dir, "TNBC_sample_metadata.csv"), show_col_types = FALSE)

dir.create(file.path(base_dir, "results"), showWarnings = FALSE)
dir.create(file.path(base_dir, "figures"), showWarnings = FALSE)

# -----------------------------
# 1. Gene ID conversion
# -----------------------------
convert_symbols <- function(symbols) {
  bitr(symbols,
       fromType = "SYMBOL",
       toType   = c("ENTREZID", "SYMBOL"),
       OrgDb    = org.Hs.eg.db) %>%
    distinct(SYMBOL, .keep_all = TRUE)
}

bg_map     <- convert_symbols(bg800$gene)
top200_map <- convert_symbols(top200$gene)
top100_map <- convert_symbols(top100$gene)

write_csv(bg_map,     file.path(base_dir, "results", "background_800_idmap.csv"))
write_csv(top200_map, file.path(base_dir, "results", "top200_idmap.csv"))
write_csv(top100_map, file.path(base_dir, "results", "top100_idmap.csv"))

bg_entrez     <- unique(bg_map$ENTREZID)
top200_entrez <- unique(top200_map$ENTREZID)
top100_entrez <- unique(top100_map$ENTREZID)

# -----------------------------
# 2. ORA: GO-BP and KEGG
# Main analysis = Top200
# Robustness supplement = Top100
# Background = MAD-HVG800
# -----------------------------
ego_top200 <- enrichGO(
  gene          = top200_entrez,
  universe      = bg_entrez,
  OrgDb         = org.Hs.eg.db,
  keyType       = "ENTREZID",
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.20,
  readable      = TRUE
)

ekegg_top200 <- enrichKEGG(
  gene          = top200_entrez,
  universe      = bg_entrez,
  organism      = "hsa",
  keyType       = "ncbi-geneid",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.20
)
ekegg_top200 <- setReadable(ekegg_top200, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")

ego_top100 <- enrichGO(
  gene          = top100_entrez,
  universe      = bg_entrez,
  OrgDb         = org.Hs.eg.db,
  keyType       = "ENTREZID",
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.20,
  readable      = TRUE
)

ekegg_top100 <- enrichKEGG(
  gene          = top100_entrez,
  universe      = bg_entrez,
  organism      = "hsa",
  keyType       = "ncbi-geneid",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.20
)
ekegg_top100 <- setReadable(ekegg_top100, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")

write_csv(as.data.frame(ego_top200),   file.path(base_dir, "results", "ORA_GO_BP_top200.csv"))
write_csv(as.data.frame(ekegg_top200), file.path(base_dir, "results", "ORA_KEGG_top200.csv"))
write_csv(as.data.frame(ego_top100),   file.path(base_dir, "results", "ORA_GO_BP_top100.csv"))
write_csv(as.data.frame(ekegg_top100), file.path(base_dir, "results", "ORA_KEGG_top100.csv"))

pdf(file.path(base_dir, "figures", "Fig4_3_Top200_GO_KEGG_ORA.pdf"), width = 12, height = 6)
print(dotplot(ego_top200, showCategory = 12, font.size = 10) + ggtitle("Top200 GO-BP ORA"))
print(dotplot(ekegg_top200, showCategory = 12, font.size = 10) + ggtitle("Top200 KEGG ORA"))
dev.off()

pdf(file.path(base_dir, "figures", "Fig4_5_Top200_vs_Top100_ORA.pdf"), width = 12, height = 8)
print(dotplot(ego_top100, showCategory = 10, font.size = 10) + ggtitle("Top100 GO-BP ORA"))
print(dotplot(ekegg_top100, showCategory = 10, font.size = 10) + ggtitle("Top100 KEGG ORA"))
dev.off()

# -----------------------------
# 3. GSEA: full ranked MAD-HVG800 list
# Ranking metric:
#   delta_log_importance
# Positive NES = enriched toward genes gaining importance after transfer
# Negative NES = enriched toward genes relatively stronger before transfer
# -----------------------------
rank_map <- convert_symbols(ranked$gene)

rank_df <- ranked %>%
  inner_join(rank_map, by = c("gene" = "SYMBOL")) %>%
  distinct(gene, .keep_all = TRUE) %>%
  arrange(desc(delta_log_importance))

geneList <- rank_df$delta_log_importance
names(geneList) <- rank_df$ENTREZID
geneList <- sort(geneList, decreasing = TRUE)

gsea_go <- gseGO(
  geneList      = geneList,
  OrgDb         = org.Hs.eg.db,
  keyType       = "ENTREZID",
  ont           = "BP",
  minGSSize     = 10,
  maxGSSize     = 500,
  pvalueCutoff  = 0.05,
  pAdjustMethod = "BH",
  verbose       = FALSE
)

gsea_kegg <- gseKEGG(
  geneList      = geneList,
  organism      = "hsa",
  keyType       = "ncbi-geneid",
  minGSSize     = 10,
  maxGSSize     = 500,
  pvalueCutoff  = 0.05,
  pAdjustMethod = "BH",
  verbose       = FALSE
)

write_csv(as.data.frame(gsea_go),   file.path(base_dir, "results", "GSEA_GO_BP_fullrank.csv"))
write_csv(as.data.frame(gsea_kegg), file.path(base_dir, "results", "GSEA_KEGG_fullrank.csv"))

# summary plot for top positive / negative NES pathways
make_gsea_bar <- function(df, title_text, n_each = 8) {
  if (nrow(df) == 0) return(NULL)
  pos <- df %>% filter(NES > 0) %>% arrange(p.adjust, desc(NES)) %>% head(n_each)
  neg <- df %>% filter(NES < 0) %>% arrange(p.adjust, NES) %>% head(n_each)
  plot_df <- bind_rows(pos, neg) %>%
    mutate(Description = forcats::fct_reorder(Description, NES))
  ggplot(plot_df, aes(x = Description, y = NES)) +
    geom_col() +
    coord_flip() +
    labs(x = NULL, y = "NES", title = title_text) +
    theme_bw(base_size = 11)
}

gsea_go_df   <- as.data.frame(gsea_go)
gsea_kegg_df <- as.data.frame(gsea_kegg)

pdf(file.path(base_dir, "figures", "Fig4_4_GSEA_GO_KEGG_summary.pdf"), width = 12, height = 8)
print(make_gsea_bar(gsea_go_df,   "GSEA GO-BP (full ranked MAD-HVG800)"))
print(make_gsea_bar(gsea_kegg_df, "GSEA KEGG (full ranked MAD-HVG800)"))
dev.off()

# -----------------------------
# 4. ssGSEA / GSVA score for Top200 gene set
# This supports Chapter 4.3.
# -----------------------------
expr_mat <- expr %>%
  distinct(Gene, .keep_all = TRUE) %>%
  as.data.frame()
rownames(expr_mat) <- expr_mat$Gene
expr_mat$Gene <- NULL
expr_mat <- as.matrix(expr_mat)

gene_sets <- list(
  Top200_signature = intersect(top200$gene, rownames(expr_mat)),
  Top100_signature = intersect(top100$gene, rownames(expr_mat)),
  Top50_signature  = intersect(top50$gene,  rownames(expr_mat))
)

ssgsea_scores <- gsva(
  expr = expr_mat,
  gset.idx.list = gene_sets,
  method = "ssgsea",
  kcdf = "Gaussian",
  abs.ranking = TRUE
)

score_df <- as.data.frame(t(ssgsea_scores)) %>%
  rownames_to_column("SampleID") %>%
  left_join(meta, by = "SampleID")

write_csv(score_df, file.path(base_dir, "results", "ssGSEA_scores_Top200_Top100_Top50.csv"))

wilcox_top200 <- wilcox.test(Top200_signature ~ Response, data = score_df)
wilcox_top100 <- wilcox.test(Top100_signature ~ Response, data = score_df)
wilcox_top50  <- wilcox.test(Top50_signature  ~ Response, data = score_df)

capture.output(
  list(
    Top200 = wilcox_top200,
    Top100 = wilcox_top100,
    Top50  = wilcox_top50
  ),
  file = file.path(base_dir, "results", "ssGSEA_wilcox_tests.txt")
)

pdf(file.path(base_dir, "figures", "Fig4_6_ssGSEA_top_signatures_boxplot.pdf"), width = 10, height = 4)
score_long <- score_df %>%
  pivot_longer(cols = c(Top200_signature, Top100_signature, Top50_signature),
               names_to = "Signature", values_to = "ssGSEA")
ggplot(score_long, aes(x = factor(Response), y = ssGSEA)) +
  geom_boxplot() +
  facet_wrap(~Signature, scales = "free_y") +
  labs(x = "Response (0 = NR, 1 = R)", y = "ssGSEA score") +
  theme_bw(base_size = 11)
dev.off()

# -----------------------------
# 5. Suggested manuscript positioning
# -----------------------------
cat("
Main analysis:
  - ORA on Top200 against MAD-HVG800 background
  - GSEA on full ranked MAD-HVG800 using delta_log_importance
  - ssGSEA on Top200 signature in TNBC_AL

Supplement:
  - Top100 ORA as convergence/robustness support
  - Top50 only as representative display set, not a main enrichment set
")
