# =========================
# 0. 加载包
# =========================
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("estimate", quietly = TRUE)) {
  install.packages("estimate", repos = "http://R-Forge.R-project.org")
}
if (!requireNamespace("ggpubr", quietly = TRUE)) install.packages("ggpubr")

# MCP-counter 只需首次安装一次
# remotes::install_github("ebecht/MCPcounter", subdir = "Source")

library(MCPcounter)
library(estimate)
library(dplyr)
library(ggplot2)
library(ggpubr)

# =========================
# 0.1 统一字号设置：五号 = 10.5 pt
# =========================
pt5 <- 10.5
text_size <- pt5 / ggplot2::.pt

theme_thesis <- theme_bw(base_size = pt5) +
  theme(
    text = element_text(size = pt5, colour = "black"),
    plot.title = element_text(size = pt5, hjust = 0.5, colour = "black"),
    plot.subtitle = element_text(size = pt5, hjust = 0.5, colour = "black"),
    axis.title.x = element_text(size = pt5, colour = "black"),
    axis.title.y = element_text(size = pt5, colour = "black"),
    axis.text.x = element_text(size = pt5, colour = "black"),
    axis.text.y = element_text(size = pt5, colour = "black"),
    legend.title = element_text(size = pt5, colour = "black"),
    legend.text = element_text(size = pt5, colour = "black"),
    strip.text = element_text(size = pt5, colour = "black"),
    panel.grid = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.6),
    axis.ticks = element_line(colour = "black", linewidth = 0.4),
    legend.key = element_blank()
  )

# =========================
# 1. 读取你的三个文件
# =========================
pred <- read.csv("crossfit_oos_pred.csv", check.names = FALSE)
meta <- read.csv("TNBC_meta.csv", check.names = FALSE)
expr_df <- read.csv("TNBC_rows.csv", check.names = FALSE)

# 看一下
head(pred)
head(meta)
head(expr_df[, 1:6])

# =========================
# 2. 处理模型输出表
# =========================
# crossfit_oos_pred.csv 里没有 SampleID
# 这里默认它和 TNBC_meta.csv 行顺序完全一致
stopifnot(nrow(pred) == nrow(meta))
stopifnot(all(pred$y == meta$Response))

model_scores <- data.frame(
  SampleID = meta$SampleID,
  Response = meta$Response,
  pred_prob = pred$p_oos_mean
)

head(model_scores)

# =========================
# 3. 处理表达矩阵
# =========================
# 第一列是 Gene，后面每列是样本
expr_mat <- as.matrix(expr_df[, -1])
rownames(expr_mat) <- expr_df$Gene
mode(expr_mat) <- "numeric"

# 去掉重复基因（如果有）
expr_mat <- expr_mat[!duplicated(rownames(expr_mat)), ]

# 表达矩阵列名
head(colnames(expr_mat))

# 确保表达矩阵样本顺序和 meta 一致
common_samples <- intersect(meta$SampleID, colnames(expr_mat))
length(common_samples)

meta_sub <- meta %>% filter(SampleID %in% common_samples)
model_scores_sub <- model_scores %>% filter(SampleID %in% common_samples)

# 按 meta 顺序重排表达矩阵列
expr_mat <- expr_mat[, meta_sub$SampleID, drop = FALSE]

# 再确认一次
stopifnot(all(colnames(expr_mat) == meta_sub$SampleID))
stopifnot(all(model_scores_sub$SampleID == meta_sub$SampleID))

# =========================
# 4. 跑 MCP-counter
# =========================
mcp_res <- MCPcounter.estimate(
  expression = expr_mat,
  featuresType = "HUGO_symbols"
)

mcp_df <- as.data.frame(t(mcp_res))
mcp_df$SampleID <- rownames(mcp_df)
rownames(mcp_df) <- NULL

# 只保留你要的两个
mcp_keep <- mcp_df %>%
  select(SampleID, `Endothelial cells`, Fibroblasts)

head(mcp_keep)

# =========================
# 5. 跑 ESTIMATE
# =========================
estimate_input <- data.frame(
  NAME = rownames(expr_mat),
  Description = rownames(expr_mat),
  expr_mat,
  check.names = FALSE
)

write.table(
  estimate_input,
  file = "estimate_input.gct",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

filterCommonGenes(
  input.f = "estimate_input.gct",
  output.f = "estimate_common_genes.gct",
  id = "GeneSymbol"
)

estimateScore(
  input.ds = "estimate_common_genes.gct",
  output.ds = "estimate_scores.gct",
  platform = "agilent"
)

# 读回结果
est_raw <- read.delim("estimate_scores.gct", skip = 2, check.names = FALSE)

score_mat <- est_raw[, -(1:2)]
rownames(score_mat) <- est_raw$NAME

estimate_df <- as.data.frame(t(score_mat))
estimate_df$SampleID <- rownames(estimate_df)
rownames(estimate_df) <- NULL

# 有的版本可能没有 TumorPurity，先看列名
print(colnames(estimate_df))

estimate_keep <- estimate_df %>%
  select(any_of(c("SampleID", "StromalScore", "ImmuneScore", "ESTIMATEScore", "TumorPurity")))

head(estimate_keep)

# =========================
# 6. 合并成总表
# =========================
all_df <- meta_sub %>%
  inner_join(model_scores_sub %>% select(SampleID, pred_prob), by = "SampleID") %>%
  inner_join(mcp_keep, by = "SampleID") %>%
  inner_join(estimate_keep, by = "SampleID")

write.csv(all_df, "44_simple_all_results.csv", row.names = FALSE)

head(all_df)

# =========================
# 7. 组间比较：R vs NR
# =========================
wil_endo <- wilcox.test(`Endothelial cells` ~ Response, data = all_df)
wil_fib  <- wilcox.test(Fibroblasts ~ Response, data = all_df)
wil_str  <- wilcox.test(StromalScore ~ Response, data = all_df)
wil_imm  <- wilcox.test(ImmuneScore ~ Response, data = all_df)

stats_group <- data.frame(
  feature = c("Endothelial cells", "Fibroblasts", "StromalScore", "ImmuneScore"),
  p_value = c(wil_endo$p.value, wil_fib$p.value, wil_str$p.value, wil_imm$p.value)
)

write.csv(stats_group, "44_group_wilcox.csv", row.names = FALSE)
print(stats_group)

# =========================
# 8. 与模型输出分数的相关性（全样本总体相关）
# =========================
cor_endo <- cor.test(all_df$`Endothelial cells`, all_df$pred_prob, method = "spearman")
cor_fib  <- cor.test(all_df$Fibroblasts, all_df$pred_prob, method = "spearman")
cor_str  <- cor.test(all_df$StromalScore, all_df$pred_prob, method = "spearman")
cor_imm  <- cor.test(all_df$ImmuneScore, all_df$pred_prob, method = "spearman")

stats_cor <- data.frame(
  feature = c("Endothelial cells", "Fibroblasts", "StromalScore", "ImmuneScore"),
  rho = c(unname(cor_endo$estimate), unname(cor_fib$estimate),
          unname(cor_str$estimate), unname(cor_imm$estimate)),
  p_value = c(cor_endo$p.value, cor_fib$p.value, cor_str$p.value, cor_imm$p.value)
)

write.csv(stats_cor, "44_model_cor.csv", row.names = FALSE)
print(stats_cor)

# =========================
# 9. 出图
# =========================
all_df$ResponseLabel <- ifelse(all_df$Response == 1, "R", "NR")

# A Endothelial：组间比较
p1 <- ggplot(all_df, aes(x = ResponseLabel, y = `Endothelial cells`, fill = ResponseLabel)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.5) +
  geom_jitter(width = 0.12, alpha = 0.8, size = 1.8) +
  stat_compare_means(method = "wilcox.test", size = text_size) +
  labs(x = "", y = "MCP-counter Endothelial score") +
  theme_thesis

# B Fibroblasts：组间比较
p2 <- ggplot(all_df, aes(x = ResponseLabel, y = Fibroblasts, fill = ResponseLabel)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.5) +
  geom_jitter(width = 0.12, alpha = 0.8, size = 1.8) +
  stat_compare_means(method = "wilcox.test", size = text_size) +
  labs(x = "", y = "MCP-counter Fibroblast score") +
  theme_thesis

# C StromalScore：组间比较
p3 <- ggplot(all_df, aes(x = ResponseLabel, y = StromalScore, fill = ResponseLabel)) +
  geom_boxplot(width = 0.6, outlier.shape = NA, linewidth = 0.5) +
  geom_jitter(width = 0.12, alpha = 0.8, size = 1.8) +
  stat_compare_means(method = "wilcox.test", size = text_size) +
  labs(x = "", y = "ESTIMATE Stromal score") +
  theme_thesis

# D Endothelial vs model score：全样本总体相关
p4 <- ggplot(all_df, aes(x = `Endothelial cells`, y = pred_prob)) +
  geom_point(aes(color = ResponseLabel), size = 2.5) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.6) +
  stat_cor(method = "spearman", color = "black", size = text_size) +
  labs(x = "MCP-counter Endothelial score",
       y = "Model predicted response probability") +
  theme_thesis

# E StromalScore vs model score：全样本总体相关
p5 <- ggplot(all_df, aes(x = StromalScore, y = pred_prob)) +
  geom_point(aes(color = ResponseLabel), size = 2.5) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.6) +
  stat_cor(method = "spearman", color = "black", size = text_size) +
  labs(x = "ESTIMATE Stromal score",
       y = "Model predicted response probability") +
  theme_thesis

# =========================
# 10. 保存图片
# =========================
ggsave("44_p1_endo_group.pdf", p1, width = 4.5, height = 4, units = "in")
ggsave("44_p2_fib_group.pdf", p2, width = 4.5, height = 4, units = "in")
ggsave("44_p3_stromal_group.pdf", p3, width = 4.5, height = 4, units = "in")
ggsave("44_p4_endo_model_overall.pdf", p4, width = 4.5, height = 4, units = "in")
ggsave("44_p5_stromal_model_overall.pdf", p5, width = 4.5, height = 4, units = "in")

ggsave("44_p1_endo_group.png", p1, width = 4.5, height = 4, units = "in", dpi = 600)
ggsave("44_p2_fib_group.png", p2, width = 4.5, height = 4, units = "in", dpi = 600)
ggsave("44_p3_stromal_group.png", p3, width = 4.5, height = 4, units = "in", dpi = 600)
ggsave("44_p4_endo_model_overall.png", p4, width = 4.5, height = 4, units = "in", dpi = 600)
ggsave("44_p5_stromal_model_overall.png", p5, width = 4.5, height = 4, units = "in", dpi = 600)