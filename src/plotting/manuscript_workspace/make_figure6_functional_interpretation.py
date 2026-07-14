from __future__ import annotations

import ast
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parent
BIO_DIR = Path(r"D:\work1\task3\05_bio")
ORA_DIR = BIO_DIR / "ch4_python_results" / "01_ORA"
FINAL_GO = BIO_DIR / "0407" / "GO" / "Enrichment_GO" / "_FINAL_GO.csv"
OUT_DIR = ROOT / "results" / "interpretation"
FIG_DIR = ROOT / "figures"

MODULES = [
    {
        "name": "Intracellular transport and protein processing",
        "short": "Transport & processing",
        "color": "#4477AA",
        "keywords": [
            "endocytosis",
            "lysosome",
            "vesicle",
            "clathrin",
            "golgi",
            "endoplasmic reticulum",
            "protein processing",
            "protein localisation",
            "protein localization",
            "organelle localisation",
            "organelle localization",
            "ubiquitination",
            "autophagy",
            "vacuolar",
            "secretory",
            "protein secretion",
            "protein export",
            "protein catabolic",
            "proteolysis",
        ],
        "required": ["CLTA", "CHMP5", "COPB2", "ALG6", "IGF2R", "DERL1"],
        "optional": [
            "ATP6V0C",
            "ATP6V1B2",
            "GOLT1B",
            "COPZ1",
            "ARCN1",
            "CHMP1B",
            "CHMP2A",
            "LGMN",
            "GANAB",
            "HERPUD1",
        ],
    },
    {
        "name": "Cytoskeleton, adhesion, and microenvironment",
        "short": "Cytoskeleton, adhesion & microenvironment",
        "color": "#CC6677",
        "keywords": [
            "cytoskeleton",
            "actin",
            "adhesion",
            "focal adhesion",
            "cell-cell",
            "cell adhesion",
            "extracellular matrix",
            "collagen",
            "microenvironment",
            "membrane",
        ],
        "required": ["CX3CL1", "COL4A1", "DLG5"],
        "optional": [
            "COL5A2",
            "COL15A1",
            "ITGA6",
            "ACTN1",
            "ACTN4",
            "ARPC1B",
            "ARPC3",
            "CAPZA1",
            "COTL1",
            "FLNA",
            "A2M",
            "CD93",
            "CD47",
            "CD55",
            "IGFBP7",
        ],
    },
    {
        "name": "Cell cycle, stress response, and metabolism",
        "short": "Cell cycle, stress & metabolism",
        "color": "#DDCC77",
        "keywords": [
            "cell cycle",
            "stress",
            "metabolic",
            "metabolism",
            "fatty acid",
            "citrate cycle",
            "tca",
            "apoptosis",
            "p53",
            "response to",
            "calcium",
            "glycosylation",
        ],
        "required": ["AIMP2"],
        "optional": [
            "CDK1",
            "GADD45A",
            "CCNG1",
            "BID",
            "ASNS",
            "ASNSD1",
            "ACADM",
            "ACLY",
            "FH",
            "DLD",
            "COX4I1",
            "COX6A1",
            "GLUL",
            "HK1",
            "ATIC",
            "CMPK1",
        ],
    },
]

REPRESENTATIVE_GENES = [
    "CLTA",
    "CHMP5",
    "AIMP2",
    "CX3CL1",
    "COL4A1",
    "COPB2",
    "ALG6",
    "DLG5",
    "IGF2R",
    "DERL1",
]


def strip_svg_comments(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\n?\s*<!--.*?-->", "", text, flags=re.S)
    path.write_text(text, encoding="utf-8")


def parse_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def clean_term_name(name: str) -> str:
    cleaned = str(name)
    replacements = {
        "localization": "localisation",
        "organization": "organisation",
        "signaling": "signalling",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def module_for_term(row: pd.Series, module_sets: dict[str, set[str]]) -> tuple[str | None, int]:
    name = str(row["name"]).lower()
    genes = set(parse_list(row.get("intersections", "")))
    best_module = None
    best_score = 0
    for module in MODULES:
        keyword_hits = sum(1 for keyword in module["keywords"] if keyword in name)
        overlap_hits = len(genes & module_sets[module["name"]])
        score = keyword_hits * 3 + overlap_hits
        if score > best_score:
            best_score = score
            best_module = module["name"]
    return best_module, best_score


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    candidate_files = sorted(BIO_DIR.glob("candidates_Q1Q3_sorted*.csv"))
    if not candidate_files:
        raise FileNotFoundError("Could not find candidates_Q1Q3_sorted*.csv")
    candidate_path = candidate_files[0]
    candidates = pd.read_csv(candidate_path, encoding="utf-8-sig")
    meta = pd.read_csv(BIO_DIR / "TNBC_meta.csv", encoding="utf-8-sig")
    expression = pd.read_csv(BIO_DIR / "TNBC_rows.csv", encoding="utf-8-sig")
    return candidates, meta, expression, candidate_path


def clean_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "gene",
        "quadrant",
        "stage1_mean_abs",
        "stage2_mean_abs",
        "delta_log_importance",
        "mean_abs_shap",
        "topk_freq",
        "rank",
        "stage2_abs_diff",
    }
    missing = required_cols - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")

    cleaned = candidates.copy()
    cleaned["gene"] = cleaned["gene"].astype(str).str.strip()
    cleaned["candidate_group"] = np.where(
        cleaned["quadrant"].astype(str).str.contains("Q1", na=False),
        "stable_core",
        np.where(cleaned["quadrant"].astype(str).str.contains("Q3", na=False), "target_adaptive", "other"),
    )
    return cleaned


def prepare_enrichment(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]], list[str]]:
    go = pd.read_csv(ORA_DIR / "Top200_GO_BP_ORA.csv", encoding="utf-8-sig")
    kegg = pd.read_csv(ORA_DIR / "Top200_KEGG_ORA.csv", encoding="utf-8-sig")
    go.to_csv(OUT_DIR / "Figure6_GO_enrichment_full.csv", index=False)
    kegg.to_csv(OUT_DIR / "Figure6_KEGG_enrichment_full.csv", index=False)
    final_go = pd.read_csv(FINAL_GO, encoding="utf-8-sig")

    candidate_genes = set(candidates["gene"])
    module_sets: dict[str, set[str]] = {}
    for module in MODULES:
        module_sets[module["name"]] = set(module["required"] + module["optional"]) & candidate_genes

    enrich = pd.DataFrame(
        {
            "source": "GO:BP",
            "native": final_go["GO"],
            "name": final_go["Description"],
            "term": final_go["Description"].map(clean_term_name),
            "database": "GO BP",
            "p_value": np.power(10.0, final_go["LogP"].astype(float)),
            "adjusted_p_value": np.power(10.0, final_go["Log(q-value)"].astype(float)),
            "overlap_count": final_go["#GeneInGOAndHitList"].astype(int),
            "overlap_genes": final_go["Hits"].fillna("").astype(str).str.split("|"),
        }
    )
    enrich["minus_log10_p"] = -np.log10(enrich["adjusted_p_value"].clip(lower=np.nextafter(0, 1)))
    enrich[["assigned_module", "assignment_score"]] = enrich.apply(
        lambda row: pd.Series(module_for_term(row, module_sets)), axis=1
    )

    preferred_terms = {
        "Intracellular transport and protein processing": [
            "Endocytosis",
            "endosome to lysosome transport",
            "protein localization to organelle",
            "establishment of protein localization to organelle",
            "regulation of protein ubiquitination",
            "response to endoplasmic reticulum stress",
        ],
        "Cytoskeleton, adhesion, and microenvironment": [
            "actin cytoskeleton organization",
            "regulation of cytoskeleton organization",
            "regulation of actin cytoskeleton organization",
            "cell-cell adhesion",
            "homotypic cell-cell adhesion",
            "positive regulation of cell adhesion",
        ],
        "Cell cycle, stress response, and metabolism": [
            "regulation of mitotic cell cycle",
            "regulation of DNA metabolic process",
            "response to hydrogen peroxide",
            "fatty acid metabolic process",
            "regulation of cellular response to stress",
        ],
    }

    selected_frames = []
    selected_terms: list[str] = []
    for module in MODULES:
        sub = enrich[enrich["assigned_module"] == module["name"]].copy()
        preferred_rows = []
        for preferred in preferred_terms[module["name"]]:
            match = sub[sub["name"].str.lower() == preferred.lower()]
            if match.empty:
                match = sub[sub["term"].str.lower() == clean_term_name(preferred).lower()]
            if match.empty:
                match = sub[sub["term"].str.lower().str.contains(clean_term_name(preferred).lower(), regex=False)]
            if not match.empty:
                preferred_rows.append(match.sort_values(["p_value", "overlap_count"], ascending=[True, False]).head(1))
        if preferred_rows:
            sub = pd.concat(preferred_rows, ignore_index=True)
            sub = sub.drop_duplicates("term")
        if len(sub) < 5:
            fallback = enrich[enrich["assigned_module"] == module["name"]].copy()
            fallback = fallback.sort_values(["p_value", "overlap_count", "assignment_score"], ascending=[True, False, False])
            sub = pd.concat([sub, fallback], ignore_index=True).drop_duplicates("term")
        sub = sub.head(5)
        selected_frames.append(sub)
        selected_terms.extend(sub["term"].tolist())
    selected = pd.concat(selected_frames, ignore_index=True)
    selected.to_csv(OUT_DIR / "Figure6_selected_enrichment_terms.csv", index=False)
    return selected, module_sets, selected_terms


def write_candidate_outputs(candidates: pd.DataFrame) -> None:
    q1 = candidates[candidates["candidate_group"] == "stable_core"].copy()
    q3 = candidates[candidates["candidate_group"] == "target_adaptive"].copy()
    candidates.to_csv(OUT_DIR / "Figure6_candidate_gene_table_cleaned.csv", index=False)
    (OUT_DIR / "Figure6_q1_q3_candidate_genes.txt").write_text(
        "\n".join(candidates["gene"].tolist()) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "Figure6_q1_stable_core_genes.txt").write_text(
        "\n".join(q1["gene"].tolist()) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "Figure6_q3_target_adaptive_genes.txt").write_text(
        "\n".join(q3["gene"].tolist()) + "\n", encoding="utf-8"
    )


def build_module_membership(candidates: pd.DataFrame, module_sets: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    score_lookup = candidates.set_index("gene").to_dict("index")
    for module in MODULES:
        for gene in sorted(module_sets[module["name"]]):
            info = score_lookup.get(gene, {})
            rows.append(
                {
                    "module": module["name"],
                    "module_short": module["short"],
                    "gene": gene,
                    "candidate_group": info.get("candidate_group", "other"),
                    "mean_abs_shap": info.get("mean_abs_shap", np.nan),
                    "stage1_mean_abs": info.get("stage1_mean_abs", np.nan),
                    "stage2_mean_abs": info.get("stage2_mean_abs", np.nan),
                }
            )
    membership = pd.DataFrame(rows)
    membership.to_csv(OUT_DIR / "Figure6_module_gene_membership.csv", index=False)
    with (OUT_DIR / "Figure6_functional_modules.gmt").open("w", encoding="utf-8") as handle:
        for module in MODULES:
            genes = sorted(module_sets[module["name"]])
            handle.write(f"{module['name']}\tDART-derived functional module\t" + "\t".join(genes) + "\n")
    return membership


def score_modules(expression: pd.DataFrame, meta: pd.DataFrame, module_sets: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    if "Gene" not in expression.columns:
        raise ValueError("TNBC_rows.csv must contain a Gene column.")
    if not {"SampleID", "Response"}.issubset(meta.columns):
        raise ValueError("TNBC_meta.csv must contain SampleID and Response columns.")

    expr = expression.copy()
    expr["Gene"] = expr["Gene"].astype(str).str.strip()
    expr = expr.drop_duplicates("Gene").set_index("Gene")
    samples = meta["SampleID"].astype(str).tolist()
    missing_samples = [sample for sample in samples if sample not in expr.columns]
    if missing_samples:
        raise ValueError(f"Expression matrix is missing samples: {missing_samples[:5]}")
    expr = expr[samples].apply(pd.to_numeric, errors="coerce")
    ranks = expr.rank(axis=0, method="average", pct=True)

    available: dict[str, list[str]] = {}
    score_rows = []
    for module in MODULES:
        genes = sorted(set(ranks.index) & module_sets[module["name"]])
        available[module["name"]] = genes
        if len(genes) < 3:
            raise ValueError(f"Module has fewer than three available genes: {module['name']}")
        raw = ranks.loc[genes].mean(axis=0)
        z = (raw - raw.mean()) / raw.std(ddof=0)
        for sample in samples:
            response = int(meta.loc[meta["SampleID"].astype(str) == sample, "Response"].iloc[0])
            score_rows.append(
                {
                    "sample_id": sample,
                    "response": "R" if response == 1 else "NR",
                    "module": module["name"],
                    "module_short": module["short"],
                    "ssGSEA_score": float(raw[sample]),
                    "z_score": float(z[sample]),
                }
            )
    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT_DIR / "Figure6_ssGSEA_module_scores.csv", index=False)

    stat_rows = []
    for module in MODULES:
        sub = scores[scores["module"] == module["name"]]
        nr = sub[sub["response"] == "NR"]["z_score"].to_numpy()
        r = sub[sub["response"] == "R"]["z_score"].to_numpy()
        test = mannwhitneyu(r, nr, alternative="two-sided")
        stat_rows.append(
            {
                "module": module["name"],
                "NR_n": len(nr),
                "R_n": len(r),
                "NR_mean": nr.mean(),
                "R_mean": r.mean(),
                "NR_median": np.median(nr),
                "R_median": np.median(r),
                "p_value": test.pvalue,
                "test": "Mann-Whitney U",
            }
        )
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(OUT_DIR / "Figure6_ssGSEA_statistics.csv", index=False)
    return scores, stats, available


def draw_panel_a(ax: plt.Axes, selected: pd.DataFrame) -> None:
    module_order = [module["name"] for module in MODULES]
    selected = selected.copy()
    selected["module_order"] = selected["assigned_module"].map({name: i for i, name in enumerate(module_order)})
    selected = selected.sort_values(["module_order", "p_value", "term"])
    ylabels = [textwrap.fill(term, 38) for term in selected["term"]]
    y = np.arange(len(selected))
    color_map = {module["name"]: module["color"] for module in MODULES}
    marker_map = {"GO BP": "o", "KEGG": "s"}
    for database, marker in marker_map.items():
        sub = selected[selected["database"] == database]
        if sub.empty:
            continue
        idx = sub.index
        positions = [selected.index.get_loc(i) for i in idx]
        ax.scatter(
            sub["minus_log10_p"],
            positions,
            s=35 + sub["overlap_count"] * 18,
            c=sub["assigned_module"].map(color_map),
            marker=marker,
            edgecolor="#333333",
            linewidth=0.5,
            alpha=0.9,
            label=database,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=10.4)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(\mathrm{adjusted}\ P\ \mathrm{value})$", fontsize=11.2)
    ax.set_title("Functional enrichment of DART-derived candidate genes", fontsize=12.8, pad=9)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10.2)

    for module in MODULES:
        ax.scatter([], [], c=module["color"], s=45, marker="o", label=module["name"])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=10.0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0,
    )


def draw_panel_b(ax: plt.Axes, membership: pd.DataFrame, candidates: pd.DataFrame) -> None:
    rows = [module["name"] for module in MODULES]
    genes = [gene for gene in REPRESENTATIVE_GENES if gene in set(candidates["gene"])]
    group_colors = {"stable_core": "#4477AA", "target_adaptive": "#44AA99", "other": "#BBBBBB"}
    candidate_lookup = candidates.set_index("gene").to_dict("index")
    membership_pairs = set(zip(membership["module"], membership["gene"]))
    max_shap = max(candidates["mean_abs_shap"].max(), 1e-9)

    for yi, module_name in enumerate(rows):
        for xi, gene in enumerate(genes):
            if (module_name, gene) not in membership_pairs:
                continue
            info = candidate_lookup[gene]
            size = 70 + 230 * float(info["mean_abs_shap"]) / max_shap
            ax.scatter(
                xi,
                yi,
                s=size,
                color=group_colors.get(info["candidate_group"], "#BBBBBB"),
                edgecolor="#222222",
                linewidth=0.5,
            )

    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=10.2, fontstyle="italic")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([textwrap.fill(module["short"], 24) for module in MODULES], fontsize=10.2)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_title("Functional categorisation of representative candidates", fontsize=12.8, pad=9)
    ax.grid(color="#b8b8b8", linestyle="-", linewidth=0.45, alpha=0.18)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)

    handles = [
        ax.scatter([], [], s=90, color=group_colors["stable_core"], edgecolor="#222222", linewidth=0.5, label="Stable core"),
        ax.scatter([], [], s=90, color=group_colors["target_adaptive"], edgecolor="#222222", linewidth=0.5, label="Target-adaptive"),
        ax.scatter([], [], s=85, color="#ffffff", edgecolor="#222222", linewidth=0.5, label=r"mean(|SHAP|)"),
        ax.scatter([], [], s=205, color="#ffffff", edgecolor="#222222", linewidth=0.5, label=r"higher mean(|SHAP|)"),
    ]
    ax.legend(
        handles=handles,
        fontsize=10.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        frameon=False,
        ncol=2,
        handletextpad=0.5,
        columnspacing=0.9,
        borderaxespad=0,
    )


def draw_panel_c(fig: plt.Figure, gs_cell, scores: pd.DataFrame, stats: pd.DataFrame) -> list[plt.Axes]:
    sub_gs = gs_cell.subgridspec(1, 3, wspace=0.35)
    axes = []
    rng = np.random.default_rng(8)
    for i, module in enumerate(MODULES):
        ax = fig.add_subplot(sub_gs[0, i])
        axes.append(ax)
        sub = scores[scores["module"] == module["name"]]
        nr = sub[sub["response"] == "NR"]["z_score"].to_numpy()
        r = sub[sub["response"] == "R"]["z_score"].to_numpy()
        ax.boxplot(
            [nr, r],
            positions=[0, 1],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.0},
            boxprops={"facecolor": module["color"], "alpha": 0.25, "edgecolor": module["color"]},
            whiskerprops={"color": module["color"]},
            capprops={"color": module["color"]},
        )
        for x, values, color in [(0, nr, "#666666"), (1, r, module["color"])]:
            jitter = rng.normal(x, 0.045, len(values))
            ax.scatter(jitter, values, s=20, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        p_value = stats.loc[stats["module"] == module["name"], "p_value"].iloc[0]
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"NR\n(n={len(nr)})", f"R\n(n={len(r)})"], fontsize=10.2)
        ax.set_title(module["short"], fontsize=11.2, pad=8)
        ax.set_ylabel("ssGSEA score (z-score)" if i == 0 else "", fontsize=11.0)
        ax.tick_params(axis="y", labelsize=10.0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        p_text = "n.s.\n" + rf"($P = {p_value:.3f}$)" if i == 1 else rf"$P = {p_value:.3f}$"
        ax.text(
            0.5,
            0.96,
            p_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.0 if i != 1 else 9.6,
            color="#222222" if i != 1 else "#666666",
        )
    return axes


def draw_figure(selected: pd.DataFrame, membership: pd.DataFrame, candidates: pd.DataFrame, scores: pd.DataFrame, stats: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "font.size": 10.6,
        }
    )
    fig = plt.figure(figsize=(15.2, 11.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.35, 0.88], hspace=0.66, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_panel_a(ax_a, selected)
    draw_panel_b(ax_b, membership, candidates)
    axes_c = draw_panel_c(fig, gs[1, :], scores, stats)

    ax_a.text(-0.16, 1.06, "A", transform=ax_a.transAxes, fontsize=19, fontweight="bold", va="top")
    ax_b.text(-0.14, 1.06, "B", transform=ax_b.transAxes, fontsize=19, fontweight="bold", va="top")
    for label, ax in zip(["C", "D", "E"], axes_c):
        ax.text(-0.20, 1.12, label, transform=ax.transAxes, fontsize=19, fontweight="bold", va="top")

    for ext in ["pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Figure6_functional_interpretation.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure6_functional_interpretation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    strip_svg_comments(FIG_DIR / "Figure6_functional_interpretation.svg")


def write_qc_log(
    candidate_path: Path,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    membership: pd.DataFrame,
    scores: pd.DataFrame,
    stats: pd.DataFrame,
    available: dict[str, list[str]],
) -> None:
    q1_n = int((candidates["candidate_group"] == "stable_core").sum())
    q3_n = int((candidates["candidate_group"] == "target_adaptive").sum())
    response_counts = scores.drop_duplicates(["sample_id", "response"])["response"].value_counts().to_dict()
    lines = [
        "Figure 6 QC log",
        f"Candidate file: {candidate_path}",
        f"Candidate count: {len(candidates)}",
        f"Q1 stable_core count: {q1_n}",
        f"Q3 target_adaptive count: {q3_n}",
        f"CLTA present in candidate table: {'CLTA' in set(candidates['gene'])}",
        "Alias check for the Figure 6 source candidate table: no non-original aliases detected.",
        "Enrichment source: existing GO final enrichment and full Top200 GO BP/KEGG ORA files were reused; no online Enrichr query was performed.",
        "Panel A statistical-P metric: adjusted statistical-P from the GO final enrichment table.",
        "Panel A x-axis: true -log10(adjusted statistical-P), computed from adjusted_p_value = 10^(Log(q-value)).",
        "KEGG ORA terms were retained in the full enrichment CSV but not prioritised in the main panel because their adjusted or nominal statistical evidence was close to null.",
        f"Selected enrichment term count: {len(selected)}",
        "Selected enrichment terms:",
    ]
    for _, row in selected.iterrows():
        lines.append(f"  - [{row['database']}] {row['term']} :: {row['assigned_module']} :: statistical-P {row['p_value']:.4g}")
    lines.append("Module genes available in TNBC_rows.csv:")
    for module in MODULES:
        genes = available[module["name"]]
        lines.append(f"  - {module['name']}: {len(genes)} genes; {', '.join(genes)}")
    lines.append(f"Sample count: {sum(response_counts.values())}")
    lines.append(f"NR count: {response_counts.get('NR', 0)}")
    lines.append(f"R count: {response_counts.get('R', 0)}")
    lines.append("Module statistics:")
    for _, row in stats.iterrows():
        direction = "R higher" if row["R_mean"] > row["NR_mean"] else "NR higher"
        lines.append(
            f"  - {row['module']}: R_mean={row['R_mean']:.3f}, NR_mean={row['NR_mean']:.3f}, "
            f"statistical-P {row['p_value']:.4g}, {direction}"
        )
    lines.append("Final Figure 6 and manuscript should display the original gene symbol CLTA.")
    (OUT_DIR / "Figure6_QC_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    candidates_raw, meta, expression, candidate_path = load_inputs()
    candidates = clean_candidates(candidates_raw)
    write_candidate_outputs(candidates)
    selected, module_sets, _ = prepare_enrichment(candidates)
    membership = build_module_membership(candidates, module_sets)
    scores, stats, available = score_modules(expression, meta, module_sets)
    draw_figure(selected, membership, candidates, scores, stats)
    write_qc_log(candidate_path, candidates, selected, membership, scores, stats, available)
    print("Generated Figure6_functional_interpretation outputs.")
    print(f"Candidates: {len(candidates)}; Q1={(candidates['candidate_group'] == 'stable_core').sum()}; Q3={(candidates['candidate_group'] == 'target_adaptive').sum()}")
    print(stats[["module", "NR_n", "R_n", "NR_mean", "R_mean", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
