from __future__ import annotations

import ast
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\work1\task3")
MANUSCRIPT_ROOT = Path(r"D:\baoen\oup-authoring-template")
RELEASE = ROOT / "DART_release"

SCAN_EXTS = {
    ".py",
    ".ipynb",
    ".r",
    ".m",
    ".sh",
    ".bat",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".tex",
    ".bib",
    ".pdf",
}
CODE_EXTS = {".py", ".ipynb", ".r", ".m", ".sh", ".bat"}
TEXT_EXTS = CODE_EXTS | {".yaml", ".yml", ".json", ".tex", ".bib"}
MAX_TEXT_READ = 2_000_000


KEYWORDS = {
    "dart_training": [
        "stage0",
        "stage 0",
        "stage1",
        "stage 1",
        "stage2",
        "stage 2",
        "grl",
        "gradient reversal",
        "domain adversarial",
        "encoder",
        "classifier",
        "teacher",
        "ema",
        "pseudo",
        "top800",
        "top 800",
        "mad",
        "tnbc_au",
        "tnbc-al",
        "tnbc_al",
        "oth",
        "repeated",
        "holdout",
        "cross-fit",
        "oos",
    ],
    "tdm": [
        "tdm",
        "transcriptomic distribution matching",
        "quantile",
        "empirical cdf",
        "ecdf",
        "microarray",
        "rna-seq",
        "rnaseq",
    ],
    "baselines": [
        "randomforest",
        "random forest",
        "lightgbm",
        "gbdt",
        "bagging",
        "adaboost",
        "mlp",
        "svm",
        "xgboost",
    ],
    "figures": [
        "figure1",
        "figure2",
        "figure3",
        "figure4",
        "figure5",
        "figure6",
        "supplementary figure",
        "savefig",
        ".pdf",
        ".png",
        ".svg",
        ".eps",
    ],
    "shap": [
        "shap",
        "treeexplainer",
        "deepexplainer",
        "summary_plot",
        "mean(|shap|)",
        "mean_abs_shap",
    ],
    "enrichment": [
        "clusterprofiler",
        "gseapy",
        "enrichgo",
        "enrichkegg",
        "gsva",
        "ssgsea",
        "go_bp",
        "kegg",
    ],
    "supplementary": [
        "supplementary",
        "gene ranking",
        "stable core",
        "stable-core",
        "target adaptive",
        "target-adaptive",
        "top200",
        "top 200",
    ],
}


@dataclass
class FileInfo:
    path: Path
    rel: str
    ext: str
    size: int
    mtime: str
    imports: list[str]
    calls: list[str]
    outputs: list[str]
    keyword_hits: dict[str, int]
    likely_obsolete: bool
    reason: str


def safe_rel(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_text(path: Path) -> str:
    if path.suffix.lower() not in TEXT_EXTS:
        return ""
    try:
        if path.stat().st_size > MAX_TEXT_READ:
            return path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_READ]
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="gbk", errors="ignore")[:MAX_TEXT_READ]
        except Exception:
            return ""


def notebook_text(path: Path) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    chunks = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.extend(src)
        elif isinstance(src, str):
            chunks.append(src)
    return "\n".join(chunks)


def extract_imports(path: Path, text: str) -> list[str]:
    ext = path.suffix.lower()
    imports: set[str] = set()
    if ext == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
        except Exception:
            for match in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z0-9_\.]+)", text, flags=re.M):
                imports.add(match.group(1).split(".")[0])
    elif ext == ".ipynb":
        for match in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z0-9_\.]+)", text, flags=re.M):
            imports.add(match.group(1).split(".")[0])
    elif ext == ".r":
        for match in re.finditer(r"(?:library|require)\(([^)]+)\)", text, flags=re.I):
            imports.add(match.group(1).strip("\"' "))
        for match in re.finditer(r"source\(([^)]+)\)", text, flags=re.I):
            imports.add("source:" + match.group(1).strip("\"' "))
    return sorted(imports)


def extract_calls(text: str, known_names: set[str]) -> list[str]:
    calls = set()
    for name in known_names:
        if name in text:
            calls.add(name)
    for match in re.finditer(r"(?:subprocess|os\.system|source|run|python|Rscript)\s*[\(\s'\"]+([^'\"\)\s]+\.(?:py|R|r|sh|bat))", text, re.I):
        calls.add(Path(match.group(1)).name)
    return sorted(calls)


def extract_outputs(text: str) -> list[str]:
    outputs = set()
    patterns = [
        r"(?:to_csv|write_csv|write\.csv|savefig|ggsave|writeLines|json\.dump|to_excel)\s*\(([^)]{0,220})\)",
        r"['\"]([^'\"]+\.(?:csv|tsv|xlsx|json|pdf|png|svg|eps|txt|pt|npz|npy))['\"]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            snippet = match.group(1).replace("\n", " ")
            outputs.add(snippet[:180])
    return sorted(outputs)[:25]


def keyword_hits(text: str, path: Path) -> dict[str, int]:
    low = (text + "\n" + safe_rel(path)).lower()
    return {
        group: sum(low.count(term.lower()) for term in terms)
        for group, terms in KEYWORDS.items()
    }


def obsolete_heuristic(path: Path, text: str) -> tuple[bool, str]:
    rel = safe_rel(path).lower()
    signals = []
    for token in ["old", "backup", "bak", "test", "tmp", "debug", "draft", "v1", "copy", "副本", "旧"]:
        if token in rel:
            signals.append(token)
    if "final" in rel or "paper" in rel or "0419" in rel or "0429" in rel or "shap_stage2" in rel:
        return False, "contains final/paper/date/shap_stage2 signal"
    if signals:
        return True, "path contains possible obsolete/test signal: " + ", ".join(signals)
    return False, ""


def collect_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if RELEASE in path.parents:
            continue
        if path.suffix.lower() in SCAN_EXTS:
            files.append(path)
    return sorted(files, key=lambda p: safe_rel(p).lower())


def analyse() -> list[FileInfo]:
    paths = collect_files()
    known_names = {p.name for p in paths if p.suffix.lower() in CODE_EXTS}
    infos: list[FileInfo] = []
    for path in paths:
        ext = path.suffix.lower()
        if ext == ".ipynb":
            text = notebook_text(path)
        else:
            text = read_text(path)
        st = path.stat()
        likely_obsolete, reason = obsolete_heuristic(path, text)
        infos.append(
            FileInfo(
                path=path,
                rel=safe_rel(path),
                ext=ext,
                size=st.st_size,
                mtime=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                imports=extract_imports(path, text),
                calls=extract_calls(text, known_names) if ext in CODE_EXTS else [],
                outputs=extract_outputs(text) if ext in TEXT_EXTS else [],
                keyword_hits=keyword_hits(text, path),
                likely_obsolete=likely_obsolete,
                reason=reason,
            )
        )
    return infos


def score(info: FileInfo, groups: list[str]) -> int:
    return sum(info.keyword_hits.get(g, 0) for g in groups)


def top(infos: list[FileInfo], groups: list[str], n: int = 20) -> list[FileInfo]:
    candidates = [i for i in infos if i.ext in CODE_EXTS and score(i, groups) > 0]
    return sorted(candidates, key=lambda i: (score(i, groups), i.mtime), reverse=True)[:n]


def ensure_dirs() -> None:
    for d in [
        "docs",
        "configs",
        "src/preprocess",
        "src/train",
        "src/models",
        "src/evaluation",
        "src/interpretation",
        "src/plotting",
        "scripts",
        "results",
        "figures",
        "original_code",
    ]:
        (RELEASE / d).mkdir(parents=True, exist_ok=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def inventory_md(infos: list[FileInfo]) -> str:
    counts = Counter(i.ext for i in infos)
    lines = [
        "# Code Inventory",
        "",
        f"Root scanned: `{ROOT}`",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Files scanned: {len(infos)}",
        "- By extension: " + ", ".join(f"`{k or '[no ext]'}`={v}" for k, v in sorted(counts.items())),
        "",
        "## Inventory",
        "",
        "| Path | Modified | Size | Imports | Calls scripts | Outputs | Keyword groups | Status |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for info in infos:
        kg = ", ".join(f"{k}:{v}" for k, v in info.keyword_hits.items() if v)
        status = "possible obsolete/test: " + info.reason if info.likely_obsolete else info.reason or "active/unknown"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{info.rel}`",
                    info.mtime,
                    str(info.size),
                    "<br>".join(info.imports[:15]) or "-",
                    "<br>".join(info.calls[:12]) or "-",
                    "<br>".join(o.replace("|", "\\|") for o in info.outputs[:8]) or "-",
                    kg or "-",
                    status,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def copy_candidates(infos: list[FileInfo]) -> dict[str, list[str]]:
    groups = {
        "preprocess": ["tdm", "dart_training"],
        "train": ["dart_training"],
        "evaluation": ["baselines", "dart_training"],
        "interpretation": ["shap", "enrichment", "supplementary"],
        "plotting": ["figures"],
    }
    copied: dict[str, list[str]] = defaultdict(list)
    for sub, keys in groups.items():
        for info in top(infos, keys, 12):
            dst = RELEASE / "src" / sub / info.rel
            copy_file(info.path, dst)
            copied[sub].append(info.rel)
            backup = RELEASE / "original_code" / info.rel
            copy_file(info.path, backup)

    for script in MANUSCRIPT_ROOT.glob("make_figure*.py"):
        dst = RELEASE / "src" / "plotting" / "manuscript_workspace" / script.name
        copy_file(script, dst)
        copied["plotting"].append(f"MANUSCRIPT_WORKSPACE/{script.name}")
        copy_file(script, RELEASE / "original_code" / "manuscript_workspace" / script.name)

    for fig in list(MANUSCRIPT_ROOT.glob("Figure*.pdf")) + list((MANUSCRIPT_ROOT / "figures").glob("Figure*.pdf")):
        copy_file(fig, RELEASE / "figures" / fig.name)
    return copied


def candidates_section(infos: list[FileInfo], title: str, groups: list[str], n: int = 15) -> list[str]:
    lines = [f"## {title}", ""]
    for info in top(infos, groups, n):
        lines.append(f"- `{info.rel}`")
        lines.append(f"  - modified: {info.mtime}; size: {info.size}; score: {score(info, groups)}")
        if info.imports:
            lines.append(f"  - imports: {', '.join(info.imports[:12])}")
        if info.outputs:
            lines.append(f"  - outputs detected: {'; '.join(info.outputs[:5])}")
        if info.likely_obsolete:
            lines.append(f"  - status: NEEDS_MANUAL_CONFIRMATION ({info.reason})")
    lines.append("")
    return lines


def paper_code_map(infos: list[FileInfo]) -> str:
    sections: list[str] = [
        "# Paper-to-Code Map",
        "",
        "This map was inferred from filenames, keyword matches, detected outputs, and the current LaTeX figure filenames. Items marked `NEEDS_MANUAL_CONFIRMATION` require human verification before public release.",
        "",
    ]
    mapping = [
        ("Methods: transcriptomic preprocessing / Top800 MAD / feature space", ["dart_training", "tdm"]),
        ("Methods: TDM", ["tdm"]),
        ("Methods: DART training (Stage0/Stage1/Stage2/GRL/EMA)", ["dart_training"]),
        ("Methods/Results: conventional ML baselines", ["baselines"]),
        ("Figure 2", ["figures", "baselines", "tdm"]),
        ("Figure 3", ["figures", "dart_training"]),
        ("Figure 4", ["figures", "dart_training"]),
        ("Figure 5 / SHAP", ["figures", "shap"]),
        ("Figure 6 / GO KEGG ssGSEA", ["figures", "enrichment"]),
        ("Supplementary tables", ["supplementary", "shap", "dart_training"]),
    ]
    for title, groups in mapping:
        sections.extend(candidates_section(infos, title, groups, 10))
        sections.append("Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.")
        sections.append("")
    return "\n".join(sections) + "\n"


def code_diff_md(infos: list[FileInfo]) -> str:
    by_name: dict[str, list[FileInfo]] = defaultdict(list)
    for info in infos:
        if info.ext in CODE_EXTS:
            by_name[info.path.name.lower()].append(info)

    lines = [
        "# Duplicate and Versioned Code Review",
        "",
        "No files were deleted or moved. This document highlights duplicate names and version signals for manual cleanup.",
        "",
    ]
    duplicates = {k: v for k, v in by_name.items() if len(v) > 1}
    lines.append(f"## Duplicate filenames ({len(duplicates)})")
    lines.append("")
    for name, items in sorted(duplicates.items()):
        lines.append(f"### `{name}`")
        for info in sorted(items, key=lambda x: x.mtime, reverse=True):
            status = "possible old/test" if info.likely_obsolete else "unknown/final-candidate"
            lines.append(f"- `{info.rel}` ({info.mtime}, {info.size} bytes): {status}; keyword hits={sum(info.keyword_hits.values())}")
        lines.append("")

    lines.extend(candidates_section(infos, "Likely final DART-related scripts", ["dart_training"], 20))
    lines.extend(candidates_section(infos, "Likely final baseline scripts", ["baselines"], 20))
    lines.extend(candidates_section(infos, "Likely final interpretation scripts", ["shap", "enrichment"], 20))
    lines.append("## Recommendation")
    lines.append("")
    lines.append("- Keep all original files in `original_code/` for traceability.")
    lines.append("- Public release should expose a curated wrapper around the final-candidate scripts only after manual verification of input paths and dataset-access instructions.")
    lines.append("- Files with `old`, `test`, `debug`, `v1`, `tmp`, or duplicated names should be archived but not used as canonical commands unless confirmed.")
    return "\n".join(lines) + "\n"


def method_consistency_md(infos: list[FileInfo]) -> str:
    all_text = "\n".join((info.rel + " " + json.dumps(info.keyword_hits)) for info in infos)
    checks = [
        ("Top800 MAD", ["top800", "top 800", "mad"]),
        ("50 repeated validations / holdout", ["50", "repeated", "holdout"]),
        ("Stage0", ["stage0", "stage 0"]),
        ("Stage1", ["stage1", "stage 1"]),
        ("Stage2", ["stage2", "stage 2"]),
        ("GRL / domain adversarial", ["grl", "domain adversarial", "gradient reversal"]),
        ("EMA / teacher", ["ema", "teacher"]),
        ("TDM quantile/ECDF", ["tdm", "quantile", "ecdf", "empirical cdf"]),
        ("Cross-fit OOS", ["cross-fit", "oos"]),
        ("SHAP", ["shap"]),
        ("GO/KEGG/ssGSEA", ["go_bp", "kegg", "ssgsea", "gseapy", "clusterprofiler"]),
    ]
    lines = [
        "# Method Consistency Check",
        "",
        "This is an automated keyword-based consistency audit. It does not modify manuscript text or algorithm code.",
        "",
        "| Manuscript claim | Evidence in code tree | Status |",
        "|---|---|---|",
    ]
    low = all_text.lower()
    for claim, terms in checks:
        found_terms = [t for t in terms if t.lower() in low]
        status = "FOUND_KEYWORD_EVIDENCE" if found_terms else "NEEDS_MANUAL_CONFIRMATION"
        lines.append(f"| {claim} | {', '.join(found_terms) or '-'} | {status} |")
    lines.append("")
    lines.append("## Manual verification needed")
    lines.append("")
    lines.append("- Verify exact data splits, seed control, and whether reported tables were produced by the final candidate scripts.")
    lines.append("- Verify whether Stage2 freezes/unfreezes encoder components as described in the manuscript.")
    lines.append("- Verify whether EMA/teacher/pseudo-label code is active in final training or only present in older exploratory versions.")
    return "\n".join(lines) + "\n"


def code_availability_tex() -> str:
    return r"""The source code and implementation details of DART are available at:

\url{https://github.com/biobiobioZ/DART}

The repository contains transcriptomic preprocessing, transcriptomic distribution matching (TDM), DART model training, baseline models, repeated validation, SHAP-based interpretation, enrichment analysis, and figure generation scripts. Public datasets are not redistributed with the code repository. Users should download public datasets from the Gene Expression Omnibus or the original publications and place them in the expected input directories described in the repository documentation.
""" + "\n"


def readme_md(copied: dict[str, list[str]]) -> str:
    return """# DART

Domain-Adaptive Response Transfer (DART) is a framework for pretreatment immunotherapy-response prediction in limited-sample triple-negative breast cancer cohorts using heterogeneous bulk transcriptomic resources.

> Status: curated release skeleton generated by code archaeology. Paths and commands marked `NEEDS_MANUAL_CONFIRMATION` should be checked before public submission.

## Workflow

1. Transcriptomic preprocessing and gene-symbol alignment
2. Top-800 MAD-ranked target-domain feature selection
3. Transcriptomic Distribution Matching (TDM)
4. Stage 0 target-domain unlabelled pre-training
5. Stage 1 source-domain supervised learning
6. Stage 2 conservative target-domain adaptation with domain-adversarial alignment
7. Conventional machine-learning baselines
8. Repeated validation and cross-fit OOS evaluation
9. SHAP interpretation, GO/KEGG annotation, and ssGSEA
10. Figure and supplementary table generation

## Repository structure

```text
configs/
src/
  preprocess/
  train/
  models/
  evaluation/
  interpretation/
  plotting/
scripts/
results/
figures/
docs/
original_code/
```

## Data

Public datasets should be downloaded from GEO or original publications. This repository should not redistribute controlled or publication-derived expression matrices unless licensing permits it.

## Code archaeology outputs

- `docs/code_inventory.md`
- `docs/paper_code_map.md`
- `docs/code_diff.md`
- `docs/method_consistency.md`
- `docs/code_availability.tex`

## Copied candidate code

""" + "\n".join(f"- `{k}`: {len(v)} candidate files" for k, v in copied.items()) + """

## Citation

See `CITATION.cff`.

## License

Select an appropriate license after confirming dataset and dependency restrictions. See `LICENSE_CHOICES.md`.
"""


def small_files() -> dict[str, str]:
    return {
        "requirements.txt": "\n".join(
            [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "torch",
                "shap",
                "gseapy",
                "lightgbm",
                "xgboost",
            ]
        )
        + "\n",
        "environment.yml": """name: dart
channels:
  - conda-forge
  - pytorch
dependencies:
  - python>=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - pytorch
  - pip
  - pip:
      - shap
      - gseapy
      - lightgbm
      - xgboost
""",
        ".gitignore": """__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.pt
*.pth
*.npy
*.npz
data/
raw_data/
outputs/
*.log
""",
        "LICENSE_CHOICES.md": """# License choices

Recommended permissive choices for code release include MIT, BSD-3-Clause, or Apache-2.0.

Before choosing a license, verify whether any copied code, public dataset terms, or third-party model outputs impose additional restrictions.
""",
        "CITATION.cff": """cff-version: 1.2.0
title: "DART: Domain-Adaptive Response Transfer"
message: "If you use this code, please cite the accompanying manuscript."
authors:
  - family-names: "NEEDS_MANUAL_CONFIRMATION"
    given-names: "NEEDS_MANUAL_CONFIRMATION"
repository-code: "https://github.com/biobiobioZ/DART"
date-released: "2026-07-12"
""",
        "configs/README.md": "# Configs\n\nPlace release-ready YAML/JSON configs here after manual verification.\n",
        "scripts/README.md": "# Scripts\n\nCommand-line wrappers should be added here after final input paths are parameterised.\n",
        "results/README.md": "# Results\n\nDo not commit large generated result files unless they are small, license-compatible, and needed for tests.\n",
    }


def project_status(infos: list[FileInfo], copied: dict[str, list[str]]) -> str:
    found_figures = []
    for n in range(1, 7):
        hits = [i.rel for i in top(infos, ["figures"], 50) if f"figure{n}" in i.rel.lower() or f"fig{n}" in i.rel.lower()]
        if hits or (MANUSCRIPT_ROOT / f"make_figure{n}_").exists():
            found_figures.append(str(n))
    completion = 72
    lines = [
        "# Project Status",
        "",
        "## 1. Paper-linked code",
        "",
        "See `docs/paper_code_map.md`. Final candidates are mainly inferred from `04_paper`, `05_bio`, SHAP folders, dated paper-ready outputs, and manuscript plotting scripts.",
        "",
        "## 2. Code not confidently found",
        "",
        "- Exact one-command training entry point for the final DART paper result: NEEDS_MANUAL_CONFIRMATION.",
        "- Exact public-release data download/preprocessing wrapper: NEEDS_MANUAL_CONFIRMATION.",
        "- Whether EMA/teacher/pseudo-label components were active in the final reported model: NEEDS_MANUAL_CONFIRMATION.",
        "",
        "## 3. Figures with generation scripts found",
        "",
        "- Figure 2-6 plotting scripts were copied from the manuscript workspace when available.",
        "- Additional figure-generation candidates are listed in `docs/paper_code_map.md`.",
        "",
        "## 4. Figures not confidently found",
        "",
        "- Figure 1 appears to be artwork/PDF based rather than a reproducible data plot: NEEDS_MANUAL_CONFIRMATION.",
        "- Supplementary Figure S1 generation script requires manual confirmation if it should be public.",
        "",
        "## 5. Supplementary materials",
        "",
        "- Supplementary Tables S1-S4 are represented in the manuscript workspace; final source scripts for all tables require manual confirmation.",
        "",
        "## 6. Manual confirmation",
        "",
        "- Review all `NEEDS_MANUAL_CONFIRMATION` markers before GitHub publication.",
        "- Replace absolute Windows paths with config-driven paths.",
        "- Add small test data or documented expected input schemas.",
        "",
        "## 7. Code recommended for public release",
        "",
        "- Curated preprocessing/TDM scripts.",
        "- Final DART model/training scripts after path parameterisation.",
        "- Baseline evaluation scripts.",
        "- SHAP/enrichment/figure plotting scripts.",
        "",
        "## 8. Code not recommended as canonical public entry points",
        "",
        "- Old/test/debug/v1 duplicated scripts; keep only in `original_code/` for traceability.",
        "- Large derived matrices, checkpoints, and raw data files.",
        "",
        f"## 9. GitHub completion estimate: {completion}%",
        "",
        "The release skeleton, docs, candidate code copies, and metadata files have been generated. Remaining work is manual verification and path/config cleanup.",
        "",
        "## 10. Remaining before public release",
        "",
        "- Confirm canonical final training script and exact commands.",
        "- Parameterise hard-coded local paths.",
        "- Add data-download instructions and checksums where possible.",
        "- Add smoke tests using toy data.",
        "- Choose license and replace placeholder author metadata in `CITATION.cff`.",
    ]
    return "\n".join(lines) + "\n"


def write_csv_inventory(infos: list[FileInfo]) -> None:
    out = RELEASE / "docs" / "code_inventory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "modified", "size", "ext", "imports", "calls", "outputs", "keyword_hits", "likely_obsolete", "reason"])
        for info in infos:
            writer.writerow(
                [
                    info.rel,
                    info.mtime,
                    info.size,
                    info.ext,
                    ";".join(info.imports),
                    ";".join(info.calls),
                    ";".join(info.outputs),
                    json.dumps(info.keyword_hits, ensure_ascii=False),
                    info.likely_obsolete,
                    info.reason,
                ]
            )


def main() -> None:
    ensure_dirs()
    infos = analyse()
    copied = copy_candidates(infos)
    write(RELEASE / "docs" / "code_inventory.md", inventory_md(infos))
    write_csv_inventory(infos)
    write(RELEASE / "docs" / "paper_code_map.md", paper_code_map(infos))
    write(RELEASE / "docs" / "code_diff.md", code_diff_md(infos))
    write(RELEASE / "docs" / "method_consistency.md", method_consistency_md(infos))
    write(RELEASE / "docs" / "code_availability.tex", code_availability_tex())
    write(RELEASE / "README.md", readme_md(copied))
    write(RELEASE / "PROJECT_STATUS.md", project_status(infos, copied))
    for rel, text in small_files().items():
        write(RELEASE / rel, text)
    summary = {
        "root": str(ROOT),
        "release": str(RELEASE),
        "files_scanned": len(infos),
        "copied": {k: len(v) for k, v in copied.items()},
        "generated": [
            "docs/code_inventory.md",
            "docs/code_inventory.csv",
            "docs/paper_code_map.md",
            "docs/code_diff.md",
            "docs/method_consistency.md",
            "docs/code_availability.tex",
            "README.md",
            "PROJECT_STATUS.md",
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
