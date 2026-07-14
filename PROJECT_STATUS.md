# Project Status

## 1. Paper-linked code

See `docs/paper_code_map.md`. Final candidates are mainly inferred from `04_paper`, `05_bio`, SHAP folders, dated paper-ready outputs, and manuscript plotting scripts.

## 2. Code not confidently found

- Exact one-command training entry point for the final DART paper result: NEEDS_MANUAL_CONFIRMATION.
- Exact public-release data download/preprocessing wrapper: NEEDS_MANUAL_CONFIRMATION.
- Whether EMA/teacher/pseudo-label components were active in the final reported model: NEEDS_MANUAL_CONFIRMATION.

## 3. Figures with generation scripts found

- Figure 2-6 plotting scripts were copied from the manuscript workspace when available.
- Additional figure-generation candidates are listed in `docs/paper_code_map.md`.

## 4. Figures not confidently found

- Figure 1 appears to be artwork/PDF based rather than a reproducible data plot: NEEDS_MANUAL_CONFIRMATION.
- Supplementary Figure S1 generation script requires manual confirmation if it should be public.

## 5. Supplementary materials

- Supplementary Tables S1-S4 are represented in the manuscript workspace; final source scripts for all tables require manual confirmation.

## 6. Manual confirmation

- Review all `NEEDS_MANUAL_CONFIRMATION` markers before GitHub publication.
- Replace absolute Windows paths with config-driven paths.
- Add small test data or documented expected input schemas.

## 7. Code recommended for public release

- Curated preprocessing/TDM scripts.
- Final DART model/training scripts after path parameterisation.
- Baseline evaluation scripts.
- SHAP/enrichment/figure plotting scripts.

## 8. Code not recommended as canonical public entry points

- Old/test/debug/v1 duplicated scripts; keep only in `original_code/` for traceability.
- Large derived matrices, checkpoints, and raw data files.

## 9. GitHub completion estimate: 72%

The release skeleton, docs, candidate code copies, and metadata files have been generated. Remaining work is manual verification and path/config cleanup.

## 10. Remaining before public release

- Confirm canonical final training script and exact commands.
- Parameterise hard-coded local paths.
- Add data-download instructions and checksums where possible.
- Add smoke tests using toy data.
- Choose license and replace placeholder author metadata in `CITATION.cff`.
