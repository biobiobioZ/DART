# Method Consistency Check

This is an automated keyword-based consistency audit. It does not modify manuscript text or algorithm code.

| Manuscript claim | Evidence in code tree | Status |
|---|---|---|
| Top800 MAD | top800, mad | FOUND_KEYWORD_EVIDENCE |
| 50 repeated validations / holdout | 50, repeated, holdout | FOUND_KEYWORD_EVIDENCE |
| Stage0 | stage0 | FOUND_KEYWORD_EVIDENCE |
| Stage1 | stage1 | FOUND_KEYWORD_EVIDENCE |
| Stage2 | stage2 | FOUND_KEYWORD_EVIDENCE |
| GRL / domain adversarial | grl | FOUND_KEYWORD_EVIDENCE |
| EMA / teacher | ema, teacher | FOUND_KEYWORD_EVIDENCE |
| TDM quantile/ECDF | tdm, ecdf | FOUND_KEYWORD_EVIDENCE |
| Cross-fit OOS | oos | FOUND_KEYWORD_EVIDENCE |
| SHAP | shap | FOUND_KEYWORD_EVIDENCE |
| GO/KEGG/ssGSEA | go_bp, kegg, ssgsea, gseapy | FOUND_KEYWORD_EVIDENCE |

## Manual verification needed

- Verify exact data splits, seed control, and whether reported tables were produced by the final candidate scripts.
- Verify whether Stage2 freezes/unfreezes encoder components as described in the manuscript.
- Verify whether EMA/teacher/pseudo-label code is active in final training or only present in older exploratory versions.
