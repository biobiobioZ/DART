# Paper-to-Code Map

This map was inferred from filenames, keyword matches, detected outputs, and the current LaTeX figure filenames. Items marked `NEEDS_MANUAL_CONFIRMATION` require human verification before public release.

## Methods: transcriptomic preprocessing / Top800 MAD / feature space

- `04_paper/shap/v4grl_v2_shap.py`
  - modified: 2026-03-05 14:42:00; size: 62600; score: 513
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, shap, sklearn, torch
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_paired.py`
  - modified: 2026-03-04 11:00:12; size: 54853; score: 452
  - imports: collections, copy, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_l2sp_on.py`
  - modified: 2026-03-04 10:51:37; size: 49578; score: 452
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full.py`
  - modified: 2026-03-04 00:10:41; size: 48645; score: 451
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot.py`
  - modified: 2026-03-04 22:59:04; size: 50143; score: 423
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2_fixed.py`
  - modified: 2026-03-04 23:21:55; size: 49443; score: 419
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2.py`
  - modified: 2026-03-04 23:05:54; size: 49275; score: 419
  - imports: collections, matplotlib, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_enplot.py`
  - modified: 2026-03-04 22:45:33; size: 48343; score: 410
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve.py`
  - modified: 2026-03-04 22:21:10; size: 48321; score: 408
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `04_paper/06_tf/allrun/v4grl_v2_stage2_history.py`
  - modified: 2026-03-19 00:26:42; size: 45202; score: 399
  - imports: collections, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Methods: TDM

- `04_paper/03_1_ml/ml_tdm_enhanced_ml_deg_baseline.py`
  - modified: 2026-04-29 23:52:55; size: 41736; score: 36
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `02 grl/grl_1/uda_twoheads_full_clickrun_fixed_v3.py`
  - modified: 2026-01-26 11:55:45; size: 47564; score: 36
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected:              {                 "chosen_for_exports": chosen,                 "OTH_val": metrics_summary.get("OTH_val", {}; D:\work1\task 3\01 uda\uda_5\outputs_uda_click\genes.json; D:\work1\task 3\02 grl\grl_1\tdmdata\OTH_TDM.csv; Path(out_dir; checkpoint_uda_best_othval.pt
- `04_paper/03_1_ml/ml_tdm_enhanced_baseline.py`
  - modified: 2026-04-29 23:34:54; size: 37359; score: 33
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `02 grl/grl_1/uda_twoheads_full_clickrun_fixed_v2.py`
  - modified: 2026-01-15 21:54:13; size: 42028; score: 32
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch
  - outputs detected:              {                 "chosen_for_exports": chosen,                 "OTH_val": metrics_summary.get("OTH_val", {}; D:\work1\task 3\01 uda\uda_5\outputs_uda_click\genes.json; D:\work1\task 3\02 grl\grl_1\int\TNBC_LA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULR_int.csv
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/process_TideBulk3.ipynb`
  - modified: 2024-12-11 01:52:26; size: 72926; score: 31
  - imports: numpy, pandas, scipy, sklearn, sksurv
  - outputs detected: 'test2.csv', index=False, columns=col_sym; -TMM.txt; -clinic.csv; ./Kegg/human_KeggPathwayNet.txt; ./genelist_8080.txt
- `02 grl/grl_1/uda_twoheads_clickrun.py`
  - modified: 2026-01-15 18:13:21; size: 38818; score: 26
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch
  - outputs detected:              {                 "OTH_val": metrics_summary.get("OTH_val", {}; D:\work1\task 3\01 uda\uda_5\outputs_uda_click\genes.json; D:\work1\task 3\02 grl\grl_1\int\TNBC_LA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULR_int.csv
- `02 grl/grl_1/uda_twoheads_full_clickrun_fixed.py`
  - modified: 2026-01-15 19:16:27; size: 39039; score: 25
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch
  - outputs detected:              {                 "OTH_val": metrics_summary.get("OTH_val", {}; D:\work1\task 3\01 uda\uda_5\outputs_uda_click\genes.json; D:\work1\task 3\02 grl\grl_1\int\TNBC_LA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULR_int.csv
- `02 grl/tdm/tdmdata/tdm_oth_to_array.py`
  - modified: 2026-01-26 11:47:44; size: 5031; score: 24
  - imports: numpy, pandas, pathlib
  - outputs detected: D:\work1\task 3\02 grl\tdm\data\TNBC_AL.csv; D:\work1\task 3\02 grl\tdm\data\TNBC_AL_TDM.csv; D:\work1\task 3\02 grl\tdm\data\TNBC_AU.csv; out_path
- `02 grl/grl_1/uda_twoheads_full_clickrun_fixed_v3_cv5fold.py`
  - modified: 2026-01-21 11:03:05; size: 43807; score: 24
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: D:\work1\task 3\02 grl\grl_1\std\BRA_unlabeled_std.csv; D:\work1\task 3\02 grl\grl_1\std\OTH_labeled_std.csv; D:\work1\task 3\02 grl\grl_1\std\TNBC_ULR_std.csv; D:\work1\task 3\02 grl\grl_1\std\TNBC_labeled_std.csv; D:\work1\task 3\02 grl\grl_1\std\outint_v3\genes.json
  - status: NEEDS_MANUAL_CONFIRMATION (path contains possible obsolete/test signal: old)
- `02 grl/okre/data/tdmdata/tdm_oth_to_array.py`
  - modified: 2026-02-20 15:13:39; size: 5031; score: 23
  - imports: numpy, pandas, pathlib
  - outputs detected: D:\work1\task 3\02 grl\tdm\data\TNBC_AL.csv; D:\work1\task 3\02 grl\tdm\data\TNBC_AL_TDM.csv; D:\work1\task 3\02 grl\tdm\data\TNBC_AU.csv; out_path

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Methods: DART training (Stage0/Stage1/Stage2/GRL/EMA)

- `04_paper/shap/v4grl_v2_shap.py`
  - modified: 2026-03-05 14:42:00; size: 62600; score: 512
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, shap, sklearn, torch
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_l2sp_on.py`
  - modified: 2026-03-04 10:51:37; size: 49578; score: 451
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_paired.py`
  - modified: 2026-03-04 11:00:12; size: 54853; score: 450
  - imports: collections, copy, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full.py`
  - modified: 2026-03-04 00:10:41; size: 48645; score: 450
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot.py`
  - modified: 2026-03-04 22:59:04; size: 50143; score: 422
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2_fixed.py`
  - modified: 2026-03-04 23:21:55; size: 49443; score: 418
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2.py`
  - modified: 2026-03-04 23:05:54; size: 49275; score: 418
  - imports: collections, matplotlib, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_enplot.py`
  - modified: 2026-03-04 22:45:33; size: 48343; score: 409
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve.py`
  - modified: 2026-03-04 22:21:10; size: 48321; score: 407
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `04_paper/06_tf/allrun/v4grl_v2_stage2_history.py`
  - modified: 2026-03-19 00:26:42; size: 45202; score: 398
  - imports: collections, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Methods/Results: conventional ML baselines

- `04_paper/03_1_ml/domain_shift_visualization_pca_scores_font_big.py`
  - modified: 2026-04-19 21:09:59; size: 24034; score: 43
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, typing, warnings, xgboost
  - outputs detected:                  csv_dir / f"{name}_scores_oth_test.csv", index=False             ;                  csv_dir / f"{name}_scores_tnbc_test.csv", index=False             ; OTH.csv; TNBCalcom.csv; [OK] saved model summary -> csv/summary_domain_shift_models.csv
- `04_paper/03_1_ml/domain_shift_visualization_pca_scores.py`
  - modified: 2026-03-16 00:40:34; size: 23045; score: 43
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, typing, warnings, xgboost
  - outputs detected:                  csv_dir / f"{name}_scores_oth_test.csv", index=False             ;                  csv_dir / f"{name}_scores_tnbc_test.csv", index=False             ; OTH.csv; TNBCalcom.csv; [OK] saved model summary -> csv/summary_domain_shift_models.csv
- `04_paper/03_1_ml/ml_tdm_enhanced_ml_deg_baseline.py`
  - modified: 2026-04-29 23:52:55; size: 41736; score: 38
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `04_paper/03_1_ml/ml_tdm_enhanced_baseline.py`
  - modified: 2026-04-29 23:34:54; size: 37359; score: 38
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `04_paper/03_ml/nook/out_combat/model.py`
  - modified: 2026-02-26 13:19:20; size: 13960; score: 38
  - imports: argparse, joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected: TNBC.csv; oth.csv; roc_pdf; roc_png, dpi=200; roc_tnbc_all_models.pdf
- `04_paper/03_ml/ok/out_ml_baselines/model.py`
  - modified: 2026-02-24 21:46:50; size: 13960; score: 38
  - imports: argparse, joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected: TNBC.csv; oth.csv; roc_pdf; roc_png, dpi=200; roc_tnbc_all_models.pdf
- `04_paper/03_1_ml/ml_repeat_stability_paperready_font_red.py`
  - modified: 2026-04-19 17:09:27; size: 33321; score: 35
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `04_paper/03_1_ml/ml_repeat_stability_paperready.py`
  - modified: 2026-03-15 12:33:01; size: 33027; score: 35
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `04_paper/03_ml/model_refined.py`
  - modified: 2026-03-20 00:48:28; size: 19965; score: 34
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected:              pred_dir / f"{name}_OTH_test_scores.csv", index=False         ;              pred_dir / f"{name}_TNBC_test_scores.csv", index=False         ; OTH.csv; TNBCalcom.csv; cm_oth_test_{best}.pdf
- `04_paper/03_ml/out_ml_baselines_clickrun/model_refined.py`
  - modified: 2026-02-26 18:15:28; size: 21311; score: 34
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected:              pred_dir / f"{name}_OTH_test_scores.csv", index=False         ;              pred_dir / f"{name}_TNBC_test_scores.csv", index=False         ; OTH.csv; TNBCalcom.csv; metrics_heatmap_oth_test.pdf

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Figure 2

- `04_paper/03_1_ml/ml_tdm_enhanced_ml_deg_baseline.py`
  - modified: 2026-04-29 23:52:55; size: 41736; score: 82
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `04_paper/03_1_ml/ml_tdm_enhanced_baseline.py`
  - modified: 2026-04-29 23:34:54; size: 37359; score: 79
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv
- `04_paper/03_ml/out_ml_baselines_clickrun/model_refined.py`
  - modified: 2026-02-26 18:15:28; size: 21311; score: 56
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected:              pred_dir / f"{name}_OTH_test_scores.csv", index=False         ;              pred_dir / f"{name}_TNBC_test_scores.csv", index=False         ; OTH.csv; TNBCalcom.csv; metrics_heatmap_oth_test.pdf
- `04_paper/03_1_ml/domain_shift_visualization_pca_scores_font_big.py`
  - modified: 2026-04-19 21:09:59; size: 24034; score: 52
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, typing, warnings, xgboost
  - outputs detected:                  csv_dir / f"{name}_scores_oth_test.csv", index=False             ;                  csv_dir / f"{name}_scores_tnbc_test.csv", index=False             ; OTH.csv; TNBCalcom.csv; [OK] saved model summary -> csv/summary_domain_shift_models.csv
- `04_paper/03_ml/model_refined.py`
  - modified: 2026-03-20 00:48:28; size: 19965; score: 52
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected:              pred_dir / f"{name}_OTH_test_scores.csv", index=False         ;              pred_dir / f"{name}_TNBC_test_scores.csv", index=False         ; OTH.csv; TNBCalcom.csv; cm_oth_test_{best}.pdf
- `04_paper/03_1_ml/domain_shift_visualization_pca_scores.py`
  - modified: 2026-03-16 00:40:34; size: 23045; score: 52
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, typing, warnings, xgboost
  - outputs detected:                  csv_dir / f"{name}_scores_oth_test.csv", index=False             ;                  csv_dir / f"{name}_scores_tnbc_test.csv", index=False             ; OTH.csv; TNBCalcom.csv; [OK] saved model summary -> csv/summary_domain_shift_models.csv
- `02 grl/grl_1/uda_twoheads_full_clickrun_fixed_v3.py`
  - modified: 2026-01-26 11:55:45; size: 47564; score: 49
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected:              {                 "chosen_for_exports": chosen,                 "OTH_val": metrics_summary.get("OTH_val", {}; D:\work1\task 3\01 uda\uda_5\outputs_uda_click\genes.json; D:\work1\task 3\02 grl\grl_1\tdmdata\OTH_TDM.csv; Path(out_dir; checkpoint_uda_best_othval.pt
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/analysis3.ipynb`
  - modified: 2024-12-11 01:52:26; size: 1248528; score: 47
  - imports: lifelines, logging, matplotlib, matplotlib_venn, numpy, os, pandas, pgmpy, random, re, scipy, seaborn
  - outputs detected: "ipipd1_sur.png", dpi=1000; "ipipd1_sur.png",dpi=1000; "pd1_sur.png", dpi=1000; "pd1_sur.png",dpi=1000; "whole_sur.png", dpi=1000
- `02 grl/grl_1/uda_twoheads_full_clickrun_fixed_v2.py`
  - modified: 2026-01-15 21:54:13; size: 42028; score: 45
  - imports: json, math, matplotlib, numpy, pandas, pathlib, random, sklearn, torch
  - outputs detected:              {                 "chosen_for_exports": chosen,                 "OTH_val": metrics_summary.get("OTH_val", {}; D:\work1\task 3\01 uda\uda_5\outputs_uda_click\genes.json; D:\work1\task 3\02 grl\grl_1\int\TNBC_LA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULA_int.csv; D:\work1\task 3\02 grl\grl_1\int\TNBC_ULR_int.csv
- `04_paper/03_1_ml/ml_repeat_stability_paperready_font_red.py`
  - modified: 2026-04-19 17:09:27; size: 33321; score: 43
  - imports: __future__, dataclasses, joblib, json, lightgbm, math, matplotlib, numpy, pandas, pathlib, sklearn, typing
  - outputs detected:   - best_params_all_repeats.csv;   - repeat_metrics_all.csv;   - summary_mean_std_all_metrics.csv;   - tables/paper_table_oth.csv;   - tables/paper_table_tnbc.csv

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Figure 3

- `04_paper/shap/v4grl_v2_shap.py`
  - modified: 2026-03-05 14:42:00; size: 62600; score: 519
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, shap, sklearn, torch
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_paired.py`
  - modified: 2026-03-04 11:00:12; size: 54853; score: 453
  - imports: collections, copy, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_l2sp_on.py`
  - modified: 2026-03-04 10:51:37; size: 49578; score: 451
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full.py`
  - modified: 2026-03-04 00:10:41; size: 48645; score: 450
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot.py`
  - modified: 2026-03-04 22:59:04; size: 50143; score: 424
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2_fixed.py`
  - modified: 2026-03-04 23:21:55; size: 49443; score: 420
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2.py`
  - modified: 2026-03-04 23:05:54; size: 49275; score: 420
  - imports: collections, matplotlib, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_enplot.py`
  - modified: 2026-03-04 22:45:33; size: 48343; score: 411
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve.py`
  - modified: 2026-03-04 22:21:10; size: 48321; score: 409
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `04_paper/06_tf/allrun/v4grl_v2_stage2_history.py`
  - modified: 2026-03-19 00:26:42; size: 45202; score: 398
  - imports: collections, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Figure 4

- `04_paper/shap/v4grl_v2_shap.py`
  - modified: 2026-03-05 14:42:00; size: 62600; score: 519
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, shap, sklearn, torch
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_paired.py`
  - modified: 2026-03-04 11:00:12; size: 54853; score: 453
  - imports: collections, copy, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_l2sp_on.py`
  - modified: 2026-03-04 10:51:37; size: 49578; score: 451
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full.py`
  - modified: 2026-03-04 00:10:41; size: 48645; score: 450
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot.py`
  - modified: 2026-03-04 22:59:04; size: 50143; score: 424
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2_fixed.py`
  - modified: 2026-03-04 23:21:55; size: 49443; score: 420
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2.py`
  - modified: 2026-03-04 23:05:54; size: 49275; score: 420
  - imports: collections, matplotlib, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_enplot.py`
  - modified: 2026-03-04 22:45:33; size: 48343; score: 411
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve.py`
  - modified: 2026-03-04 22:21:10; size: 48321; score: 409
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `04_paper/06_tf/allrun/v4grl_v2_stage2_history.py`
  - modified: 2026-03-19 00:26:42; size: 45202; score: 398
  - imports: collections, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Figure 5 / SHAP

- `04_paper/shap/v4grl_v2_shap.py`
  - modified: 2026-03-05 14:42:00; size: 62600; score: 175
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, shap, sklearn, torch
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/analysis3.ipynb`
  - modified: 2024-12-11 01:52:26; size: 1248528; score: 132
  - imports: lifelines, logging, matplotlib, matplotlib_venn, numpy, os, pandas, pgmpy, random, re, scipy, seaborn
  - outputs detected: "ipipd1_sur.png", dpi=1000; "ipipd1_sur.png",dpi=1000; "pd1_sur.png", dpi=1000; "pd1_sur.png",dpi=1000; "whole_sur.png", dpi=1000
- `04_paper/shap/shap_stage2_agg/make_stage2_agg_beeswarm.py`
  - modified: 2026-03-05 18:42:15; size: 5959; score: 57
  - imports: argparse, glob, matplotlib, numpy, pandas, pathlib, shap
  - outputs detected: *_shap_values.npz; out_path, bbox_inches="tight", dpi=args.dpi; stage2_final_agg_beeswarm_top{topn}_dpi{args.dpi}.pdf; stage2_final_agg_shap_meanabs.csv
- `04_paper/06_tf/duibi/irnet/predict.py`
  - modified: 2026-03-19 01:53:32; size: 23398; score: 49
  - imports: argparse, logging, numpy, os, pandas, random, re, scipy, sklearn, spektral, tensorflow, tensorflow_addons
  - outputs detected: ./Kegg/human_KeggPathwayNet.txt; ./genelist_8080.txt; ./prediction_results.txt; Pathway_relation.csv; Pathway_weight.csv
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/predict.py`
  - modified: 2024-12-11 01:52:26; size: 23398; score: 49
  - imports: argparse, logging, numpy, os, pandas, random, re, scipy, sklearn, spektral, tensorflow, tensorflow_addons
  - outputs detected: ./Kegg/human_KeggPathwayNet.txt; ./genelist_8080.txt; ./prediction_results.txt; Pathway_relation.csv; Pathway_weight.csv
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/Clean_train.ipynb`
  - modified: 2024-12-11 01:52:26; size: 112737; score: 45
  - imports: logging, math, numpy, os, pandas, random, re, scipy, sklearn, spektral, tensorflow, tensorflow_addons
  - outputs detected: ./genelist_8080.txt; _log.txt; _zscore.npz; immunotherapy_IMPACT_onehotlabel_kegg_pathgraph_Checkmate025_Niv_zscore.npz; immunotherapy_IMPACT_onehotlabel_kegg_pathgraph_GSE176307_zscore.npz
- `04_paper/shap/shap_stage2_agg/make_stage2_single_beeswarm_hd.py`
  - modified: 2026-03-05 18:52:15; size: 3486; score: 40
  - imports: argparse, matplotlib, numpy, pathlib, shap
  - outputs detected: out_pdf, bbox_inches="tight", dpi=args.dpi; stage2_final_single_beeswarm_top{args.topn}_dpi{args.dpi}.pdf
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/process_TideBulk3.ipynb`
  - modified: 2024-12-11 01:52:26; size: 72926; score: 39
  - imports: numpy, pandas, scipy, sklearn, sksurv
  - outputs detected: 'test2.csv', index=False, columns=col_sym; -TMM.txt; -clinic.csv; ./Kegg/human_KeggPathwayNet.txt; ./genelist_8080.txt
- `04_paper/05_sc/GSE118389/step5_redraw_celltype_and_plot_genes_modules.py`
  - modified: 2026-03-08 20:10:31; size: 12382; score: 34
  - imports: matplotlib, numpy, pandas, scanpy, scipy
  - outputs detected: "candidate_gene_mean_by_major_celltype.csv"; "dotplot_priority_genes_by_major_celltype.pdf"; "dotplot_priority_genes_by_major_celltype.png", dpi=300; "heatmap_module_scores_by_major_celltype.pdf", bbox_inches="tight"; "heatmap_module_scores_by_major_celltype.png", dpi=300, bbox_inches="tight"
- `04_paper/03_ml/out_ml_baselines_clickrun/model_refined.py`
  - modified: 2026-02-26 18:15:28; size: 21311; score: 29
  - imports: joblib, lightgbm, matplotlib, numpy, pandas, pathlib, sklearn, warnings, xgboost
  - outputs detected:              pred_dir / f"{name}_OTH_test_scores.csv", index=False         ;              pred_dir / f"{name}_TNBC_test_scores.csv", index=False         ; OTH.csv; TNBCalcom.csv; metrics_heatmap_oth_test.pdf

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Figure 6 / GO KEGG ssGSEA

- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/analysis3.ipynb`
  - modified: 2024-12-11 01:52:26; size: 1248528; score: 101
  - imports: lifelines, logging, matplotlib, matplotlib_venn, numpy, os, pandas, pgmpy, random, re, scipy, seaborn
  - outputs detected: "ipipd1_sur.png", dpi=1000; "ipipd1_sur.png",dpi=1000; "pd1_sur.png", dpi=1000; "pd1_sur.png",dpi=1000; "whole_sur.png", dpi=1000
- `05_bio/run_ch4_python_pipeline.py`
  - modified: 2026-04-06 01:27:48; size: 27752; score: 95
  - imports: __future__, argparse, gprofiler, gseapy, math, matplotlib, numpy, pandas, pathlib, re, scipy, sys
  - outputs detected: GO_BP_prerank_NES_barplot.png; GO_BP_prerank_results.csv; KEGG_prerank_NES_barplot.png; KEGG_prerank_results.csv; MAD_HVG800_background.csv
- `05_bio/ch4_redo/run_ch4_GO_KEGG_GSEA_pipeline.R`
  - modified: 2026-04-06 01:18:20; size: 9732; score: 59
  - imports: GSVA, clusterProfiler, enrichplot, fgsea, org.Hs.eg.db, patchwork, pheatmap, tidyverse
  - outputs detected: Fig4_3_Top200_GO_KEGG_ORA.pdf; Fig4_4_GSEA_GO_KEGG_summary.pdf; Fig4_5_Top200_vs_Top100_ORA.pdf; Fig4_6_ssGSEA_top_signatures_boxplot.pdf; GSEA_GO_BP_fullrank.csv
- `04_paper/ssGSEA/GSVA/run_tnbc_gsva_python_final_int_ticks.py`
  - modified: 2026-04-20 10:01:07; size: 20667; score: 50
  - imports: argparse, collections, matplotlib, numpy, os, pandas, re, scipy, textwrap
  - outputs detected:          os.path.join(args.outdir, "all_methods_summary_stats.csv"; all_methods_summary_stats.csv; long_path, index=False; os.path.join(outdir, f"{plot_name}_scores_wide.csv"; os.path.join(outdir, f"{plot_name}_stats.csv"
- `04_paper/ssGSEA/GSVA/run_tnbc_gsva_python_final.py`
  - modified: 2026-04-20 09:43:56; size: 20421; score: 50
  - imports: argparse, collections, matplotlib, numpy, os, pandas, re, scipy, textwrap
  - outputs detected:          os.path.join(args.outdir, "all_methods_summary_stats.csv"; all_methods_summary_stats.csv; long_path, index=False; os.path.join(outdir, f"{plot_name}_scores_wide.csv"; os.path.join(outdir, f"{plot_name}_stats.csv"
- `04_paper/ssGSEA/run_ssgsea.py`
  - modified: 2026-03-25 15:22:14; size: 17214; score: 41
  - imports: argparse, gseapy, math, matplotlib, numpy, pandas, pathlib, scipy
  - outputs detected:          outdir / "ssgsea_group_comparison.csv", index=False     ;          outdir / "ssgsea_pred_correlation.csv", index=False     ; boxplot_{mod}.pdf; boxplot_{mod}.png; outdir / "ssgsea_module_heatmap.pdf", bbox_inches="tight"
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/Clean_train.ipynb`
  - modified: 2024-12-11 01:52:26; size: 112737; score: 39
  - imports: logging, math, numpy, os, pandas, random, re, scipy, sklearn, spektral, tensorflow, tensorflow_addons
  - outputs detected: ./genelist_8080.txt; _log.txt; _zscore.npz; immunotherapy_IMPACT_onehotlabel_kegg_pathgraph_Checkmate025_Niv_zscore.npz; immunotherapy_IMPACT_onehotlabel_kegg_pathgraph_GSE176307_zscore.npz
- `04_paper/06_tf/duibi/irnet/IRnet-main/IRnet-main/training files/process_TideBulk3.ipynb`
  - modified: 2024-12-11 01:52:26; size: 72926; score: 39
  - imports: numpy, pandas, scipy, sklearn, sksurv
  - outputs detected: 'test2.csv', index=False, columns=col_sym; -TMM.txt; -clinic.csv; ./Kegg/human_KeggPathwayNet.txt; ./genelist_8080.txt
- `04_paper/05_sc/GSE118389/step5_redraw_celltype_and_plot_genes_modules.py`
  - modified: 2026-03-08 20:10:31; size: 12382; score: 32
  - imports: matplotlib, numpy, pandas, scanpy, scipy
  - outputs detected: "candidate_gene_mean_by_major_celltype.csv"; "dotplot_priority_genes_by_major_celltype.pdf"; "dotplot_priority_genes_by_major_celltype.png", dpi=300; "heatmap_module_scores_by_major_celltype.pdf", bbox_inches="tight"; "heatmap_module_scores_by_major_celltype.png", dpi=300, bbox_inches="tight"
- `04_paper/ssGSEA/GSVA/run_tnbc_gsva_python.py`
  - modified: 2026-04-20 09:02:17; size: 8715; score: 29
  - imports: argparse, matplotlib, numpy, os, pandas, re, scipy
  - outputs detected:          os.path.join(args.outdir, "tnbc_gsva_ssgsea_summary.csv"; os.path.join(args.outdir, "tnbc_gsva_scores_direct_ecdf.csv"; os.path.join(args.outdir, "tnbc_gsva_scores_gaussian_approx.csv"; os.path.join(args.outdir, "tnbc_ssgsea_scores_python.csv"; tnbc_gsva_scores_direct_ecdf.csv

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

## Supplementary tables

- `04_paper/shap/v4grl_v2_shap.py`
  - modified: 2026-03-05 14:42:00; size: 62600; score: 680
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, shap, sklearn, torch
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_l2sp_on.py`
  - modified: 2026-03-04 10:51:37; size: 49578; score: 451
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full_paired.py`
  - modified: 2026-03-04 11:00:12; size: 54853; score: 450
  - imports: collections, copy, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_ablation_full.py`
  - modified: 2026-03-04 00:10:41; size: 48645; score: 450
  - imports: collections, copy, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot.py`
  - modified: 2026-03-04 22:59:04; size: 50143; score: 422
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2_fixed.py`
  - modified: 2026-03-04 23:21:55; size: 49443; score: 418
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_paperplot_v2.py`
  - modified: 2026-03-04 23:05:54; size: 49275; score: 418
  - imports: collections, matplotlib, numpy, os, pandas, pathlib, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve_enplot.py`
  - modified: 2026-03-04 22:45:33; size: 48343; score: 409
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `02 grl/okre/xiaorong/v4grl_v2_with_traincurve.py`
  - modified: 2026-03-04 22:21:10; size: 48321; score: 407
  - imports: collections, json, math, matplotlib, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str
- `04_paper/06_tf/allrun/v4grl_v2_stage2_history.py`
  - modified: 2026-03-19 00:26:42; size: 45202; score: 398
  - imports: collections, json, math, numpy, os, pandas, pathlib, random, sklearn, torch, typing
  - outputs detected: /root/grl/data/BRA_unlabeled_std.csv; /root/grl/data/OTH_labeled_std.csv; /root/grl/data/TNBC_labeled_std.csv; /root/grl/data/genes.json; CFG, f, indent=2, default=str

Most likely final-version rule: prefer paths containing `04_paper`, `paperready`, `out_*_0419`, `out_*_0429`, `04_tf`, `shap_stage2`, or manuscript plotting scripts copied from the LaTeX workspace. This is an inference and should be checked manually.

