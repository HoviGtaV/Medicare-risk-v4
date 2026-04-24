# Pipeline Trace - Active v4

## Purpose

This document traces the active v4 pipeline from raw source files to final scoring outputs.

Historical v3 material is preserved separately under:

- `docs/history_reference`
- `src/legacy`
- `_archive_local_not_for_git`

Those historical paths are retained for documentation and auditability only.

---

## A. Raw inputs

### CMS source
Annual Medicare Physician & Other Practitioners by Provider files stored under:
- `data/raw/cms`

### OIG source
LEIE exclusion source stored under:
- `data/raw/oig/UPDATED.csv`

### Reference material
Documentation and schema references stored under:
- `data/raw/reference`

---

## B. Cleaning and standardization

### CMS cleaning
- `src/cleaning/build_cms_all_years_clean.py`
- `src/cleaning/Column_Cleaning.py`
- `src/cleaning/schema_audit.py`

Purpose:
- standardize CMS columns across years
- normalize naming
- prepare downstream joins

---

## C. Label construction

### Exclusion history
- `src/labels/build_exclusion_history.py`

Purpose:
- clean OIG exclusion history
- standardize dates
- prepare exclusion windows

### Provider-year joining
- `src/labels/build_provider_year_joined.py`

Purpose:
- join cleaned CMS provider-year records with exclusion history support fields

### Final labeled provider-year table
- `src/labels/build_provider_year_labeled.py`

Purpose:
- create the provider-year modeling table with target labels

Primary labeled artifact:
- `data/processed/provider_year_labeled.parquet`

---

## D. Active v4 feature generation

### Feature script
- `build_provider_year_features_v4.py`

Primary output:
- `data/processed/provider_year_features_v4.parquet`

Feature families include:
- raw utilization and payment totals
- ratio and intensity features
- drug / medical mix features
- demographic share features
- chronic-condition summary features
- specialty-year peer-relative features
- provider temporal lag and change features
- rarity flags

---

## E. Time-based split design

### Split script
- `make_v4_splits.py`

Outputs:
- `data/processed/train_v4.parquet`
- `data/processed/valid_v4.parquet`
- `data/processed/test_v4.parquet`

Time windows:
- train = 2018-2021
- validation = 2022
- test = 2023

---

## F. Baseline model

### Logistic regression baseline
- `train_logreg_v4.py`

Outputs:
- `models_v4/logreg/model.joblib`
- `models_v4/logreg/metrics_valid.json`
- `models_v4/logreg/valid_scored.parquet`
- `models_v4/logreg/coefficients.csv`

Purpose:
- provide a simple interpretable baseline for comparison against CatBoost

---

## G. Validation model

### CatBoost validation model
- `train_catboost_v4.py`

Outputs:
- `models_v4/catboost/model.cbm`
- `models_v4/catboost/metrics_valid.json`
- `models_v4/catboost/valid_scored.parquet`
- `models_v4/catboost/feature_importance.csv`

Purpose:
- train on 2018-2021
- validate on 2022
- choose the main nonlinear ranking model

---

## H. Final model retrain and test

### Final CatBoost
- `retrain_and_test_final_catboost_v4.py`

Outputs:
- `models_v4/final/model.cbm`
- `models_v4/final/metrics_test.json`
- `models_v4/final/test_scored.parquet`
- `models_v4/final/feature_importance.csv`

Purpose:
- retrain on 2018-2022
- evaluate once on 2023

---

## I. Ranking analysis

### Top-K analysis
- `analyze_topk_v4.py`

Outputs:
- `outputs/metrics_v4/topk_summary_v4.csv`
- `outputs/metrics_v4/top500_provider_type_mix_v4.csv`
- `outputs/metrics_v4/top1000_provider_type_mix_v4.csv`

Purpose:
- evaluate ranking usefulness at operational review cutoffs

---

## J. Explainability

### SHAP explanations
- `explain_shap_v4.py`

Outputs:
- `outputs/shap_v4/global_shap_importance.csv`
- `outputs/shap_v4/shap_summary_bar.png`
- `outputs/shap_v4/shap_beeswarm.png`
- `outputs/shap_v4/local_explanations_top5.csv`

Purpose:
- explain global feature impact
- inspect local high-risk examples

---

## K. Model comparison

### Comparison summary
- `make_model_comparison_v4.py`

Output:
- `outputs/metrics_v4/model_comparison_v4.csv`

Purpose:
- compare logistic baseline, validation CatBoost, and final test-stage CatBoost

---

## L. Batch scoring

### Production-style scoring
- `score_batch_v4.py`

Example:
- input: `examples/sample_input_v4.csv`
- output: `examples/sample_output_v4.csv`

Purpose:
- score already-prepared provider-year feature tables with the final v4 model

---

## M. QA and monitoring

### QA scripts
- `src/qa/check_final_feature_tables_v4.py`
- `src/qa/check_final_labels_v4.py`
- `src/qa/check_oig_date_consistency_v4.py`

Outputs:
- `submission/qa/check_final_feature_tables_v4.csv`
- `submission/qa/check_final_labels_v4.csv`
- `submission/qa/check_oig_date_consistency_v4.csv`

### Monitoring artifacts
- `submission/tools/make_monitoring_artifacts.py`

Outputs:
- `submission/monitoring/score_drift_summary.csv`
- `submission/monitoring/feature_drift_summary.csv`
- `submission/monitoring/score_distribution_2022_vs_2023.png`
- `submission/monitoring/MONITORING_NOTE.md`

### Submission QC
- `submission/tools/run_submission_qc.py`
- `submission/tools/run_submission_qc_final.py`

Outputs:
- `submission/qa/final_submission_checklist.csv`
- `submission/qa/hard_deliverables_check.csv`

---

## N. Current active artifact summary

### Active models
- `models_v4/logreg`
- `models_v4/catboost`
- `models_v4/final`

### Active metrics and analysis
- `outputs/metrics_v4`
- `outputs/shap_v4`

### Active submission deliverables
- `submission/docs`
- `submission/monitoring`
- `submission/qa`
- `submission/report`
- `submission/slides`
- `submission/video`

---

## O. Historical preservation policy

v3 scripts, outputs, samples, and snapshots were intentionally removed from the active execution path and preserved in history/archive locations rather than deleted.

This preserves:
- reproducibility context
- audit trail
- documentation continuity
- comparison against older versions
