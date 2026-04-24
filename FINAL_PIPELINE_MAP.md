# Final Pipeline Map - Active v4

## Project status

This repository now uses the **v4 pipeline** as the active production / submission path.

Historical v3 materials are intentionally preserved for auditability and documentation in:

- `docs/history_reference`
- `src/legacy`
- `_archive_local_not_for_git`

These historical materials are **not** the active execution path.

---

## Active end-to-end pipeline

### 1. Raw source inputs
- CMS Medicare Physician & Other Practitioners by Provider annual files
- OIG LEIE exclusion file
- CMS methodology / data dictionary reference files

### 2. Cleaning
- `src/cleaning/build_cms_all_years_clean.py`
- `src/cleaning/Column_Cleaning.py`
- `src/cleaning/schema_audit.py`

### 3. Label-building
- `src/labels/build_exclusion_history.py`
- `src/labels/build_provider_year_joined.py`
- `src/labels/build_provider_year_labeled.py`

### 4. Active feature engineering
- `build_provider_year_features_v4.py`

Primary output:
- `data/processed/provider_year_features_v4.parquet`

### 5. Time-based split creation
- `make_v4_splits.py`

Primary outputs:
- `data/processed/train_v4.parquet`
- `data/processed/valid_v4.parquet`
- `data/processed/test_v4.parquet`

### 6. Baseline model
- `train_logreg_v4.py`

Primary outputs:
- `models_v4/logreg/model.joblib`
- `models_v4/logreg/metrics_valid.json`
- `models_v4/logreg/valid_scored.parquet`

### 7. Validation model
- `train_catboost_v4.py`

Primary outputs:
- `models_v4/catboost/model.cbm`
- `models_v4/catboost/metrics_valid.json`
- `models_v4/catboost/valid_scored.parquet`

### 8. Final model retrain and 2023 evaluation
- `retrain_and_test_final_catboost_v4.py`

Primary outputs:
- `models_v4/final/model.cbm`
- `models_v4/final/metrics_test.json`
- `models_v4/final/test_scored.parquet`

### 9. Ranking analysis
- `analyze_topk_v4.py`

Primary outputs:
- `outputs/metrics_v4/topk_summary_v4.csv`
- `outputs/metrics_v4/top500_provider_type_mix_v4.csv`
- `outputs/metrics_v4/top1000_provider_type_mix_v4.csv`

### 10. Explainability
- `explain_shap_v4.py`

Primary outputs:
- `outputs/shap_v4/global_shap_importance.csv`
- `outputs/shap_v4/shap_summary_bar.png`
- `outputs/shap_v4/shap_beeswarm.png`
- `outputs/shap_v4/local_explanations_top5.csv`

### 11. Model comparison
- `make_model_comparison_v4.py`

Primary output:
- `outputs/metrics_v4/model_comparison_v4.csv`

### 12. Batch scoring
- `score_batch_v4.py`

Example usage:
- input: `examples/sample_input_v4.csv`
- output: `examples/sample_output_v4.csv`

### 13. Submission / QA / monitoring artifacts
- `submission/tools/build_full_data_dictionary_v4.py`
- `src/qa/check_final_feature_tables_v4.py`
- `src/qa/check_final_labels_v4.py`
- `src/qa/check_oig_date_consistency_v4.py`
- `submission/tools/make_monitoring_artifacts.py`
- `submission/tools/run_submission_qc.py`
- `submission/tools/run_submission_qc_final.py`

---

## Active evaluation windows

- Train: `2018-2021`
- Validation: `2022`
- Final train: `2018-2022`
- Test: `2023`

---

## Active model family

- Baseline: logistic regression
- Main model: CatBoost
- Final deployed scoring script: `score_batch_v4.py`

---

## Current active artifact roots

- models: `models_v4`
- metrics: `outputs/metrics_v4`
- shap: `outputs/shap_v4`

---

## Historical note

All v3 scripts, v3 outputs, v3 snapshots, and duplicate archived project copies were moved out of the active path on purpose. They remain available only for traceability, audit, and documentation.
