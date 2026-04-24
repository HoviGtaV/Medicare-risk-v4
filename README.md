# Medicare Provider Exclusion Risk Project - Active v4

## Overview

This repository contains the active **v4** pipeline for ranking Medicare provider-year records by exclusion risk using CMS Medicare Physician & Other Practitioners by Provider data together with OIG LEIE exclusion history.

The output is a **risk score for review prioritization**, not a legal finding of fraud, guilt, or intent.

This repository has been cleaned so that:

- **v4** is the active execution and submission path
- historical **v3** materials are preserved only for auditability and documentation
- active-path v3 execution references have been removed from the live project path

Historical material is intentionally retained under:

- `docs/history_reference`
- `src/legacy`
- `_archive_local_not_for_git`

---

## Project links

### GitHub repository

- `https://github.com/HoviGtaV/Medicare-risk-v4`

### Public Docker image

- `hovig2004/medicare-exclusion-risk-v4:latest`

### Docker Hub

- `https://hub.docker.com/r/hovig2004/medicare-exclusion-risk-v4`

---

## Project objective

The project predicts whether a provider-year record is followed by an OIG exclusion event within a fixed future horizon.

### Unit of analysis

- one row = one provider-year
- provider identifier = `npi_clean`
- year field = `summary_year`

### Target

- `target_excluded_24m`

This target is an exclusion-based proxy label for downstream risk ranking.

For the formal labeling rules, caveats, and interpretation limits, see:

- `submission/docs/LABEL_POLICY_APPENDIX.md`

---

## Active v4 pipeline

### 1. Cleaning and standardization

- `src/cleaning/build_cms_all_years_clean.py`
- `src/cleaning/Column_Cleaning.py`
- `src/cleaning/schema_audit.py`

### 2. Label construction

- `src/labels/build_exclusion_history.py`
- `src/labels/build_provider_year_joined.py`
- `src/labels/build_provider_year_labeled.py`

### 3. Feature engineering

- `build_provider_year_features_v4.py`

Primary artifact:

- `data/processed/provider_year_features_v4.parquet`

### 4. Time-based split creation

- `make_v4_splits.py`

Primary artifacts:

- `data/processed/train_v4.parquet`
- `data/processed/valid_v4.parquet`
- `data/processed/test_v4.parquet`

### 5. Baseline model

- `train_logreg_v4.py`

Primary artifacts:

- `models_v4/logreg/model.joblib`
- `models_v4/logreg/metrics_valid.json`
- `models_v4/logreg/valid_scored.parquet`
- `models_v4/logreg/coefficients.csv`

### 6. Main validation model

- `train_catboost_v4.py`

Primary artifacts:

- `models_v4/catboost/model.cbm`
- `models_v4/catboost/metrics_valid.json`
- `models_v4/catboost/valid_scored.parquet`
- `models_v4/catboost/feature_importance.csv`

### 7. Final model retrain and holdout test

- `retrain_and_test_final_catboost_v4.py`

Primary artifacts:

- `models_v4/final/model.cbm`
- `models_v4/final/metrics_test.json`
- `models_v4/final/test_scored.parquet`
- `models_v4/final/feature_importance.csv`

### 8. Ranking analysis and explainability

- `analyze_topk_v4.py`
- `explain_shap_v4.py`
- `make_model_comparison_v4.py`

Primary artifacts:

- `outputs/metrics_v4/topk_summary_v4.csv`
- `outputs/metrics_v4/top500_provider_type_mix_v4.csv`
- `outputs/metrics_v4/top1000_provider_type_mix_v4.csv`
- `outputs/metrics_v4/model_comparison_v4.csv`
- `outputs/shap_v4/global_shap_importance.csv`
- `outputs/shap_v4/shap_summary_bar.png`
- `outputs/shap_v4/shap_beeswarm.png`
- `outputs/shap_v4/local_explanations_top5.csv`

### 9. Submission, QA, and monitoring

- `submission/tools/build_full_data_dictionary_v4.py`
- `src/qa/check_final_feature_tables_v4.py`
- `src/qa/check_final_labels_v4.py`
- `src/qa/check_oig_date_consistency_v4.py`
- `submission/tools/make_monitoring_artifacts.py`
- `submission/tools/run_submission_qc.py`
- `submission/tools/run_submission_qc_final.py`

---

## Time-based evaluation design

The active evaluation design is:

- Train: `2018-2021`
- Validation: `2022`
- Final train: `2018-2022`
- Holdout test: `2023`

This preserves temporal order and keeps final evaluation on a future holdout period.

---

## Active data and QA summary

From the current v4 QA outputs:

### Modeling splits

- `train_v4`: 4,637,638 rows, 243 columns, 516 positives
- `valid_v4`: 1,230,288 rows, 243 columns, 168 positives
- `test_v4`: 1,259,335 rows, 243 columns, 151 positives

### Full labeled table

- total labeled rows: `8,215,957`
- total positives: `993`

### Duplicate provider-year rows after split preparation

- train: `49`
- validation: `4`
- test: `2`

### OIG date consistency QA

- missing exclusion-date rows: `0`
- reindate-before-excldate violations: `0`
- waiverdate-before-excldate violations: `0`

---

## Model results

### Validation 2022

#### Logistic baseline

- AUPRC: `0.0013735`
- ROC AUC: `0.8187673`
- Precision@500: `0.006`
- Precision@1000: `0.003`

#### CatBoost v4

- AUPRC: `0.0034991`
- ROC AUC: `0.8698072`
- Precision@500: `0.016`
- Precision@1000: `0.013`
- Lift@500: `117.1703`
- Lift@1000: `95.2009`

### Final 2023 holdout test

#### Final CatBoost v4

- AUPRC: `0.0027841`
- ROC AUC: `0.8208614`
- Precision@500: `0.010`
- Precision@1000: `0.010`
- Lift@500: `83.3997`
- Lift@1000: `83.3997`

### Top-K ranking summary on 2023

From `outputs/metrics_v4/topk_summary_v4.csv`:

- Top 100: `1` positive, precision `0.0100`
- Top 500: `5` positives, precision `0.0100`
- Top 1000: `10` positives, precision `0.0100`
- Top 5000: `27` positives, precision `0.0054`

---

## Explainability

Global SHAP artifacts were generated for the active v4 model.

Top features in the current global SHAP summary include:

- `bene_cc_ph_cancer6_v2_pct`
- `ph_condition_pct_mean_specialty_year_median`
- `srvcs_per_bene_lag1`
- `bene_dual_share_specialty_year_median`
- `stdzd_amt_per_bene_specialty_year_median`

Artifacts:

- `outputs/shap_v4/global_shap_importance.csv`
- `outputs/shap_v4/shap_summary_bar.png`
- `outputs/shap_v4/shap_beeswarm.png`
- `outputs/shap_v4/local_explanations_top5.csv`

---

## Monitoring

Monitoring artifacts were generated from the active final model using the validation and holdout splits.

Artifacts:

- `submission/monitoring/score_drift_summary.csv`
- `submission/monitoring/feature_drift_summary.csv`
- `submission/monitoring/score_distribution_2022_vs_2023.png`
- `submission/monitoring/MONITORING_NOTE.md`

These monitoring outputs were generated from:

- `models_v4/final/model.cbm`
- `data/processed/valid_v4.parquet`
- `data/processed/test_v4.parquet`

---

## Batch scoring

The current deployment story is strongest for **batch scoring** of an already prepared provider-year feature table.

That means:

- input = a CSV or parquet file with the required v4 feature columns
- output = scored rows with `score_catboost_final_v4`

### Local example

```powershell
python .\score_batch_v4.py --input .\examples\sample_input_v4.csv --output .\examples\sample_output_v4.csv
```

### Docker example

```powershell
docker pull hovig2004/medicare-exclusion-risk-v4:latest
docker run --rm -v "${PWD}\examples:/app/examples" hovig2004/medicare-exclusion-risk-v4:latest python score_batch_v4.py --input /app/examples/sample_input_v4.csv --output /app/examples/sample_output_docker_v4.csv
```

### Optional local Docker build

```powershell
docker build --no-cache -t medicare-exclusion-risk-v4 .
```

---

## Submission materials included in this repository

- one-page report: `submission/report/report_one_page_final.md`
- final presentation slides: `submission/slides/presentation_final_content.md`
- video demo script: `submission/video/video_demo_script_final.md`
- pipeline trace: `submission/docs/PIPELINE_TRACE.md`
- label policy appendix: `submission/docs/LABEL_POLICY_APPENDIX.md`
- full data dictionary: `submission/docs/DATA_DICTIONARY_V4_FULL.md`
- monitoring artifacts: `submission/monitoring`
- QA outputs: `submission/qa`

---

## Important interpretation note

This project produces a **risk ranking score** based on exclusion-linked proxy labels.

It should be used for:

- review prioritization
- ranking
- monitoring
- analytical follow-up

It should **not** be interpreted as:

- proof of fraud
- proof of guilt
- proof of criminal intent
- an automatic enforcement decision
