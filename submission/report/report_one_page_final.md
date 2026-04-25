# Medicare Provider Exclusion Risk Scoring — One-Page Project Report

## Team Members
- Hovig Madjarian
- Rayan Salman
- Wassim Mouawad

## Data Sources
This project uses two public data sources:
1. **CMS Medicare Physician & Other Practitioners by Provider** — provider-level summary data with utilization, payment, and beneficiary-related measures.
2. **OIG List of Excluded Individuals and Entities (LEIE)** — used to build a supervised proxy label based on later exclusion events.

## Project Goal
The goal of this project is to build a **provider-level Medicare risk scoring system** that ranks provider-year records by exclusion-related risk. The model does **not** predict legal fraud directly and should **not** be interpreted as proof of guilt or criminal intent. Instead, it produces a **risk score for review prioritization** using public Medicare summaries and exclusion-based proxy labels.

## Problem Framing
The target is `target_excluded_24m`, which labels a provider-year as positive if the provider is excluded within 24 months after the end of that provider-year. This makes the task a **proxy-label prediction problem**, not a perfect fraud-classification problem. The project is also highly imbalanced, because positive cases are extremely rare relative to the full dataset.

## Approach
We built an end-to-end v4 pipeline with the following stages:
- cleaning and standardization of CMS provider data
- construction of exclusion-history labels from OIG LEIE
- provider-year feature engineering
- time-based train/validation/test split design
- logistic regression baseline
- CatBoost main model
- SHAP explainability
- batch scoring and simple drift monitoring

The final evaluation design was:
- **Train:** 2018–2021
- **Validation:** 2022
- **Final holdout test:** 2023

## Methods Used
The dataset contained:
- **Train:** 4,637,638 rows, 243 columns, 516 positives
- **Validation:** 1,230,288 rows, 243 columns, 168 positives
- **Test:** 1,259,335 rows, 243 columns, 151 positives

Because the full logistic regression baseline was too slow and did not converge well, we redesigned it as a valid imbalance-aware baseline:
- keep **all positive cases**
- downsample **negative cases in training only**
- evaluate on the **full untouched validation set**

The final main model was **CatBoost**, which was better suited for structured tabular data and nonlinear interactions. Feature engineering in v4 improved over v3 by adding:
- richer beneficiary-mix features
- chronic burden summaries
- drug vs medical composition
- beneficiary risk score interactions
- provider-history temporal features
- lag, year-over-year, and spike features
- stronger peer-relative context

## Key Results
### Logistic Regression Baseline (Validation 2022)
- **AUPRC:** 0.0013735
- **ROC-AUC:** 0.8187673
- **Precision@500:** 0.006
- **Precision@1000:** 0.003

### CatBoost Main Model
**Validation 2022**
- **AUPRC:** 0.0034991
- **ROC-AUC:** 0.8698072
- **Precision@500:** 0.016
- **Precision@1000:** 0.013
- **Lift@500:** 117.17
- **Lift@1000:** 95.20

**Final Holdout Test 2023**
- **AUPRC:** 0.0027841
- **ROC-AUC:** 0.8208614
- **Precision@500:** 0.010
- **Precision@1000:** 0.010
- **Lift@500:** 83.40
- **Lift@1000:** 83.40

### v3 vs v4 Improvement
Compared with the archived v3 final test, v4 showed a clear improvement in ranking quality:
- **v3 AUPRC:** 0.0006627 → **v4 AUPRC:** 0.0027841
- **v3 ROC-AUC:** 0.7676 → **v4 ROC-AUC:** 0.8209
- **v3 Precision@1000:** 0.001 → **v4 Precision@1000:** 0.010

## Interpretation and Final Deliverables
The final system includes:
- a batch scorer (`score_batch_v4.py`)
- a Dockerized scoring workflow
- SHAP explainability outputs
- monitoring artifacts for score and feature drift
- GitHub repository with reproducible code

This project should be interpreted as an **exclusion-risk ranking system for review prioritization**, not as a legal fraud detector.
