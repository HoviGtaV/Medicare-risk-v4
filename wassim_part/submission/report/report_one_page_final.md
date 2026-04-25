# Final One-Page Report

## Team Members
- Wassim Mouawad
- Hovig Majarian
- Rayan Salman

## Project Title
Insurance Fraud Project — Provider-Level Medicare Risk Scoring from Claims Summaries and Exclusion Labels

## Data Source
This project uses public CMS Medicare provider summary data for years 2017-2023 and OIG LEIE exclusion data. The unit of analysis is the provider-year.

## Problem Framing
The system is designed as a provider risk-scoring model, not a legal fraud detector. The target label is a proxy: whether a provider is excluded within 24 months after the provider-year summary end. Because exclusions are imperfect proxies for fraud, the output must be interpreted as risk ranking rather than legal proof.

## Approach
We cleaned and joined CMS and OIG data, built a deterministic 24-month forward exclusion label, created provider-year features, and trained models using a time-based split. Features included payment totals, utilization counts, per-beneficiary and per-service ratios, specialty-year median comparisons, specialty-year percentile ranks, log features, and rarity flags.

## Methods Used
- Logistic Regression baseline
- CatBoost main model
- SHAP explainability
- Batch scoring deployment script
- Docker packaging
- Simple drift-monitoring artifact

## Final Split Design
- Train: 2018-2021
- Validation: 2022
- Final train for chosen model: 2018-2022
- Final untouched test: 2023

## Final Result
Official final model: CatBoost

2023 test performance:
- AUPRC: 0.0006626745788939973
- ROC-AUC: 0.7676166053703076
- Precision@1000: 0.001
- Precision@5000: 0.0012
- Lift@1000: 8.395553333333334

## Interpretation
The final model captures real ranking signal above random selection in a highly imbalanced and noisy proxy-label setting. However, very small top-K precision remains limited. Therefore, the system is best interpreted as a provider risk-ranking prototype for review prioritization, not a definitive fraud detection system.


