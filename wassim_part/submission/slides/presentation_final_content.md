# Final Presentation Content

## Slide 1 — Title
Insurance Fraud Project — Provider-Level Medicare Risk Scoring

## Slide 2 — Problem Framing
- Risk score, not legal fraud proof
- Proxy labels, not perfect ground truth
- Output is for ranking and review support

## Slide 3 — Data Sources
- CMS Medicare provider summary data
- OIG LEIE exclusions
- Provider-year unit of analysis

## Slide 4 — Label Policy
- `target_excluded_24m`
- positive if exclusion occurs within 24 months after year end
- same-year or earlier exclusions removed
- incomplete follow-up removed

## Slide 5 — Feature Engineering
- raw totals
- per-beneficiary ratios
- per-service ratios
- structural payment ratios
- specialty-year medians
- percentile ranks
- rarity flags
- log features

## Slide 6 — Split Design
- Train: 2018-2021
- Validation: 2022
- Final train: 2018-2022
- Final test: 2023

## Slide 7 — Models
- Logistic Regression baseline
- CatBoost main model
- final chosen model = CatBoost

## Slide 8 — Final 2023 Test Result
- AUPRC 0.0006627
- ROC-AUC 0.7676
- Precision@1000 0.001
- Precision@5000 0.0012
- Lift@1000 8.4

## Slide 9 — Explainability
- SHAP summary bar
- SHAP beeswarm
- local explanation examples

## Slide 10 — Batch Scoring + Docker
- feature-ready input
- risk score, rank, percentile
- Dockerized scorer works

## Slide 11 — Monitoring + Limitations
- score drift comparison
- feature drift summary
- proxy labels
- severe imbalance
- batch scoring stronger than live single-record scoring

## Slide 12 — Conclusion
- technically credible full ML pipeline
- honest interpretation
- risk-ranking tool, not legal judgment

