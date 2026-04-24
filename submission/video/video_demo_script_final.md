# Final 3-Minute Demo Script

## 0:00 - 0:20
This project builds a provider-level Medicare fraud-risk scoring system using public Medicare summary data and OIG exclusion-based proxy labels.

## 0:20 - 0:40
Important framing: this is not a legal fraud detector. The model produces a risk score for provider-year profiles using exclusion-based proxy labels.

## 0:40 - 1:00
We combine CMS provider summary data from 2017 to 2023 with OIG exclusion information and create a forward-looking label called `target_excluded_24m`.

## 1:00 - 1:25
We engineer raw totals, per-beneficiary and per-service ratios, structural payment ratios, specialty-year comparisons, log features, and rarity flags.

## 1:25 - 1:45
We use a time-based split: train on 2018-2021, validate on 2022, and keep 2023 as the final holdout year.

## 1:45 - 2:05
We train a logistic regression baseline and a stronger CatBoost model. The final CatBoost model reaches ROC-AUC about 0.768 and lift@1000 about 8.4 on the 2023 holdout year.

## 2:05 - 2:25
For explainability, we use SHAP. The strongest drivers include financial ratios, service intensity, and specialty-relative percentile features.

## 2:25 - 2:45
For deployment, we built a batch scorer that accepts a feature-ready provider-year table and outputs risk score, rank, and percentile. The scorer also runs inside Docker.

## 2:45 - 3:00
The final system is a technically credible risk-ranking prototype with honest framing, but it should be used for review prioritization rather than legal judgment.

