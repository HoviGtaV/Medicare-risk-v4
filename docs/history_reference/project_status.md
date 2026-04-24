# Freeze Snapshot: v1 current baseline

## Snapshot folder
freeze_v1_20260410_1200

## What this freeze contains
- current core scripts
- current processed modeling tables
- current logistic regression artifacts
- current CatBoost artifacts
- file manifest
- project tree

## Current project framing
- Unit of analysis: provider-year
- Label: target_excluded_24m
- Label meaning: excluded within 24 months after summary year end
- Framing: provider risk scoring using exclusion-based proxy labels
- Not a legal fraud detector

## Current official split
- Train: 2017-2021
- Validation: 2022
- Test: 2023

## Current main processed files
- data/processed/provider_year_labeled.parquet
- data/processed/provider_year_features.parquet
- data/processed/train_v2.parquet
- data/processed/valid_v2.parquet
- data/processed/test_v2.parquet

## Current results summary

### Logistic regression validation
- AUPRC: 0.0007730497683929459
- ROC AUC: 0.8027907431429293
- Precision@100: 0.0
- Precision@500: 0.0
- Precision@1000: 0.0

### CatBoost validation
- AUPRC: 0.0007746043020769477
- ROC AUC: 0.7686146090559203
- Precision@100: 0.0
- Precision@500: 0.006
- Precision@1000: 0.005
- Precision@5000: 0.0016

### CatBoost 2023 test
- AUPRC: 0.0003403063454035295
- ROC AUC: 0.7026411172958974
- Precision@100: 0.0
- Precision@500: 0.0
- Precision@1000: 0.0
- Precision@5000: 0.0006

## What is already working
- data ingestion
- CMS + OIG cleaning
- label creation
- provider-year feature table
- baseline logistic model
- stronger CatBoost model
- time-based train/valid/test evaluation
- saved model outputs

## Known weaknesses in current version
- severe class imbalance
- weak top-K precision on final 2023 test
- temporal generalization drop from validation to test
- constant missingness flags should not be used
- likely need stronger features and more robust time-based model selection

## Planned next version (v2_improved)
- preserve same label policy and problem framing
- improve features (log transforms, stronger financial ratios, rarity flags)
- use rolling time validation
- compare logistic, CatBoost, and one additional boosted-tree model
- retrain winner on 2018-2022
- keep 2023 as final untouched test
- add SHAP, batch scorer, Docker, README, slides, report
