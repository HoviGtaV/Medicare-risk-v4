import json
import pandas as pd

rows = []

with open(r"models_v4/logreg/metrics_valid.json", "r") as f:
    j = json.load(f)
rows.append({
    "stage": "validation_2022",
    "model": "logistic_v4",
    "auprc": j["auprc_valid"],
    "roc_auc": j["roc_auc_valid"],
    "precision_at_500": j["precision_at_500_valid"],
    "precision_at_1000": j["precision_at_1000_valid"],
    "lift_at_500": j["lift_at_500_valid"],
    "lift_at_1000": j["lift_at_1000_valid"],
})

with open(r"models_v4/catboost/metrics_valid.json", "r") as f:
    j = json.load(f)
rows.append({
    "stage": "validation_2022",
    "model": "catboost_v4",
    "auprc": j["auprc_valid"],
    "roc_auc": j["roc_auc_valid"],
    "precision_at_500": j["precision_at_500_valid"],
    "precision_at_1000": j["precision_at_1000_valid"],
    "lift_at_500": j["lift_at_500_valid"],
    "lift_at_1000": j["lift_at_1000_valid"],
})

with open(r"models_v4/final/metrics_test.json", "r") as f:
    j = json.load(f)
rows.append({
    "stage": "test_2023",
    "model": "catboost_final_v4",
    "auprc": j["auprc_test"],
    "roc_auc": j["roc_auc_test"],
    "precision_at_500": j["precision_at_500_test"],
    "precision_at_1000": j["precision_at_1000_test"],
    "lift_at_500": j["lift_at_500_test"],
    "lift_at_1000": j["lift_at_1000_test"],
})

df = pd.DataFrame(rows)
df.to_csv(r"outputs/metrics_v4/model_comparison_v4.csv", index=False)
print(df.to_string(index=False))
