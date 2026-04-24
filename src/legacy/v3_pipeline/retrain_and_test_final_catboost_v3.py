import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

# ----------------------------
# Paths
# ----------------------------
train_path = Path("data/processed/train_v3.parquet")
valid_path = Path("data/processed/valid_v3.parquet")
test_path  = Path("data/processed/test_v3.parquet")

out_dir = Path("models_v3/final")
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
train_df = pd.read_parquet(train_path)
valid_df = pd.read_parquet(valid_path)
test_df  = pd.read_parquet(test_path)

print("Train v3 shape:", train_df.shape)
print("Valid v3 shape:", valid_df.shape)
print("Test v3 shape:", test_df.shape)

# Combine 2018-2021 + 2022 for final training
final_train_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)

print("\nFinal combined train shape:", final_train_df.shape)

target_col = "target_excluded_24m"

always_drop = [
    "npi_clean",
    "summary_year",
    target_col,
]

constant_drop = [
    "tot_benes_missing",
    "tot_srvcs_missing",
    "tot_sbmtd_chrg_missing",
    "tot_mdcr_alowd_amt_missing",
    "tot_mdcr_pymt_amt_missing",
    "tot_mdcr_stdzd_amt_missing",
]

drop_cols = always_drop + [c for c in constant_drop if c in final_train_df.columns]

feature_cols = [c for c in final_train_df.columns if c not in drop_cols]
categorical_cols = ["rndrng_prvdr_type"]
cat_feature_indices = [feature_cols.index(c) for c in categorical_cols]

X_train = final_train_df[feature_cols].copy()
y_train = final_train_df[target_col].astype(int).copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].astype(int).copy()

print("\nFeature count:", len(feature_cols))
print("Categorical columns:", categorical_cols)
print("Cat feature indices:", cat_feature_indices)

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

# ----------------------------
# Metrics helpers
# ----------------------------
def precision_at_k(y_true, scores, k):
    k = min(k, len(y_true))
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[order]))

def recall_at_k(y_true, scores, k):
    total_pos = int(np.sum(y_true))
    if total_pos == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(scores)[::-1][:k]
    return float(np.sum(np.asarray(y_true)[order]) / total_pos)

base_rate = float(np.mean(y_test))

# ----------------------------
# Final model
# Using 203 iterations because the best validation model
# stopped around iteration 202
# ----------------------------
model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=203,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100
)

print("\nTraining final CatBoost on 2018-2022...")
model.fit(train_pool)

# ----------------------------
# Test predictions
# ----------------------------
test_scores = model.predict_proba(test_pool)[:, 1]

metrics = {
    "model": "catboost_final_v3",
    "n_final_train": int(len(final_train_df)),
    "n_test": int(len(test_df)),
    "final_train_positive_count": int(y_train.sum()),
    "test_positive_count": int(y_test.sum()),
    "final_train_positive_rate": float(y_train.mean()),
    "test_positive_rate": float(y_test.mean()),
    "iterations_used": 203,
    "auprc_test": float(average_precision_score(y_test, test_scores)),
    "roc_auc_test": float(roc_auc_score(y_test, test_scores)),
    "precision_at_100_test": precision_at_k(y_test, test_scores, 100),
    "precision_at_500_test": precision_at_k(y_test, test_scores, 500),
    "precision_at_1000_test": precision_at_k(y_test, test_scores, 1000),
    "precision_at_5000_test": precision_at_k(y_test, test_scores, 5000),
    "recall_at_500_test": recall_at_k(y_test, test_scores, 500),
    "recall_at_1000_test": recall_at_k(y_test, test_scores, 1000),
    "lift_at_500_test": precision_at_k(y_test, test_scores, 500) / base_rate if base_rate > 0 else None,
    "lift_at_1000_test": precision_at_k(y_test, test_scores, 1000) / base_rate if base_rate > 0 else None,
}

print("\n=== FINAL CATBOOST V3 TEST METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

# ----------------------------
# Save final model and outputs
# ----------------------------
model.save_model(str(out_dir / "model.cbm"))

with open(out_dir / "metrics_test.json", "w") as f:
    json.dump(metrics, f, indent=2)

test_scored = test_df[["npi_clean", "summary_year", "rndrng_prvdr_type", target_col]].copy()
test_scored["score_catboost_final_v3"] = test_scores
test_scored = test_scored.sort_values("score_catboost_final_v3", ascending=False)
test_scored.to_parquet(out_dir / "test_scored.parquet", index=False)

fi = model.get_feature_importance(type="FeatureImportance")
fi_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": fi,
}).sort_values("importance", ascending=False)

fi_df.to_csv(out_dir / "feature_importance.csv", index=False)

Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
pd.DataFrame([metrics]).to_csv("outputs/metrics/final_catboost_v3_metrics.csv", index=False)

print("\nSaved:")
print(out_dir / "model.cbm")
print(out_dir / "metrics_test.json")
print(out_dir / "test_scored.parquet")
print(out_dir / "feature_importance.csv")
print("outputs/metrics/final_catboost_v3_metrics.csv")
