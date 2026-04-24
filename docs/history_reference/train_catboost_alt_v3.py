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

out_dir = Path("models_v3/catboost_alt")
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
train_df = pd.read_parquet(train_path)
valid_df = pd.read_parquet(valid_path)

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)

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

drop_cols = always_drop + [c for c in constant_drop if c in train_df.columns]

feature_cols = [c for c in train_df.columns if c not in drop_cols]
categorical_cols = ["rndrng_prvdr_type"]
cat_feature_indices = [feature_cols.index(c) for c in categorical_cols]

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].astype(int).copy()

X_valid = valid_df[feature_cols].copy()
y_valid = valid_df[target_col].astype(int).copy()

print("\nFeature count:", len(feature_cols))
print("Categorical columns:", categorical_cols)
print("Cat feature indices:", cat_feature_indices)

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_feature_indices)

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

base_rate = float(np.mean(y_valid))

# ----------------------------
# Alternative model
# ----------------------------
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",
    iterations=400,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=10,
    auto_class_weights="Balanced",
    early_stopping_rounds=100,
    random_seed=42,
    verbose=100
)

print("\nFitting alternative CatBoost...")
model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True
)

valid_scores = model.predict_proba(valid_pool)[:, 1]

metrics = {
    "model": "catboost_alt",
    "n_train": int(len(train_df)),
    "n_valid": int(len(valid_df)),
    "train_positive_count": int(y_train.sum()),
    "valid_positive_count": int(y_valid.sum()),
    "train_positive_rate": float(y_train.mean()),
    "valid_positive_rate": float(y_valid.mean()),
    "best_iteration": int(model.get_best_iteration()),
    "auprc_valid": float(average_precision_score(y_valid, valid_scores)),
    "roc_auc_valid": float(roc_auc_score(y_valid, valid_scores)),
    "precision_at_100_valid": precision_at_k(y_valid, valid_scores, 100),
    "precision_at_500_valid": precision_at_k(y_valid, valid_scores, 500),
    "precision_at_1000_valid": precision_at_k(y_valid, valid_scores, 1000),
    "precision_at_5000_valid": precision_at_k(y_valid, valid_scores, 5000),
    "recall_at_500_valid": recall_at_k(y_valid, valid_scores, 500),
    "recall_at_1000_valid": recall_at_k(y_valid, valid_scores, 1000),
    "lift_at_500_valid": precision_at_k(y_valid, valid_scores, 500) / base_rate if base_rate > 0 else None,
    "lift_at_1000_valid": precision_at_k(y_valid, valid_scores, 1000) / base_rate if base_rate > 0 else None,
}

print("\n=== ALTERNATIVE CATBOOST VALIDATION METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

model.save_model(str(out_dir / "model.cbm"))

with open(out_dir / "metrics_valid.json", "w") as f:
    json.dump(metrics, f, indent=2)

valid_scored = valid_df[["npi_clean", "summary_year", "rndrng_prvdr_type", target_col]].copy()
valid_scored["score_catboost_alt"] = valid_scores
valid_scored = valid_scored.sort_values("score_catboost_alt", ascending=False)
valid_scored.to_parquet(out_dir / "valid_scored.parquet", index=False)

fi = model.get_feature_importance(type="FeatureImportance")
fi_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": fi,
}).sort_values("importance", ascending=False)

fi_df.to_csv(out_dir / "feature_importance.csv", index=False)

print("\nSaved:")
print(out_dir / "model.cbm")
print(out_dir / "metrics_valid.json")
print(out_dir / "valid_scored.parquet")
print(out_dir / "feature_importance.csv")
