import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# Paths
# ----------------------------
train_path = Path("data/processed/train_v4.parquet")
valid_path = Path("data/processed/valid_v4.parquet")

out_dir = Path("models_v4/logreg")
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Config
# ----------------------------
target_col = "target_excluded_24m"
random_state = 42

# Keep all positives, sample negatives for faster but still meaningful baseline
negative_to_positive_ratio = 20

# Moderate grid, still strong enough for evidence
candidate_C = [0.5, 1.0, 2.0]

# Moderate training budget
max_iter = 200
tol = 1e-3

# ----------------------------
# Load data
# ----------------------------
train_df = pd.read_parquet(train_path)
valid_df = pd.read_parquet(valid_path)

print("Train shape (full):", train_df.shape)
print("Valid shape (full):", valid_df.shape)

# ----------------------------
# Feature policy
# ----------------------------
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
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

print("\nFeature count:", len(feature_cols))
print("Categorical columns:", categorical_cols)
print("Numeric columns:", len(numeric_cols))

# ----------------------------
# Build sampled training set
# ----------------------------
train_pos = train_df[train_df[target_col] == 1].copy()
train_neg = train_df[train_df[target_col] == 0].copy()

n_pos = len(train_pos)
n_neg_keep = min(len(train_neg), negative_to_positive_ratio * n_pos)

train_neg_sampled = train_neg.sample(
    n=n_neg_keep,
    random_state=random_state
)

train_fit_df = pd.concat([train_pos, train_neg_sampled], axis=0)
train_fit_df = train_fit_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

print("\nTraining subset created:")
print("All positives kept:", len(train_pos))
print("Negatives sampled:", len(train_neg_sampled))
print("Train subset shape:", train_fit_df.shape)
print("Train subset positive rate:", float(train_fit_df[target_col].mean()))

X_train = train_fit_df[feature_cols].copy()
y_train = train_fit_df[target_col].astype(int).copy()

X_valid = valid_df[feature_cols].copy()
y_valid = valid_df[target_col].astype(int).copy()

# ----------------------------
# Preprocessing
# ----------------------------
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

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
# Train / select best model
# ----------------------------
all_metrics = []
best = None

for C in candidate_C:
    print(f"\n=== Training LogisticRegression v4, C={C} ===")

    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        tol=tol,
        class_weight="balanced",
        random_state=random_state,
        verbose=1
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    valid_scores = pipe.predict_proba(X_valid)[:, 1]

    metrics = {
        "model": "logreg_v4",
        "C": C,
        "n_train_full": int(len(train_df)),
        "n_train_subset": int(len(train_fit_df)),
        "n_valid": int(len(valid_df)),
        "train_subset_positive_count": int(y_train.sum()),
        "train_subset_negative_count": int((y_train == 0).sum()),
        "valid_positive_count": int(y_valid.sum()),
        "train_subset_positive_rate": float(y_train.mean()),
        "valid_positive_rate": float(y_valid.mean()),
        "negative_to_positive_ratio_used": int(negative_to_positive_ratio),
        "max_iter": int(max_iter),
        "tol": float(tol),
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

    all_metrics.append(metrics)

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    current_key = (
        metrics["auprc_valid"],
        metrics["precision_at_500_valid"],
        metrics["precision_at_1000_valid"]
    )

    best_key = None if best is None else (
        best["metrics"]["auprc_valid"],
        best["metrics"]["precision_at_500_valid"],
        best["metrics"]["precision_at_1000_valid"]
    )

    if best is None or current_key > best_key:
        best = {
            "C": C,
            "pipe": pipe,
            "metrics": metrics,
            "scores": valid_scores,
        }

# ----------------------------
# Save summary table
# ----------------------------
metrics_df = pd.DataFrame(all_metrics).sort_values(
    ["auprc_valid", "precision_at_500_valid", "precision_at_1000_valid"],
    ascending=False
)

metrics_df.to_csv(out_dir / "all_runs_metrics.csv", index=False)

# ----------------------------
# Save best model
# ----------------------------
best_pipe = best["pipe"]
best_scores = best["scores"]
best_metrics = best["metrics"]

joblib.dump(best_pipe, out_dir / "model.joblib")

with open(out_dir / "metrics_valid.json", "w") as f:
    json.dump(best_metrics, f, indent=2)

valid_scored = valid_df[["npi_clean", "summary_year", "rndrng_prvdr_type", target_col]].copy()
valid_scored["score_logreg_v4"] = best_scores
valid_scored = valid_scored.sort_values("score_logreg_v4", ascending=False)
valid_scored.to_parquet(out_dir / "valid_scored.parquet", index=False)

feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
coefs = best_pipe.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "abs_coefficient": np.abs(coefs),
}).sort_values("abs_coefficient", ascending=False)

coef_df.to_csv(out_dir / "coefficients.csv", index=False)

Path("outputs/metrics_v4").mkdir(parents=True, exist_ok=True)
metrics_df.to_csv("outputs/metrics_v4/logreg_v4_model_selection.csv", index=False)

print("\n=== BEST LOGREG V4 ===")
print(f"Best C: {best['C']}")
for k, v in best_metrics.items():
    print(f"{k}: {v}")

print("\nSaved:")
print(out_dir / "model.joblib")
print(out_dir / "metrics_valid.json")
print(out_dir / "valid_scored.parquet")
print(out_dir / "coefficients.csv")
print(out_dir / "all_runs_metrics.csv")
print("outputs/metrics_v4/logreg_v4_model_selection.csv")