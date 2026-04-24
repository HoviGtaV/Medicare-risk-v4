from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

Path("models").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)

train_df = pd.read_parquet("data/processed/train.parquet")
val_df = pd.read_parquet("data/processed/val.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")

target_col = "target_excluded_24m"
id_cols = ["npi_clean", "summary_year"]
cat_cols = ["rndrng_prvdr_type"]

drop_cols = id_cols + [target_col]
feature_cols = [c for c in train_df.columns if c not in drop_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

# all positives + manageable negative sample
pos_train = train_df[train_df[target_col] == 1]
neg_train = train_df[train_df[target_col] == 0].sample(n=200000, random_state=42)
train_sample = pd.concat([pos_train, neg_train], axis=0).sample(frac=1, random_state=42)

print("Train full shape:", train_df.shape)
print("Train sample shape:", train_sample.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

X_train = train_sample[feature_cols]
y_train = train_sample[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

val_scores = model.predict_proba(X_val)[:, 1]
test_scores = model.predict_proba(X_test)[:, 1]

def precision_at_k(y_true, scores, k):
    order = np.argsort(-scores)[:k]
    return float(np.mean(y_true.iloc[order]))

metrics = {
    "val_auprc": float(average_precision_score(y_val, val_scores)),
    "val_roc_auc": float(roc_auc_score(y_val, val_scores)),
    "test_auprc": float(average_precision_score(y_test, test_scores)),
    "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
    "val_precision_at_100": precision_at_k(y_val, val_scores, 100),
    "val_precision_at_1000": precision_at_k(y_val, val_scores, 1000),
    "test_precision_at_100": precision_at_k(y_test, test_scores, 100),
    "test_precision_at_1000": precision_at_k(y_test, test_scores, 1000),
}

print("\nMetrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

joblib.dump(model, "models/logreg_baseline.joblib")

with open("models/logreg_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

val_pred = val_df[id_cols + [target_col]].copy()
val_pred["score_logreg"] = val_scores
val_pred.to_csv("reports/tables/val_predictions_logreg.csv", index=False)

test_pred = test_df[id_cols + [target_col]].copy()
test_pred["score_logreg"] = test_scores
test_pred.to_csv("reports/tables/test_predictions_logreg.csv", index=False)

print("\nSaved:")
print("models/logreg_baseline.joblib")
print("models/logreg_metrics.json")
print("reports/tables/val_predictions_logreg.csv")
print("reports/tables/test_predictions_logreg.csv")