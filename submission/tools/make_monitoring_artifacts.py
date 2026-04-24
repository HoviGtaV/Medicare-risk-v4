from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

OUT_DIR = Path("submission/monitoring")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_PATH = "data/processed/valid_v4.parquet"
TEST_PATH = "data/processed/test_v4.parquet"
MODEL_PATH = "models_v4/final/model.cbm"

TARGET_COL = "target_excluded_24m"
DROP_COLS = ["npi_clean", "summary_year", TARGET_COL]
CAT_COLS = ["rndrng_prvdr_type"]
SCORE_COL = "score_catboost_final_v4"

def load_and_score(path, split_name, model):
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    scores = model.predict_proba(X)[:, 1]
    df[SCORE_COL] = scores

    score_summary = {
        "split": split_name,
        "rows": int(len(df)),
        "positive_count": int(df[TARGET_COL].sum()),
        "positive_rate": float(df[TARGET_COL].mean()),
        "score_mean": float(df[SCORE_COL].mean()),
        "score_median": float(df[SCORE_COL].median()),
        "score_p90": float(df[SCORE_COL].quantile(0.90)),
        "score_p95": float(df[SCORE_COL].quantile(0.95)),
        "score_p99": float(df[SCORE_COL].quantile(0.99)),
        "avg_score_positive_rows": float(df.loc[df[TARGET_COL] == 1, SCORE_COL].mean()) if int(df[TARGET_COL].sum()) > 0 else np.nan,
        "avg_score_negative_rows": float(df.loc[df[TARGET_COL] == 0, SCORE_COL].mean()),
    }

    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    feature_summary = pd.DataFrame({
        "feature": numeric_feature_cols,
        f"{split_name}_mean": [float(df[c].mean()) for c in numeric_feature_cols],
        f"{split_name}_std": [float(df[c].std()) for c in numeric_feature_cols],
    })

    return df[[SCORE_COL]].copy(), score_summary, feature_summary

print("Loading final model:", MODEL_PATH)
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

valid_scores_df, valid_score_summary, valid_feature_summary = load_and_score(VALID_PATH, "valid_2022", model)
test_scores_df, test_score_summary, test_feature_summary = load_and_score(TEST_PATH, "test_2023", model)

score_summary_df = pd.DataFrame([valid_score_summary, test_score_summary])
score_summary_path = OUT_DIR / "score_drift_summary.csv"
score_summary_df.to_csv(score_summary_path, index=False)

feature_drift = valid_feature_summary.merge(test_feature_summary, on="feature", how="inner")
feature_drift["mean_delta"] = feature_drift["test_2023_mean"] - feature_drift["valid_2022_mean"]
feature_drift["pooled_std"] = ((feature_drift["valid_2022_std"].fillna(0) + feature_drift["test_2023_std"].fillna(0)) / 2.0).replace(0, np.nan)
feature_drift["abs_standardized_mean_diff"] = (feature_drift["mean_delta"].abs() / feature_drift["pooled_std"]).fillna(0.0)
feature_drift = feature_drift.sort_values("abs_standardized_mean_diff", ascending=False)

feature_drift_path = OUT_DIR / "feature_drift_summary.csv"
feature_drift.to_csv(feature_drift_path, index=False)

rng = np.random.default_rng(42)
valid_plot_scores = valid_scores_df[SCORE_COL].to_numpy()
test_plot_scores = test_scores_df[SCORE_COL].to_numpy()

if len(valid_plot_scores) > 100000:
    valid_plot_scores = rng.choice(valid_plot_scores, size=100000, replace=False)
if len(test_plot_scores) > 100000:
    test_plot_scores = rng.choice(test_plot_scores, size=100000, replace=False)

plt.figure(figsize=(10, 6))
plt.hist(valid_plot_scores, bins=50, alpha=0.6, label="2022 validation")
plt.hist(test_plot_scores, bins=50, alpha=0.6, label="2023 test")
plt.xlabel("Predicted score")
plt.ylabel("Count")
plt.title("Score distribution: 2022 validation vs 2023 test")
plt.legend()
plot_path = OUT_DIR / "score_distribution_2022_vs_2023.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

note_path = OUT_DIR / "MONITORING_NOTE.md"
note_lines = [
    "# Monitoring Note (v4)",
    "",
    "These monitoring artifacts were generated from the active v4 final model.",
    "",
    f"- Model: `{MODEL_PATH}`",
    f"- Validation split: `{VALID_PATH}`",
    f"- Test split: `{TEST_PATH}`",
    "",
    "Generated files:",
    f"- `{score_summary_path}`",
    f"- `{feature_drift_path}`",
    f"- `{plot_path}`",
]

note_path.write_text("\n".join(note_lines), encoding="utf-8")

print("\nSaved:")
print(score_summary_path)
print(feature_drift_path)
print(plot_path)
print(note_path)
