from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# ----------------------------
# Paths
# ----------------------------
test_path = Path("data/processed/test_v3.parquet")
model_path = Path("models_v3/final/model.cbm")

out_dir = Path("outputs/shap")
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load model and test data
# ----------------------------
df = pd.read_parquet(test_path)

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

drop_cols = always_drop + [c for c in constant_drop if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()

model = CatBoostClassifier()
model.load_model(str(model_path))

print("Loaded test shape:", df.shape)
print("Feature count:", len(feature_cols))

# ----------------------------
# Sample rows for SHAP
# ----------------------------
sample_size = min(5000, len(X))
sample_df = X.sample(n=sample_size, random_state=42).reset_index(drop=True)

print("SHAP sample size:", len(sample_df))

# ----------------------------
# Compute SHAP values
# ----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_df)

if isinstance(shap_values, list):
    shap_array = shap_values[1]
else:
    shap_array = shap_values

# ----------------------------
# Global importance CSV
# ----------------------------
mean_abs_shap = np.abs(shap_array).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv(out_dir / "global_shap_importance.csv", index=False)

print("\nTop 15 SHAP features:")
print(importance_df.head(15).to_string(index=False))

# ----------------------------
# Summary bar plot
# ----------------------------
plt.figure()
shap.summary_plot(shap_array, sample_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------
# Beeswarm plot
# ----------------------------
plt.figure()
shap.summary_plot(shap_array, sample_df, show=False)
plt.tight_layout()
plt.savefig(out_dir / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
plt.close()

# ----------------------------
# Local explanation examples
# ----------------------------
scores = model.predict_proba(X)[:, 1]

scored = df[["npi_clean", "summary_year", "rndrng_prvdr_type", target_col]].copy()
scored["score"] = scores
scored = scored.sort_values("score", ascending=False).reset_index(drop=True)

top_examples = scored.head(5).copy()

local_rows = []
for _, row in top_examples.iterrows():
    npi = row["npi_clean"]
    yr = row["summary_year"]

    idx = df.index[(df["npi_clean"] == npi) & (df["summary_year"] == yr)][0]
    x_row = X.loc[[idx]]
    shap_row = explainer.shap_values(x_row)
    if isinstance(shap_row, list):
        shap_row = shap_row[1]
    shap_row = np.array(shap_row).reshape(-1)

    contrib_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": shap_row,
        "abs_shap": np.abs(shap_row)
    }).sort_values("abs_shap", ascending=False)

    top3 = contrib_df.head(3)["feature"].tolist()

    local_rows.append({
        "npi_clean": row["npi_clean"],
        "summary_year": row["summary_year"],
        "rndrng_prvdr_type": row["rndrng_prvdr_type"],
        "true_label": row[target_col],
        "score": row["score"],
        "top_driver_1": top3[0] if len(top3) > 0 else None,
        "top_driver_2": top3[1] if len(top3) > 1 else None,
        "top_driver_3": top3[2] if len(top3) > 2 else None,
    })

local_df = pd.DataFrame(local_rows)
local_df.to_csv(out_dir / "local_explanations_top5.csv", index=False)

print("\nSaved:")
print(out_dir / "global_shap_importance.csv")
print(out_dir / "shap_summary_bar.png")
print(out_dir / "shap_beeswarm.png")
print(out_dir / "local_explanations_top5.csv")
