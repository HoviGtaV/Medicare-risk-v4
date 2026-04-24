import numpy as np
import pandas as pd
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_labeled.parquet")
print("Loaded labeled table:", df.shape)

target_col = "target_excluded_24m"
group_cols = ["summary_year", "rndrng_prvdr_type"]

raw_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
]

base_cols = [
    "npi_clean",
    "summary_year",
    "rndrng_prvdr_type",
    target_col,
] + raw_cols

features_df = df[base_cols].copy()

for col in raw_cols:
    features_df[f"{col}_missing"] = features_df[col].isna().astype(int)

def safe_divide(a, b):
    b = b.replace(0, np.nan)
    return a / b

# Existing ratio features
features_df["srvcs_per_bene"] = safe_divide(features_df["tot_srvcs"], features_df["tot_benes"])
features_df["sbmtd_chrg_per_bene"] = safe_divide(features_df["tot_sbmtd_chrg"], features_df["tot_benes"])
features_df["alowd_amt_per_bene"] = safe_divide(features_df["tot_mdcr_alowd_amt"], features_df["tot_benes"])
features_df["pymt_amt_per_bene"] = safe_divide(features_df["tot_mdcr_pymt_amt"], features_df["tot_benes"])

features_df["sbmtd_chrg_per_srvc"] = safe_divide(features_df["tot_sbmtd_chrg"], features_df["tot_srvcs"])
features_df["alowd_amt_per_srvc"] = safe_divide(features_df["tot_mdcr_alowd_amt"], features_df["tot_srvcs"])
features_df["pymt_amt_per_srvc"] = safe_divide(features_df["tot_mdcr_pymt_amt"], features_df["tot_srvcs"])

# New structural ratios
features_df["sbmtd_to_alowd_ratio"] = safe_divide(features_df["tot_sbmtd_chrg"], features_df["tot_mdcr_alowd_amt"])
features_df["alowd_to_pymt_ratio"] = safe_divide(features_df["tot_mdcr_alowd_amt"], features_df["tot_mdcr_pymt_amt"])
features_df["stdzd_to_pymt_ratio"] = safe_divide(features_df["tot_mdcr_stdzd_amt"], features_df["tot_mdcr_pymt_amt"])
features_df["sbmtd_to_pymt_ratio"] = safe_divide(features_df["tot_sbmtd_chrg"], features_df["tot_mdcr_pymt_amt"])

# Log features for raw totals
for col in raw_cols:
    features_df[f"log1p_{col}"] = np.log1p(features_df[col].clip(lower=0))

# Log features for main intensity variables
log_ratio_cols = [
    "srvcs_per_bene",
    "sbmtd_chrg_per_bene",
    "alowd_amt_per_bene",
    "pymt_amt_per_bene",
    "sbmtd_chrg_per_srvc",
    "alowd_amt_per_srvc",
    "pymt_amt_per_srvc",
]
for col in log_ratio_cols:
    features_df[f"log1p_{col}"] = np.log1p(features_df[col].clip(lower=0))

# Peer median / deviation / percentile for raw totals
for col in raw_cols:
    median_col = f"{col}_specialty_year_median"
    diff_col = f"{col}_minus_specialty_year_median"
    pct_col = f"{col}_specialty_year_pct_rank"

    features_df[median_col] = features_df.groupby(group_cols)[col].transform("median")
    features_df[diff_col] = features_df[col] - features_df[median_col]
    features_df[pct_col] = features_df.groupby(group_cols)[col].rank(pct=True)

# Extra percentile ranks for key ratios
ratio_rank_cols = [
    "srvcs_per_bene",
    "sbmtd_chrg_per_bene",
    "alowd_amt_per_bene",
    "pymt_amt_per_bene",
    "sbmtd_to_alowd_ratio",
    "alowd_to_pymt_ratio",
    "sbmtd_to_pymt_ratio",
]
for col in ratio_rank_cols:
    features_df[f"{col}_specialty_year_pct_rank"] = features_df.groupby(group_cols)[col].rank(pct=True)

# Rarity flags
flag_cols = [
    "tot_srvcs",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_alowd_amt",
    "srvcs_per_bene",
    "sbmtd_to_pymt_ratio",
]
for col in flag_cols:
    pct_rank = features_df.groupby(group_cols)[col].rank(pct=True)
    features_df[f"flag_top1pct_{col}_within_specialty_year"] = (pct_rank >= 0.99).astype(int)
    features_df[f"flag_top5pct_{col}_within_specialty_year"] = (pct_rank >= 0.95).astype(int)

features_df = features_df.replace([np.inf, -np.inf], np.nan)

print("\nFeature table shape:", features_df.shape)
print("\nTarget counts:")
print(features_df[target_col].value_counts(dropna=False))
print("\nYear counts:")
print(features_df["summary_year"].value_counts().sort_index())
print("\nTop missing %:")
print((features_df.isna().mean() * 100).sort_values(ascending=False).head(25))
print("\nConstant columns:")
print([c for c in features_df.columns if features_df[c].nunique(dropna=False) <= 1])

features_df.to_parquet("data/processed/provider_year_features_v3.parquet", index=False)
print("\nSaved: data/processed/provider_year_features_v3.parquet")
