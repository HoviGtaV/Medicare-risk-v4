from pathlib import Path
import numpy as np
import pandas as pd

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_labeled.parquet")

print("Loaded labeled table:", df.shape)

raw_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
]

keep_cols = [
    "npi_clean",
    "summary_year",
    "rndrng_prvdr_type",
    "target_excluded_24m",
] + raw_cols

features_df = df[keep_cols].copy()


def safe_divide(a, b):
    b_nonzero = b.replace(0, np.nan)
    return a / b_nonzero


# missing flags
for col in raw_cols:
    features_df[f"{col}_missing"] = features_df[col].isna().astype(int)

# ratio features
features_df["srvcs_per_bene"] = safe_divide(features_df["tot_srvcs"], features_df["tot_benes"])
features_df["sbmtd_chrg_per_bene"] = safe_divide(features_df["tot_sbmtd_chrg"], features_df["tot_benes"])
features_df["alowd_amt_per_bene"] = safe_divide(features_df["tot_mdcr_alowd_amt"], features_df["tot_benes"])
features_df["pymt_amt_per_bene"] = safe_divide(features_df["tot_mdcr_pymt_amt"], features_df["tot_benes"])

features_df["sbmtd_chrg_per_srvc"] = safe_divide(features_df["tot_sbmtd_chrg"], features_df["tot_srvcs"])
features_df["alowd_amt_per_srvc"] = safe_divide(features_df["tot_mdcr_alowd_amt"], features_df["tot_srvcs"])
features_df["pymt_amt_per_srvc"] = safe_divide(features_df["tot_mdcr_pymt_amt"], features_df["tot_srvcs"])

ratio_cols = [
    "srvcs_per_bene",
    "sbmtd_chrg_per_bene",
    "alowd_amt_per_bene",
    "pymt_amt_per_bene",
    "sbmtd_chrg_per_srvc",
    "alowd_amt_per_srvc",
    "pymt_amt_per_srvc",
]

# log transforms for heavy-tailed raw columns
for col in raw_cols:
    features_df[f"log1p_{col}"] = np.log1p(features_df[col])

# peer features by summary_year + provider type
group_cols = ["summary_year", "rndrng_prvdr_type"]

peer_cols = raw_cols + ratio_cols

for col in peer_cols:
    median_col = f"{col}_specialty_year_median"
    diff_col = f"{col}_minus_specialty_year_median"
    pct_col = f"{col}_specialty_year_pct_rank"

    features_df[median_col] = features_df.groupby(group_cols)[col].transform("median")
    features_df[diff_col] = features_df[col] - features_df[median_col]
    features_df[pct_col] = features_df.groupby(group_cols)[col].rank(pct=True)

# replace inf with nan
features_df = features_df.replace([np.inf, -np.inf], np.nan)

print("\nFeature table shape:", features_df.shape)

print("\nTarget counts:")
print(features_df["target_excluded_24m"].value_counts(dropna=False))

print("\nTop missing %:")
print((features_df.isna().mean() * 100).sort_values(ascending=False).head(20))

dup_count = features_df.duplicated(subset=["npi_clean", "summary_year"]).sum()
print("\nDuplicate provider-year rows:", dup_count)

if dup_count > 0:
    raise ValueError(f"Duplicate provider-year rows found: {dup_count}")

features_df.to_parquet("data/processed/provider_year_features.parquet", index=False)

print("\nSaved:")
print("data/processed/provider_year_features.parquet")