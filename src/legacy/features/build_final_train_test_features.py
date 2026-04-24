import pandas as pd
import numpy as np
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)

train_df = pd.read_parquet("data/processed/train_labeled.parquet")
test_df = pd.read_parquet("data/processed/test_labeled.parquet")
peer_ref = pd.read_parquet("data/interim/peer_reference_tables/train_peer_reference.parquet")

raw_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
]

def safe_divide(a, b):
    b_nonzero = b.replace(0, np.nan)
    return a / b_nonzero

def build_features(df, peer_ref):
    df = df.copy()

    keep_cols = [
        "npi_clean",
        "summary_year",
        "rndrng_prvdr_type",
        "target_excluded_24m",
    ] + raw_cols

    out = df[keep_cols].copy()

    # missing flags
    for col in raw_cols:
        out[f"{col}_missing"] = out[col].isna().astype(int)

    # ratios
    out["srvcs_per_bene"] = safe_divide(out["tot_srvcs"], out["tot_benes"])
    out["sbmtd_chrg_per_bene"] = safe_divide(out["tot_sbmtd_chrg"], out["tot_benes"])
    out["alowd_amt_per_bene"] = safe_divide(out["tot_mdcr_alowd_amt"], out["tot_benes"])
    out["pymt_amt_per_bene"] = safe_divide(out["tot_mdcr_pymt_amt"], out["tot_benes"])

    out["sbmtd_chrg_per_srvc"] = safe_divide(out["tot_sbmtd_chrg"], out["tot_srvcs"])
    out["alowd_amt_per_srvc"] = safe_divide(out["tot_mdcr_alowd_amt"], out["tot_srvcs"])
    out["pymt_amt_per_srvc"] = safe_divide(out["tot_mdcr_pymt_amt"], out["tot_srvcs"])

    # merge peer medians from TRAIN reference only
    out = out.merge(peer_ref, on=["summary_year", "rndrng_prvdr_type"], how="left")

    # deviation from median
    for col in raw_cols:
        med_col = f"{col}_specialty_year_median"
        out[f"{col}_minus_specialty_year_median"] = out[col] - out[med_col]

    return out

train_features = build_features(train_df, peer_ref)
test_features = build_features(test_df, peer_ref)

print("Train features shape:", train_features.shape)
print("Test features shape:", test_features.shape)

print("\nTrain target counts:")
print(train_features["target_excluded_24m"].value_counts(dropna=False))

print("\nTest target counts:")
print(test_features["target_excluded_24m"].value_counts(dropna=False))

train_features.to_parquet("data/processed/train.parquet", index=False)
test_features.to_parquet("data/processed/test.parquet", index=False)

print("\nSaved:")
print("data/processed/train.parquet")
print("data/processed/test.parquet")