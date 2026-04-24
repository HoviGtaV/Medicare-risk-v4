import pandas as pd
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_features_v3.parquet")

print("Loaded:", df.shape)
dup_count = df.duplicated(subset=["npi_clean", "summary_year"]).sum()
print("Duplicate (npi_clean, summary_year) before dedupe:", dup_count)

df = df.drop_duplicates(subset=["npi_clean", "summary_year"], keep="first").copy()
print("Shape after dedupe:", df.shape)

train_df = df[df["summary_year"].isin([2018, 2019, 2020, 2021])].copy()
valid_df = df[df["summary_year"] == 2022].copy()
test_df  = df[df["summary_year"] == 2023].copy()

summary_rows = [
    {
        "split": "train_v3",
        "rows": len(train_df),
        "positives": int(train_df["target_excluded_24m"].sum()),
        "positive_rate": float(train_df["target_excluded_24m"].mean()),
        "unique_npi": int(train_df["npi_clean"].nunique()),
        "years": "2018-2021",
    },
    {
        "split": "valid_v3",
        "rows": len(valid_df),
        "positives": int(valid_df["target_excluded_24m"].sum()),
        "positive_rate": float(valid_df["target_excluded_24m"].mean()),
        "unique_npi": int(valid_df["npi_clean"].nunique()),
        "years": "2022",
    },
    {
        "split": "test_v3",
        "rows": len(test_df),
        "positives": int(test_df["target_excluded_24m"].sum()),
        "positive_rate": float(test_df["target_excluded_24m"].mean()),
        "unique_npi": int(test_df["npi_clean"].nunique()),
        "years": "2023",
    },
]

summary_df = pd.DataFrame(summary_rows)

train_df.to_parquet("data/processed/train_v3.parquet", index=False)
valid_df.to_parquet("data/processed/valid_v3.parquet", index=False)
test_df.to_parquet("data/processed/test_v3.parquet", index=False)
summary_df.to_csv("outputs/metrics/split_summary_v3.csv", index=False)

print("\nSplit summary:")
print(summary_df.to_string(index=False))

print("\nSaved:")
print("data/processed/train_v3.parquet")
print("data/processed/valid_v3.parquet")
print("data/processed/test_v3.parquet")
print("outputs/metrics/split_summary_v3.csv")
