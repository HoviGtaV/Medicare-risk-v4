from pathlib import Path
import pandas as pd

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_features.parquet")

train_df = df[df["summary_year"].between(2017, 2020)].copy()
val_df   = df[df["summary_year"] == 2021].copy()
test_df  = df[df["summary_year"].between(2022, 2023)].copy()

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain target counts:")
print(train_df["target_excluded_24m"].value_counts(dropna=False))

print("\nVal target counts:")
print(val_df["target_excluded_24m"].value_counts(dropna=False))

print("\nTest target counts:")
print(test_df["target_excluded_24m"].value_counts(dropna=False))

print("\nTrain positive rate:", train_df["target_excluded_24m"].mean())
print("Val positive rate:", val_df["target_excluded_24m"].mean())
print("Test positive rate:", test_df["target_excluded_24m"].mean())

train_df.to_parquet("data/processed/train.parquet", index=False)
val_df.to_parquet("data/processed/val.parquet", index=False)
test_df.to_parquet("data/processed/test.parquet", index=False)

print("\nSaved:")
print("data/processed/train.parquet")
print("data/processed/val.parquet")
print("data/processed/test.parquet")