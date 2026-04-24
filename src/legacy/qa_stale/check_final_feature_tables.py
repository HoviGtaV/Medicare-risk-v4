import pandas as pd

train_df = pd.read_parquet("data/processed/train.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain missing % top 20:")
print((train_df.isna().mean() * 100).sort_values(ascending=False).head(20))

print("\nTest missing % top 20:")
print((test_df.isna().mean() * 100).sort_values(ascending=False).head(20))

print("\nTrain target counts:")
print(train_df["target_excluded_24m"].value_counts(dropna=False))

print("\nTest target counts:")
print(test_df["target_excluded_24m"].value_counts(dropna=False))