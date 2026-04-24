from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/provider_year_features_v4.parquet")
OUT_DIR = Path("data/processed")

TRAIN_PATH = OUT_DIR / "train_v4.parquet"
VALID_PATH = OUT_DIR / "valid_v4.parquet"
TEST_PATH  = OUT_DIR / "test_v4.parquet"

TARGET_COL = "target_excluded_24m"
YEAR_COL = "summary_year"

print(f"Loading: {INPUT_PATH}")
df = pd.read_parquet(INPUT_PATH, columns=[YEAR_COL, TARGET_COL])
print("Loaded shape:", df.shape)

print("Reading full feature table...")
full_df = pd.read_parquet(INPUT_PATH)
print("Full table shape:", full_df.shape)

required_cols = [YEAR_COL, TARGET_COL]
missing = [c for c in required_cols if c not in full_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

train_df = full_df[full_df[YEAR_COL].between(2018, 2021)].copy()
valid_df = full_df[full_df[YEAR_COL] == 2022].copy()
test_df  = full_df[full_df[YEAR_COL] == 2023].copy()

print("\nSplit shapes:")
print("Train:", train_df.shape)
print("Valid:", valid_df.shape)
print("Test :", test_df.shape)

print("\nTarget counts:")
print("Train:")
print(train_df[TARGET_COL].value_counts(dropna=False))
print("\nValid:")
print(valid_df[TARGET_COL].value_counts(dropna=False))
print("\nTest:")
print(test_df[TARGET_COL].value_counts(dropna=False))

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nSaving split parquet files...")
train_df.to_parquet(TRAIN_PATH, index=False)
valid_df.to_parquet(VALID_PATH, index=False)
test_df.to_parquet(TEST_PATH, index=False)

print("\nSaved:")
print(TRAIN_PATH)
print(VALID_PATH)
print(TEST_PATH)