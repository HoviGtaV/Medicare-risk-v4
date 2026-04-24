from pathlib import Path
import pandas as pd


Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/interim/qa").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_joined.parquet")

df["year_end"] = pd.to_datetime(df["summary_year"].astype(str) + "-12-31")
df["horizon_end"] = pd.to_datetime((df["summary_year"] + 2).astype(str) + "-12-31")

censor_date = pd.Timestamp("2026-02-28")
df["followup_complete"] = df["horizon_end"] <= censor_date

df["target_excluded_24m"] = 0

positive_mask = (
    df["first_excl_date"].notna() &
    (df["first_excl_date"] > df["year_end"]) &
    (df["first_excl_date"] <= df["horizon_end"])
)
df.loc[positive_mask, "target_excluded_24m"] = 1

prior_or_same_year_excl_mask = (
    df["first_excl_date"].notna() &
    (df["first_excl_date"] <= df["year_end"])
)

model_df = df[
    (~prior_or_same_year_excl_mask) &
    (df["followup_complete"])
].copy()

bad_pos = model_df[
    (model_df["target_excluded_24m"] == 1) &
    (model_df["first_excl_date"] <= model_df["year_end"])
].copy()
bad_pos.to_csv("data/interim/qa/bad_positives_after_labeling.csv", index=False)

dup_mask = model_df.duplicated(subset=["npi_clean", "summary_year"], keep=False)
dup_df = model_df.loc[dup_mask].copy()
dup_df.to_csv("data/interim/qa/provider_year_duplicates_after_labeling.csv", index=False)

if len(dup_df) > 0:
    raise ValueError(
        f"Labeled table is not unique on npi_clean + summary_year. Duplicate rows: {len(dup_df)}"
    )

print("Rows in modeling table:", len(model_df))
print("Positive count:", int(model_df["target_excluded_24m"].sum()))
print("Positive rate:", model_df["target_excluded_24m"].mean())

print("\nDropped due to prior/same-year exclusion:", int(prior_or_same_year_excl_mask.sum()))
print("Dropped due to incomplete follow-up:", int((~df["followup_complete"]).sum()))
print("Bad positives:", len(bad_pos))

print("\nPositive rate by year:")
print(model_df.groupby("summary_year")["target_excluded_24m"].mean())

model_df.to_parquet("data/processed/provider_year_labeled.parquet", index=False)

print("\nSaved:")
print("data/processed/provider_year_labeled.parquet")
print("data/interim/qa/bad_positives_after_labeling.csv")
print("data/interim/qa/provider_year_duplicates_after_labeling.csv")