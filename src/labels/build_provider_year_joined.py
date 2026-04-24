from pathlib import Path
import pandas as pd


Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/interim/qa").mkdir(parents=True, exist_ok=True)

cms = pd.read_parquet("data/interim/cms_all_years_clean.parquet")
oig = pd.read_parquet("data/interim/oig_provider_history.parquet")

oig_small = oig[
    ["npi_clean", "first_excl_date", "first_reindate",
     "num_exclusion_records", "has_multiple_oig_rows"]
].copy()

dup_mask_before = cms.duplicated(subset=["npi_clean", "summary_year"], keep=False)
dup_before = cms.loc[dup_mask_before].copy()
dup_before.to_csv("data/interim/qa/cms_duplicate_provider_years_before_join.csv", index=False)

if len(dup_before) > 0:
    raise ValueError(
        f"CMS is not unique on npi_clean + summary_year. Duplicate rows: {len(dup_before)}"
    )

joined = cms.merge(oig_small, on="npi_clean", how="left")

dup_mask_after = joined.duplicated(subset=["npi_clean", "summary_year"], keep=False)
dup_after = joined.loc[dup_mask_after].copy()
dup_after.to_csv("data/interim/qa/provider_year_duplicates_after_join.csv", index=False)

if len(dup_after) > 0:
    raise ValueError(
        f"Joined table is not unique on npi_clean + summary_year. Duplicate rows: {len(dup_after)}"
    )

print("CMS shape:", cms.shape)
print("OIG provider history shape:", oig_small.shape)
print("Joined shape:", joined.shape)
print("Matched rows:", joined["first_excl_date"].notna().sum())
print("Matched rate:", joined["first_excl_date"].notna().mean())

joined.to_parquet("data/processed/provider_year_joined.parquet", index=False)

print("\nSaved:")
print("data/processed/provider_year_joined.parquet")
print("data/interim/qa/cms_duplicate_provider_years_before_join.csv")
print("data/interim/qa/provider_year_duplicates_after_join.csv")