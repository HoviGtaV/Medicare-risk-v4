import pandas as pd
import re
from pathlib import Path

# ----------------------------
# Helpers
# ----------------------------
def clean_columns(columns):
    cleaned = []
    for c in columns:
        c = c.strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        cleaned.append(c)
    return cleaned

def clean_npi(series):
    s = series.astype(str).str.replace(r"\D", "", regex=True).str.strip()
    s = s.where(s.str.len() == 10, None)
    s = s.where(s != "0000000000", None)
    return s

# ----------------------------
# Make sure output folder exists
# ----------------------------
Path("data/interim").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load and combine CMS years
# ----------------------------
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
cms_frames = []

for year in years:
    path = f"data/raw/cms/mup_provider_{year}.csv"
    df = pd.read_csv(path, low_memory=False)
    df.columns = clean_columns(df.columns)
    df["npi_clean"] = clean_npi(df["rndrng_npi"])
    df["summary_year"] = year
    cms_frames.append(df)
    print(f"Loaded CMS {year}: {df.shape}")

cms_all = pd.concat(cms_frames, ignore_index=True)
print("\nCombined CMS shape:", cms_all.shape)

# ----------------------------
# Load and clean LEIE
# ----------------------------
leie = pd.read_csv("data/raw/oig/UPDATED.csv", dtype=str)
leie.columns = clean_columns(leie.columns)

print("\nLEIE columns:")
print(leie.columns.tolist())

leie["npi_clean"] = clean_npi(leie["npi"])
leie["excldate"] = pd.to_datetime(leie["excldate"], format="%Y%m%d", errors="coerce")
leie["reindate"] = pd.to_datetime(leie["reindate"], format="%Y%m%d", errors="coerce")

print("\nLEIE valid npi rate:", leie["npi_clean"].notna().mean())
print("LEIE excldate missing rate:", leie["excldate"].isna().mean())

# Keep only the columns needed for joining
leie_small = leie[["npi_clean", "excldate", "reindate"]].copy()

# ----------------------------
# Exact NPI join
# ----------------------------
joined = cms_all.merge(
    leie_small,
    on="npi_clean",
    how="left"
)

print("\nJoined shape:", joined.shape)
print("Rows with excldate after join:", joined["excldate"].notna().sum())

# ----------------------------
# Temporary 24-month label for each year
# Positive: excldate is within 24 months after end of summary_year
# Drop: excldate on or before year_end
# Negative: otherwise
# ----------------------------
year_end = pd.to_datetime(joined["summary_year"].astype(str) + "-12-31")
horizon_end = year_end + pd.DateOffset(years=2)

joined["label_temp"] = 0

positive_mask = (joined["excldate"] > year_end) & (joined["excldate"] <= horizon_end)
joined.loc[positive_mask, "label_temp"] = 1

# keep only rows with no exclusion yet by year end, or exclusion after year end
model_df = joined[(joined["excldate"].isna()) | (joined["excldate"] > year_end)].copy()

print("\nTemporary label counts:")
print(model_df["label_temp"].value_counts(dropna=False))

print("\nPositive counts by summary_year:")
print(model_df.groupby("summary_year")["label_temp"].sum())

# ----------------------------
# Save outputs
# ----------------------------
cms_all.to_parquet("data/interim/cms_all_years_clean.parquet", index=False)
leie.to_parquet("data/interim/leie_clean.parquet", index=False)
joined.to_parquet("data/interim/cms_leie_joined_all_years.parquet", index=False)
model_df.to_parquet("data/interim/provider_all_years_labeled_temp.parquet", index=False)

print("\nSaved files:")
print("- data/interim/cms_all_years_clean.parquet")
print("- data/interim/leie_clean.parquet")
print("- data/interim/cms_leie_joined_all_years.parquet")
print("- data/interim/provider_all_years_labeled_temp.parquet")