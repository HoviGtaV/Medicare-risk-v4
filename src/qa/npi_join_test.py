import pandas as pd
import re
from pathlib import Path

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

Path("data/interim").mkdir(parents=True, exist_ok=True)

cms = pd.read_csv("data/raw/cms/mup_provider_2023.csv", low_memory=False)
leie = pd.read_csv("data/raw/oig/UPDATED.csv", dtype=str)

cms.columns = clean_columns(cms.columns)
leie.columns = clean_columns(leie.columns)

print("CMS columns with npi:")
print([n for n in cms.columns if "npi" in n])

print("\nLEIE columns with npi:")
print([n for n in leie.columns if "npi" in n])

cms["npi_clean"] = clean_npi(cms["rndrng_npi"])
leie["npi_clean"] = clean_npi(leie["npi"])

print("\nCMS valid npi rate:", cms["npi_clean"].notna().mean())
print("LEIE valid npi rate:", leie["npi_clean"].notna().mean())

leie["excldate"] = pd.to_datetime(leie["excldate"], format="%Y%m%d", errors="coerce")
leie["reindate"] = pd.to_datetime(leie["reindate"], format="%Y%m%d", errors="coerce")

cms["summary_year"] = 2023

joined = cms.merge(
    leie[["npi_clean", "excldate", "reindate"]],
    on="npi_clean",
    how="left"
)

print("\nJoined shape:", joined.shape)
print("Exact NPI matches with excldate:", joined["excldate"].notna().sum())

year_end = pd.Timestamp("2023-12-31")
horizon_end = pd.Timestamp("2025-12-31")

joined["label_temp"] = 0
joined.loc[
    (joined["excldate"] > year_end) & (joined["excldate"] <= horizon_end),
    "label_temp"
] = 1

model_df = joined[
    (joined["excldate"].isna()) | (joined["excldate"] > year_end)
].copy()

print("\nTemporary label counts:")
print(model_df["label_temp"].value_counts(dropna=False))

cms.to_parquet("data/interim/cms_2023_clean.parquet", index=False)
leie.to_parquet("data/interim/leie_clean.parquet", index=False)
model_df.to_parquet("data/interim/provider_2023_labeled_temp.parquet", index=False)


print("\nSaved files in data/interim")