import pandas as pd

df = pd.read_parquet("data/processed/provider_year_labeled.parquet")

df["label_confidence"] = "none"
df.loc[df["match_type"] == "exact_npi", "label_confidence"] = "high"

print(df["label_confidence"].value_counts(dropna=False))

df.to_parquet("data/processed/provider_year_labeled.parquet", index=False)

print("Saved updated provider_year_labeled.parquet")