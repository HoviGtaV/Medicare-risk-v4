import pandas as pd
from pathlib import Path

Path("data/interim/qa").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_joined.parquet")

bad_rein = df[
    df["reindate"].notna() &
    df["excldate"].notna() &
    (df["reindate"] < df["excldate"])
].copy()

print("Invalid REINDATE < EXCLDATE rows:", len(bad_rein))

if len(bad_rein) > 0:
    bad_rein.to_csv("data/interim/qa/invalid_reindate_before_excldate.csv", index=False)
    print("Saved invalid rows to data/interim/qa/invalid_reindate_before_excldate.csv")