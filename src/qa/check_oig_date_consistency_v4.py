from pathlib import Path
import pandas as pd

OUT_DIR = Path("submission/qa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

path = "data/raw/oig/UPDATED.csv"
df = pd.read_csv(path, dtype=str).fillna("")

def parse_yyyymmdd(series):
    cleaned = series.replace({"": pd.NA, "00000000": pd.NA})
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")

df["excl_date_parsed"] = parse_yyyymmdd(df["EXCLDATE"])
df["reindate_parsed"] = parse_yyyymmdd(df["REINDATE"])
df["waiverdate_parsed"] = parse_yyyymmdd(df["WAIVERDATE"])

summary = pd.DataFrame([{
    "total_rows": int(len(df)),
    "rows_with_nonzero_npi": int((df["NPI"].fillna("") != "0000000000").sum()),
    "missing_excldate_rows": int(df["excl_date_parsed"].isna().sum()),
    "reindate_before_excldate_rows": int(((df["reindate_parsed"].notna()) & (df["excl_date_parsed"].notna()) & (df["reindate_parsed"] < df["excl_date_parsed"])).sum()),
    "waiverdate_before_excldate_rows": int(((df["waiverdate_parsed"].notna()) & (df["excl_date_parsed"].notna()) & (df["waiverdate_parsed"] < df["excl_date_parsed"])).sum()),
}])

out_path = OUT_DIR / "check_oig_date_consistency_v4.csv"
summary.to_csv(out_path, index=False)

print(summary.to_string(index=False))
print("\nSaved:", out_path)