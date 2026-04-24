from pathlib import Path
import pandas as pd

out_dir = Path("submission/qa")
out_dir.mkdir(parents=True, exist_ok=True)

candidate_paths = [
    "data/processed/provider_year_joined.parquet",
    "data/interim/cms_leie_joined_all_years.parquet",
    "data/interim/leie_clean.parquet"
]

path = next((p for p in candidate_paths if Path(p).exists()), None)
if path is None:
    raise FileNotFoundError("No joined or LEIE file found for date consistency check.")

df = pd.read_parquet(path)

excl_candidates = ["first_excl_date", "excldate", "exclusion_date"]
rein_candidates = ["first_reindate", "reindate", "reinstatement_date"]

excl_col = next((c for c in excl_candidates if c in df.columns), None)
rein_col = next((c for c in rein_candidates if c in df.columns), None)

summary = {
    "source_file": path,
    "rows": int(len(df)),
    "exclusion_column_used": excl_col,
    "reinstatement_column_used": rein_col,
}

if excl_col is not None:
    excl = pd.to_datetime(df[excl_col], errors="coerce")
    summary["non_null_exclusion_dates"] = int(excl.notna().sum())

if rein_col is not None:
    rein = pd.to_datetime(df[rein_col], errors="coerce")
    summary["non_null_reinstatement_dates"] = int(rein.notna().sum())

if excl_col is not None and rein_col is not None:
    invalid = ((rein.notna()) & (excl.notna()) & (rein < excl)).sum()
    summary["reinstatement_before_exclusion_rows"] = int(invalid)

summary_df = pd.DataFrame([summary])
summary_df.to_csv(out_dir / "check_oig_date_consistency_v3.csv", index=False)

print(summary_df.to_string(index=False))
print("\nSaved:", out_dir / "check_oig_date_consistency_v3.csv")
