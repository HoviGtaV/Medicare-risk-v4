from pathlib import Path
import pandas as pd

out_dir = Path("submission/qa")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_labeled.parquet")

target_col = "target_excluded_24m"

# detect exclusion column
excl_candidates = ["first_excl_date", "excldate", "exclusion_date"]
excl_col = next((c for c in excl_candidates if c in df.columns), None)

year_end = df["year_end"] if "year_end" in df.columns else pd.to_datetime(df["summary_year"].astype(str) + "-12-31")
horizon_end = df["horizon_end"] if "horizon_end" in df.columns else pd.to_datetime((df["summary_year"] + 2).astype(str) + "-12-31")

summary = {
    "rows": int(len(df)),
    "positives": int(df[target_col].sum()),
    "positive_rate": float(df[target_col].mean())
}

if excl_col is not None:
    excl = pd.to_datetime(df[excl_col], errors="coerce")
    bad_positive = ((df[target_col] == 1) & ((excl <= year_end) | (excl > horizon_end))).sum()
    prior_or_same_year_excl_rows = (excl <= year_end).sum()

    summary["exclusion_column_used"] = excl_col
    summary["bad_positive_rows"] = int(bad_positive)
    summary["prior_or_same_year_exclusion_rows_present"] = int(prior_or_same_year_excl_rows)

summary_df = pd.DataFrame([summary])
summary_df.to_csv(out_dir / "check_final_labels_v3.csv", index=False)

print(summary_df.to_string(index=False))
print("\nSaved:", out_dir / "check_final_labels_v3.csv")
