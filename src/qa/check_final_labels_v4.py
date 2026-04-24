from pathlib import Path
import pandas as pd

OUT_DIR = Path("submission/qa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

path = "data/processed/provider_year_labeled.parquet"
df = pd.read_parquet(path, columns=["npi_clean", "summary_year", "target_excluded_24m"])

year_summary = (
    df.groupby("summary_year", dropna=False)
      .agg(
          rows=("npi_clean", "size"),
          unique_npi=("npi_clean", "nunique"),
          positives=("target_excluded_24m", "sum"),
          positive_rate=("target_excluded_24m", "mean"),
      )
      .reset_index()
      .sort_values("summary_year")
)

overall = pd.DataFrame([{
    "summary_year": "ALL",
    "rows": int(len(df)),
    "unique_npi": int(df["npi_clean"].nunique()),
    "positives": int(df["target_excluded_24m"].sum()),
    "positive_rate": float(df["target_excluded_24m"].mean()),
}])

result = pd.concat([year_summary, overall], ignore_index=True)
out_path = OUT_DIR / "check_final_labels_v4.csv"
result.to_csv(out_path, index=False)

print(result.to_string(index=False))
print("\nSaved:", out_path)
