from pathlib import Path
import pandas as pd

out_dir = Path("submission/qa")
out_dir.mkdir(parents=True, exist_ok=True)

files = {
    "train_v3": "data/processed/train_v3.parquet",
    "valid_v3": "data/processed/valid_v3.parquet",
    "test_v3": "data/processed/test_v3.parquet",
}

rows = []
for name, path in files.items():
    df = pd.read_parquet(path)
    rows.append({
        "split": name,
        "rows": len(df),
        "columns": len(df.columns),
        "positives": int(df["target_excluded_24m"].sum()),
        "positive_rate": float(df["target_excluded_24m"].mean()),
        "duplicate_npi_year": int(df.duplicated(subset=["npi_clean", "summary_year"]).sum()),
        "unique_npi": int(df["npi_clean"].nunique()),
        "years_present": ",".join(map(str, sorted(df["summary_year"].unique().tolist())))
    })

summary = pd.DataFrame(rows)
summary.to_csv(out_dir / "check_final_feature_tables_v3.csv", index=False)

print(summary.to_string(index=False))
print("\nSaved:", out_dir / "check_final_feature_tables_v3.csv")
