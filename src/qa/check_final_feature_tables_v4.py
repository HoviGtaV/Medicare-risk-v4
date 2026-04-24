from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


OUT_DIR = Path("submission/qa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    "train_v4": "data/processed/train_v4.parquet",
    "valid_v4": "data/processed/valid_v4.parquet",
    "test_v4": "data/processed/test_v4.parquet",
}

rows = []

for split_name, path in paths.items():
    pf = pq.ParquetFile(path)
    col_count = len(pf.schema.names)

    df = pd.read_parquet(path, columns=["npi_clean", "summary_year", "target_excluded_24m"])
    years = sorted(df["summary_year"].dropna().unique().tolist())

    rows.append({
        "split": split_name,
        "rows": int(len(df)),
        "cols": int(col_count),
        "positives": int(df["target_excluded_24m"].sum()),
        "positive_rate": float(df["target_excluded_24m"].mean()),
        "duplicate_npi_year_rows": int(df.duplicated(["npi_clean", "summary_year"]).sum()),
        "min_year": int(min(years)),
        "max_year": int(max(years)),
        "years": ",".join(str(y) for y in years),
    })

summary = pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
out_path = OUT_DIR / "check_final_feature_tables_v4.csv"
summary.to_csv(out_path, index=False)

print(summary.to_string(index=False))
print("\nSaved:", out_path)
