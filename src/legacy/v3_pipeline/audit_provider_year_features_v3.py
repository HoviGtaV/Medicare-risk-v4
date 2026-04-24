import pandas as pd
from pathlib import Path

Path("outputs/eda").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/provider_year_features_v3.parquet")

shape = df.shape
year_counts = df["summary_year"].value_counts().sort_index()
target_counts = df["target_excluded_24m"].value_counts(dropna=False)
target_rate = df["target_excluded_24m"].mean()
dup_count = df.duplicated(subset=["npi_clean", "summary_year"]).sum()

missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
constant_cols = pd.DataFrame({
    "column": [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
})

dtype_df = df.dtypes.astype(str).reset_index()
dtype_df.columns = ["column", "dtype"]

report_lines = []
report_lines.append(f"shape: {shape}")
report_lines.append("")
report_lines.append("year counts:")
report_lines.append(year_counts.to_string())
report_lines.append("")
report_lines.append("target counts:")
report_lines.append(target_counts.to_string())
report_lines.append(f"")
report_lines.append(f"target rate: {target_rate}")
report_lines.append(f"duplicate (npi_clean, summary_year) rows: {dup_count}")
report_lines.append("")
report_lines.append("top missing %:")
report_lines.append(missing_pct.head(30).to_string())
report_lines.append("")
report_lines.append("constant columns:")
report_lines.append(constant_cols.to_string(index=False))
report_lines.append("")
report_lines.append("dtype summary:")
report_lines.append(dtype_df.to_string(index=False))

Path("outputs/eda/feature_table_v3_audit.txt").write_text("\n".join(report_lines), encoding="utf-8")
missing_pct.reset_index().rename(columns={"index": "column", 0: "missing_pct"}).to_csv("outputs/metrics/feature_table_v3_missingness.csv", index=False)
constant_cols.to_csv("outputs/metrics/feature_table_v3_constants.csv", index=False)
dtype_df.to_csv("outputs/metrics/feature_table_v3_dtypes.csv", index=False)

print("\n=== V3 AUDIT ===")
print("\n".join(report_lines))
print("\nSaved:")
print("outputs/eda/feature_table_v3_audit.txt")
print("outputs/metrics/feature_table_v3_missingness.csv")
print("outputs/metrics/feature_table_v3_constants.csv")
print("outputs/metrics/feature_table_v3_dtypes.csv")
