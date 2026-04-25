from pathlib import Path
import pandas as pd

INPUT_PATH = "data/processed/test_v4.parquet"
OUT_DIR = Path("submission/docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading: {INPUT_PATH}")
df = pd.read_parquet(INPUT_PATH)
print("Loaded shape:", df.shape)

rows = []
for col in df.columns:
    s = df[col]
    non_null = s.dropna()
    sample_values = [str(x) for x in non_null.head(3).tolist()]
    rows.append({
        "column": col,
        "dtype": str(s.dtype),
        "missing_pct": float(s.isna().mean() * 100.0),
        "non_null_count": int(s.notna().sum()),
        "unique_non_null": int(non_null.nunique(dropna=True)),
        "sample_values": " | ".join(sample_values) if sample_values else ""
    })

dd = pd.DataFrame(rows).sort_values("column").reset_index(drop=True)

csv_path = OUT_DIR / "data_dictionary_v4_full.csv"
md_path = OUT_DIR / "DATA_DICTIONARY_V4_FULL.md"

dd.to_csv(csv_path, index=False)

lines = []
lines.append("# Full Data Dictionary (v4)")
lines.append("")
lines.append(f"- Source table: `{INPUT_PATH}`")
lines.append(f"- Rows: `{len(df):,}`")
lines.append(f"- Columns: `{df.shape[1]:,}`")
lines.append("")

for _, row in dd.iterrows():
    lines.append(f"## {row['column']}")
    lines.append(f"- dtype: `{row['dtype']}`")
    lines.append(f"- missing_pct: `{row['missing_pct']:.6f}`")
    lines.append(f"- non_null_count: `{row['non_null_count']}`")
    lines.append(f"- unique_non_null: `{row['unique_non_null']}`")
    lines.append(f"- sample_values: `{row['sample_values']}`")
    lines.append("")

md_path.write_text("\n".join(lines), encoding="utf-8")

print("\nSaved:")
print(csv_path)
print(md_path)
