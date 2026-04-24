from pathlib import Path
import re
import pandas as pd

RAW_DIR = Path("data/raw/cms")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_map = {
    "npi_clean": [
        "rndrng_npi",
        "npi",
        "provider_npi",
        "prvdr_npi",
        "rendering_npi"
    ],
    "rndrng_prvdr_type": [
        "rndrng_prvdr_type",
        "provider_type",
        "provider_type_desc",
        "prvdr_type",
        "rndrng_prvdr_spclty"
    ],
    "tot_benes": [
        "tot_benes",
        "bene_unique_cnt",
        "tot_bene_cnt"
    ],
    "tot_srvcs": [
        "tot_srvcs",
        "line_srvc_cnt",
        "tot_services",
        "srvc_cnt"
    ],
    "tot_sbmtd_chrg": [
        "tot_sbmtd_chrg",
        "tot_sbmtd_chrg_amt",
        "tot_submitted_chrg_amt",
        "tot_submitted_charge_amt"
    ],
    "tot_mdcr_alowd_amt": [
        "tot_mdcr_alowd_amt",
        "tot_mdcr_allowed_amt",
        "tot_medicare_allowed_amt"
    ],
    "tot_mdcr_pymt_amt": [
        "tot_mdcr_pymt_amt",
        "tot_mdcr_payment_amt",
        "tot_medicare_payment_amt"
    ],
    "tot_mdcr_stdzd_amt": [
        "tot_mdcr_stdzd_amt",
        "tot_mdcr_standardized_amt",
        "tot_medicare_standardized_amt"
    ],
}

def normalize_cols(cols):
    return [c.strip().lower().replace(" ", "_") for c in cols]

def find_match(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

files = sorted(RAW_DIR.glob("mup_provider_*.csv"))
if not files:
    raise FileNotFoundError(f"No CMS raw files found in {RAW_DIR}")

frames = []
schema_rows = []

for path in files:
    year_match = re.search(r"(20\d{2})", path.name)
    if not year_match:
        raise ValueError(f"Could not infer year from filename: {path.name}")
    year = int(year_match.group(1))

    header = pd.read_csv(path, nrows=0)
    original_cols = list(header.columns)
    norm_cols = normalize_cols(original_cols)
    col_map = dict(zip(norm_cols, original_cols))

    mapping = {}
    missing = []
    for final_col, candidates in candidate_map.items():
        match = find_match(norm_cols, candidates)
        if match is None:
            missing.append(final_col)
        else:
            mapping[final_col] = col_map[match]

    if missing:
        print(f"\nERROR in file: {path.name}")
        print("Available normalized columns:")
        print(norm_cols[:120])
        raise ValueError(f"Missing required columns for {path.name}: {missing}")

    usecols = list(mapping.values())
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    rename_map = {v: k for k, v in mapping.items()}
    df = df.rename(columns=rename_map)

    # standardize types
    df["npi_clean"] = (
        df["npi_clean"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(10)
    )

    for c in [
        "tot_benes",
        "tot_srvcs",
        "tot_sbmtd_chrg",
        "tot_mdcr_alowd_amt",
        "tot_mdcr_pymt_amt",
        "tot_mdcr_stdzd_amt",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["rndrng_prvdr_type"] = df["rndrng_prvdr_type"].astype(str).str.strip()
    df["summary_year"] = year

    # basic row filter: keep rows with a 10-digit NPI
    df = df[df["npi_clean"].str.len() == 10].copy()

    schema_rows.append({
        "file": path.name,
        "year": year,
        "rows_after_basic_clean": len(df),
        "columns_used": ",".join(mapping.keys())
    })

    frames.append(df)

cms_all = pd.concat(frames, ignore_index=True)
cms_all.to_parquet(OUT_DIR / "cms_all_years_clean.parquet", index=False)

cms_2023 = cms_all[cms_all["summary_year"] == 2023].copy()
cms_2023.to_parquet(OUT_DIR / "cms_2023_clean.parquet", index=False)

schema_df = pd.DataFrame(schema_rows)
schema_df.to_csv(OUT_DIR / "cms_all_years_clean_build_log.csv", index=False)

print("Saved:")
print(OUT_DIR / "cms_all_years_clean.parquet")
print(OUT_DIR / "cms_2023_clean.parquet")
print(OUT_DIR / "cms_all_years_clean_build_log.csv")
print("Final shape:", cms_all.shape)
