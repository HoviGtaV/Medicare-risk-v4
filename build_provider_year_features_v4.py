import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

INPUT_PATH = "data/processed/provider_year_labeled.parquet"
OUTPUT_PATH = "data/processed/provider_year_features_v4.parquet"

Path("data/processed").mkdir(parents=True, exist_ok=True)

target_col = "target_excluded_24m"
group_cols = ["summary_year", "rndrng_prvdr_type"]

id_cols = [
    "npi_clean",
    "summary_year",
    "rndrng_prvdr_type",
    target_col,
]

raw_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
]

drug_med_cols = [
    "drug_tot_hcpcs_cds",
    "drug_tot_benes",
    "drug_tot_srvcs",
    "drug_sbmtd_chrg",
    "drug_mdcr_alowd_amt",
    "drug_mdcr_pymt_amt",
    "drug_mdcr_stdzd_amt",
    "med_tot_hcpcs_cds",
    "med_tot_benes",
    "med_tot_srvcs",
    "med_sbmtd_chrg",
    "med_mdcr_alowd_amt",
    "med_mdcr_pymt_amt",
    "med_mdcr_stdzd_amt",
]

demo_count_cols = [
    "bene_avg_age",
    "bene_age_lt_65_cnt",
    "bene_age_65_74_cnt",
    "bene_age_75_84_cnt",
    "bene_age_gt_84_cnt",
    "bene_feml_cnt",
    "bene_male_cnt",
    "bene_race_wht_cnt",
    "bene_race_black_cnt",
    "bene_race_api_cnt",
    "bene_race_hspnc_cnt",
    "bene_race_natind_cnt",
    "bene_race_othr_cnt",
    "bene_dual_cnt",
    "bene_ndual_cnt",
    "bene_avg_risk_scre",
]

behavioral_pct_cols = [
    "bene_cc_bh_adhd_othcd_v1_pct",
    "bene_cc_bh_alcohol_drug_v1_pct",
    "bene_cc_bh_tobacco_v1_pct",
    "bene_cc_bh_alz_nonalzdem_v2_pct",
    "bene_cc_bh_anxiety_v1_pct",
    "bene_cc_bh_bipolar_v1_pct",
    "bene_cc_bh_mood_v2_pct",
    "bene_cc_bh_depress_v1_pct",
    "bene_cc_bh_pd_v1_pct",
    "bene_cc_bh_ptsd_v1_pct",
    "bene_cc_bh_schizo_othpsy_v1_pct",
]

physical_pct_cols = [
    "bene_cc_ph_asthma_v2_pct",
    "bene_cc_ph_afib_v2_pct",
    "bene_cc_ph_cancer6_v2_pct",
    "bene_cc_ph_ckd_v2_pct",
    "bene_cc_ph_copd_v2_pct",
    "bene_cc_ph_diabetes_v2_pct",
    "bene_cc_ph_hf_nonihd_v2_pct",
    "bene_cc_ph_hyperlipidemia_v2_pct",
    "bene_cc_ph_hypertension_v2_pct",
    "bene_cc_ph_ischemicheart_v2_pct",
    "bene_cc_ph_osteoporosis_v2_pct",
    "bene_cc_ph_parkinson_v2_pct",
    "bene_cc_ph_arthritis_v2_pct",
    "bene_cc_ph_stroke_tia_v2_pct",
]

def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.divide(b.replace(0, np.nan))

def add_float32(df: pd.DataFrame, name: str, values) -> None:
    df[name] = pd.Series(values, index=df.index, dtype="float32")

def add_int8(df: pd.DataFrame, name: str, values) -> None:
    s = pd.Series(values, index=df.index)
    df[name] = s.fillna(0).astype("int8")

def add_int16(df: pd.DataFrame, name: str, values) -> None:
    s = pd.Series(values, index=df.index)
    df[name] = s.fillna(0).astype("int16")

print(f"Reading parquet schema from: {INPUT_PATH}")
schema_cols = set(pq.ParquetFile(INPUT_PATH).schema.names)

required_cols = id_cols + raw_cols
missing_required = [c for c in required_cols if c not in schema_cols]
if missing_required:
    raise ValueError(f"Missing required columns in labeled parquet: {missing_required}")

optional_cols = drug_med_cols + demo_count_cols + behavioral_pct_cols + physical_pct_cols
selected_cols = required_cols + [c for c in optional_cols if c in schema_cols]

print(f"Reading only needed columns: {len(selected_cols)} columns")
df = pd.read_parquet(INPUT_PATH, columns=selected_cols)
print("Loaded labeled table:", df.shape)

# Type cleanup
if "npi_clean" in df.columns:
    df["npi_clean"] = df["npi_clean"].astype("string")
if "rndrng_prvdr_type" in df.columns:
    df["rndrng_prvdr_type"] = df["rndrng_prvdr_type"].astype("string")

df["summary_year"] = pd.to_numeric(df["summary_year"], errors="coerce").fillna(0).astype("int16")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")

for col in df.columns:
    if col not in {"npi_clean", "rndrng_prvdr_type", "summary_year", target_col}:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

print("Step 1/6: core ratio features")

add_float32(df, "srvcs_per_bene", safe_divide(df["tot_srvcs"], df["tot_benes"]))
add_float32(df, "sbmtd_chrg_per_bene", safe_divide(df["tot_sbmtd_chrg"], df["tot_benes"]))
add_float32(df, "alowd_amt_per_bene", safe_divide(df["tot_mdcr_alowd_amt"], df["tot_benes"]))
add_float32(df, "pymt_amt_per_bene", safe_divide(df["tot_mdcr_pymt_amt"], df["tot_benes"]))
add_float32(df, "stdzd_amt_per_bene", safe_divide(df["tot_mdcr_stdzd_amt"], df["tot_benes"]))

add_float32(df, "sbmtd_chrg_per_srvc", safe_divide(df["tot_sbmtd_chrg"], df["tot_srvcs"]))
add_float32(df, "alowd_amt_per_srvc", safe_divide(df["tot_mdcr_alowd_amt"], df["tot_srvcs"]))
add_float32(df, "pymt_amt_per_srvc", safe_divide(df["tot_mdcr_pymt_amt"], df["tot_srvcs"]))
add_float32(df, "stdzd_amt_per_srvc", safe_divide(df["tot_mdcr_stdzd_amt"], df["tot_srvcs"]))

add_float32(df, "sbmtd_to_alowd_ratio", safe_divide(df["tot_sbmtd_chrg"], df["tot_mdcr_alowd_amt"]))
add_float32(df, "alowd_to_pymt_ratio", safe_divide(df["tot_mdcr_alowd_amt"], df["tot_mdcr_pymt_amt"]))
add_float32(df, "stdzd_to_pymt_ratio", safe_divide(df["tot_mdcr_stdzd_amt"], df["tot_mdcr_pymt_amt"]))
add_float32(df, "sbmtd_to_pymt_ratio", safe_divide(df["tot_sbmtd_chrg"], df["tot_mdcr_pymt_amt"]))

if "drug_tot_srvcs" in df.columns:
    add_float32(df, "drug_srvcs_share", safe_divide(df["drug_tot_srvcs"], df["tot_srvcs"]))
if "med_tot_srvcs" in df.columns:
    add_float32(df, "med_srvcs_share", safe_divide(df["med_tot_srvcs"], df["tot_srvcs"]))
if "drug_mdcr_pymt_amt" in df.columns:
    add_float32(df, "drug_pymt_share", safe_divide(df["drug_mdcr_pymt_amt"], df["tot_mdcr_pymt_amt"]))
if "med_mdcr_pymt_amt" in df.columns:
    add_float32(df, "med_pymt_share", safe_divide(df["med_mdcr_pymt_amt"], df["tot_mdcr_pymt_amt"]))
if "drug_mdcr_alowd_amt" in df.columns:
    add_float32(df, "drug_alowd_share", safe_divide(df["drug_mdcr_alowd_amt"], df["tot_mdcr_alowd_amt"]))
if "med_mdcr_alowd_amt" in df.columns:
    add_float32(df, "med_alowd_share", safe_divide(df["med_mdcr_alowd_amt"], df["tot_mdcr_alowd_amt"]))

share_specs = [
    ("bene_age_lt_65_cnt", "bene_age_lt_65_share"),
    ("bene_age_65_74_cnt", "bene_age_65_74_share"),
    ("bene_age_75_84_cnt", "bene_age_75_84_share"),
    ("bene_age_gt_84_cnt", "bene_age_gt_84_share"),
    ("bene_feml_cnt", "bene_feml_share"),
    ("bene_male_cnt", "bene_male_share"),
    ("bene_race_wht_cnt", "bene_race_wht_share"),
    ("bene_race_black_cnt", "bene_race_black_share"),
    ("bene_race_api_cnt", "bene_race_api_share"),
    ("bene_race_hspnc_cnt", "bene_race_hspnc_share"),
    ("bene_race_natind_cnt", "bene_race_natind_share"),
    ("bene_race_othr_cnt", "bene_race_othr_share"),
    ("bene_dual_cnt", "bene_dual_share"),
    ("bene_ndual_cnt", "bene_ndual_share"),
]

for src, dst in share_specs:
    if src in df.columns:
        add_float32(df, dst, safe_divide(df[src], df["tot_benes"]))

available_bh_cols = [c for c in behavioral_pct_cols if c in df.columns]
available_ph_cols = [c for c in physical_pct_cols if c in df.columns]

if available_bh_cols:
    add_float32(df, "bh_condition_pct_mean", df[available_bh_cols].mean(axis=1, skipna=True))
    add_float32(df, "bh_condition_pct_max", df[available_bh_cols].max(axis=1, skipna=True))

if available_ph_cols:
    add_float32(df, "ph_condition_pct_mean", df[available_ph_cols].mean(axis=1, skipna=True))
    add_float32(df, "ph_condition_pct_max", df[available_ph_cols].max(axis=1, skipna=True))

if "bene_avg_risk_scre" in df.columns and "bene_dual_share" in df.columns:
    add_float32(df, "risk_score_x_dual_share", df["bene_avg_risk_scre"] * df["bene_dual_share"])

gc.collect()

print("Step 2/6: log features")

log_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
    "srvcs_per_bene",
    "sbmtd_chrg_per_bene",
    "alowd_amt_per_bene",
    "pymt_amt_per_bene",
    "stdzd_amt_per_bene",
    "sbmtd_chrg_per_srvc",
    "alowd_amt_per_srvc",
    "pymt_amt_per_srvc",
    "stdzd_amt_per_srvc",
]

for col in log_cols:
    if col in df.columns:
        add_float32(df, f"log1p_{col}", np.log1p(df[col].clip(lower=0)))

gc.collect()

print("Step 3/6: peer-relative specialty-year features")

peer_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
    "srvcs_per_bene",
    "sbmtd_chrg_per_bene",
    "alowd_amt_per_bene",
    "pymt_amt_per_bene",
    "stdzd_amt_per_bene",
    "sbmtd_to_alowd_ratio",
    "alowd_to_pymt_ratio",
    "sbmtd_to_pymt_ratio",
]

for extra_col in ["bene_avg_risk_scre", "bene_dual_share", "bh_condition_pct_mean", "ph_condition_pct_mean"]:
    if extra_col in df.columns:
        peer_cols.append(extra_col)

specialty_year_groups = df.groupby(group_cols, dropna=False, sort=False)

for i, col in enumerate(peer_cols, start=1):
    print(f"  peer {i}/{len(peer_cols)}: {col}")
    median_vals = specialty_year_groups[col].transform("median")
    pct_rank_vals = specialty_year_groups[col].rank(pct=True)
    add_float32(df, f"{col}_specialty_year_median", median_vals)
    add_float32(df, f"{col}_minus_specialty_year_median", df[col] - median_vals)
    add_float32(df, f"{col}_specialty_year_pct_rank", pct_rank_vals)

gc.collect()

print("Step 4/6: temporal provider features")

df = df.sort_values(["npi_clean", "summary_year"]).reset_index(drop=True)
provider_groups = df.groupby("npi_clean", sort=False, dropna=False)

add_int8(df, "provider_observed_year_count", provider_groups.cumcount() + 1)
first_year = provider_groups["summary_year"].transform("min")
add_int16(df, "first_observed_year", first_year)
add_int8(df, "years_since_first_observed", df["summary_year"] - first_year)

prev_year = provider_groups["summary_year"].shift(1)
year_gap = df["summary_year"] - prev_year
add_int8(df, "year_gap_from_prev", year_gap.fillna(0))
add_int8(df, "observed_prev_year", (year_gap == 1).astype("int8"))

temporal_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
    "srvcs_per_bene",
    "pymt_amt_per_bene",
    "sbmtd_to_pymt_ratio",
]

for extra_col in ["bene_avg_risk_scre", "bene_dual_share", "ph_condition_pct_mean"]:
    if extra_col in df.columns:
        temporal_cols.append(extra_col)

for i, col in enumerate(temporal_cols, start=1):
    print(f"  temporal {i}/{len(temporal_cols)}: {col}")
    lag1 = provider_groups[col].shift(1)
    add_float32(df, f"{col}_lag1", lag1)
    add_float32(df, f"{col}_yoy_abs_change", df[col] - lag1)
    add_float32(df, f"{col}_yoy_pct_change", safe_divide(df[col] - lag1, lag1.abs()))
    add_int8(df, f"flag_spike_{col}_gt2x_lag1", ((lag1 > 0) & (df[col] >= 2 * lag1)).astype("int8"))

rank_change_bases = [
    "tot_mdcr_pymt_amt",
    "srvcs_per_bene",
    "sbmtd_to_pymt_ratio",
]
for base in rank_change_bases:
    rank_col = f"{base}_specialty_year_pct_rank"
    if rank_col in df.columns:
        prev_rank = provider_groups[rank_col].shift(1)
        add_float32(df, f"{base}_pct_rank_lag1", prev_rank)
        add_float32(df, f"{base}_pct_rank_yoy_change", df[rank_col] - prev_rank)

gc.collect()

print("Step 5/6: rarity flags")

flag_cols = [
    "tot_srvcs",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_alowd_amt",
    "srvcs_per_bene",
    "sbmtd_to_pymt_ratio",
]
if "bene_avg_risk_scre" in df.columns:
    flag_cols.append("bene_avg_risk_scre")

specialty_year_groups = df.groupby(group_cols, dropna=False, sort=False)
for col in flag_cols:
    pct_rank = specialty_year_groups[col].rank(pct=True)
    add_int8(df, f"flag_top1pct_{col}_within_specialty_year", (pct_rank >= 0.99).astype("int8"))
    add_int8(df, f"flag_top5pct_{col}_within_specialty_year", (pct_rank >= 0.95).astype("int8"))

print("Step 6/6: cleanup and save")

for col in df.columns:
    if pd.api.types.is_float_dtype(df[col]):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype("float32")

print("\nFeature table shape:", df.shape)
print("\nTarget counts:")
print(df[target_col].value_counts(dropna=False))
print("\nYear counts:")
print(df["summary_year"].value_counts().sort_index())
print("\nTop missing %:")
print((df.isna().mean() * 100).sort_values(ascending=False).head(30))
print("\nConstant columns:")
print([c for c in df.columns if df[c].nunique(dropna=False) <= 1])

df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
