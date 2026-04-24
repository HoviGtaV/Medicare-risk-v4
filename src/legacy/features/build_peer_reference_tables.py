import pandas as pd
from pathlib import Path

Path("data/interim/peer_reference_tables").mkdir(parents=True, exist_ok=True)

train_df = pd.read_parquet("data/processed/train_labeled.parquet")

raw_cols = [
    "tot_benes",
    "tot_srvcs",
    "tot_sbmtd_chrg",
    "tot_mdcr_alowd_amt",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
]

group_cols = ["summary_year", "rndrng_prvdr_type"]

peer_ref = train_df[group_cols + raw_cols].copy()
peer_ref = peer_ref.groupby(group_cols, dropna=False)[raw_cols].median().reset_index()

peer_ref = peer_ref.rename(columns={c: f"{c}_specialty_year_median" for c in raw_cols})

print("Peer reference shape:", peer_ref.shape)

peer_ref.to_parquet("data/interim/peer_reference_tables/train_peer_reference.parquet", index=False)

print("Saved data/interim/peer_reference_tables/train_peer_reference.parquet")