from pathlib import Path
import pandas as pd
import numpy as np
import re

RAW_OIG_DIR = Path("data/raw/oig")
INTERIM_DIR = Path("data/interim")
QA_DIR = INTERIM_DIR / "qa"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)


def clean_npi(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    s = s.replace(".0", "")
    s = re.sub(r"\D", "", s)
    if s == "0000000000":
        return pd.NA
    if len(s) != 10:
        return pd.NA
    return s


def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        "npi": "npi_raw",
        "excldate": "excldate_raw",
        "reindate": "reindate_raw",
        "lastname": "lastname",
        "first_name": "firstname",
        "firstname": "firstname",
        "midname": "middlename",
        "busname": "businessname",
        "businessname": "businessname",
        "address": "address",
        "addr1": "address",
        "city": "city",
        "state": "state",
        "zip": "zip",
    }

    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_map)
    return df


def load_updated_oig_csv(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = normalize_columns(df)

    df["source_file"] = path.name
    df["source_record_id"] = np.arange(len(df))

    required_cols = [
        "npi_raw", "excldate_raw", "reindate_raw",
        "lastname", "firstname", "businessname",
        "address", "city", "state", "zip"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["npi_clean"] = df["npi_raw"].apply(clean_npi)
    df["npi_valid"] = df["npi_clean"].notna()

    df["excldate"] = pd.to_datetime(df["excldate_raw"], errors="coerce")
    df["reindate"] = pd.to_datetime(df["reindate_raw"], errors="coerce")

    keep_cols = [
        "source_file",
        "source_record_id",
        "npi_raw",
        "npi_clean",
        "npi_valid",
        "excldate_raw",
        "reindate_raw",
        "excldate",
        "reindate",
        "lastname",
        "firstname",
        "businessname",
        "address",
        "city",
        "state",
        "zip",
    ]
    return df[keep_cols].copy()


def build_event_history():
    updated_file = RAW_OIG_DIR / "UPDATED.csv"
    if not updated_file.exists():
        raise FileNotFoundError("Missing data/raw/oig/UPDATED.csv")

    events = load_updated_oig_csv(updated_file)
    events = events.drop_duplicates()
    return events


def build_provider_history(events):
    valid = events[events["npi_clean"].notna()].copy()

    provider_history = (
        valid.groupby("npi_clean", as_index=False)
        .agg(
            first_excl_date=("excldate", "min"),
            first_reindate=("reindate", "min"),
            num_exclusion_records=("npi_clean", "size"),
        )
    )

    provider_history["has_multiple_oig_rows"] = (
        provider_history["num_exclusion_records"] > 1
    )

    return provider_history


def build_qa(events, provider_history):
    bad_reindate_mask = (
        events["reindate"].notna() &
        events["excldate"].notna() &
        (events["reindate"] < events["excldate"])
    )

    qa_summary = pd.DataFrame({
        "metric": [
            "event_rows_total",
            "event_rows_valid_npi",
            "event_rows_missing_or_invalid_npi",
            "event_rows_nonnull_excldate",
            "event_rows_nonnull_reindate",
            "event_rows_bad_reindate_before_excldate",
            "provider_rows_total",
            "provider_rows_multiple_oig_records",
        ],
        "value": [
            len(events),
            int(events["npi_valid"].sum()),
            int((~events["npi_valid"]).sum()),
            int(events["excldate"].notna().sum()),
            int(events["reindate"].notna().sum()),
            int(bad_reindate_mask.sum()),
            len(provider_history),
            int(provider_history["has_multiple_oig_rows"].sum()),
        ]
    })

    bad_reindates = events.loc[bad_reindate_mask].copy()
    return qa_summary, bad_reindates


def main():
    events = build_event_history()
    provider_history = build_provider_history(events)
    qa_summary, bad_reindates = build_qa(events, provider_history)

    events.to_parquet(INTERIM_DIR / "oig_event_history.parquet", index=False)
    provider_history.to_parquet(INTERIM_DIR / "oig_provider_history.parquet", index=False)
    qa_summary.to_csv(QA_DIR / "oig_history_summary.csv", index=False)
    bad_reindates.to_csv(QA_DIR / "oig_bad_reindates.csv", index=False)

    print("Saved:")
    print(INTERIM_DIR / "oig_event_history.parquet")
    print(INTERIM_DIR / "oig_provider_history.parquet")
    print(QA_DIR / "oig_history_summary.csv")
    print(QA_DIR / "oig_bad_reindates.csv")
    print()
    print(qa_summary)


if __name__ == "__main__":
    main()