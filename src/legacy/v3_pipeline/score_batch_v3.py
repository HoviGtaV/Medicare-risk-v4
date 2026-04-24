import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Input must be .csv or .parquet")

def save_table(df: pd.DataFrame, path_str: str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output must be .csv or .parquet")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Feature-ready input file (.csv or .parquet)")
    parser.add_argument("--model", default="models_v3/final/model.cbm", help="Path to CatBoost model")
    parser.add_argument("--output", required=True, help="Output scored file (.csv or .parquet)")
    args = parser.parse_args()

    df = load_table(args.input)

    always_drop = [
        "npi_clean",
        "summary_year",
        "target_excluded_24m",
    ]

    constant_drop = [
        "tot_benes_missing",
        "tot_srvcs_missing",
        "tot_sbmtd_chrg_missing",
        "tot_mdcr_alowd_amt_missing",
        "tot_mdcr_pymt_amt_missing",
        "tot_mdcr_stdzd_amt_missing",
    ]

    drop_cols = always_drop + [c for c in constant_drop if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    model = CatBoostClassifier()
    model.load_model(args.model)

    X = df[feature_cols].copy()
    scores = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["risk_score"] = scores
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)
    out["risk_rank"] = out.index + 1
    out["risk_percentile"] = 100 * (1 - (out["risk_rank"] - 1) / len(out))

    save_table(out, args.output)

    print("Scored rows:", len(out))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
