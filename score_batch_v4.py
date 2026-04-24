import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

TARGET_COL = "target_excluded_24m"
DROP_COLS = ["npi_clean", "summary_year", TARGET_COL]
SCORE_COL = "score_catboost_final_v4"
DEFAULT_MODEL_PATH = "models_v4/final/model.cbm"


def read_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .parquet")


def write_table(df: pd.DataFrame, path_str: str) -> None:
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .parquet")


def main():
    parser = argparse.ArgumentParser(description="Score provider-year rows with CatBoost final v4.")
    parser.add_argument("--input", required=True, help="Path to feature-ready input file (.csv or .parquet)")
    parser.add_argument("--output", required=True, help="Path to scored output file (.csv or .parquet)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to CatBoost model")
    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    df = read_table(args.input)
    print(f"Input shape: {df.shape}")

    required_basic = ["npi_clean", "summary_year", "rndrng_prvdr_type"]
    missing_basic = [c for c in required_basic if c not in df.columns]
    if missing_basic:
        raise ValueError(f"Missing required identifier/basic columns: {missing_basic}")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    if "rndrng_prvdr_type" not in feature_cols:
        raise ValueError("Column 'rndrng_prvdr_type' must exist in the input.")

    X = df[feature_cols].copy()

    model = CatBoostClassifier()
    model.load_model(args.model)

    print(f"Scoring with model: {args.model}")
    scores = model.predict_proba(X)[:, 1]

    out = df[["npi_clean", "summary_year", "rndrng_prvdr_type"]].copy()
    if TARGET_COL in df.columns:
        out[TARGET_COL] = df[TARGET_COL]

    out[SCORE_COL] = scores
    out = out.sort_values(SCORE_COL, ascending=False).reset_index(drop=True)

    write_table(out, args.output)
    print(f"Saved scored output: {args.output}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
