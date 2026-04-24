from pathlib import Path
import pandas as pd

Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("models_v3/final/test_scored.parquet")

target_col = "target_excluded_24m"
score_col = "score_catboost_final_v3"

ks = [100, 500, 1000, 5000]

rows = []
for k in ks:
    topk = df.head(k).copy()
    positives = int(topk[target_col].sum())
    precision = float(topk[target_col].mean())
    rows.append({
        "k": k,
        "rows_considered": len(topk),
        "positives_found": positives,
        "precision_at_k": precision,
        "avg_score": float(topk[score_col].mean()),
        "median_score": float(topk[score_col].median()),
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv("outputs/metrics/topk_summary_v3.csv", index=False)

top500_mix = (
    df.head(500)["rndrng_prvdr_type"]
    .value_counts()
    .reset_index()
)
top500_mix.columns = ["rndrng_prvdr_type", "count_top500"]
top500_mix.to_csv("outputs/metrics/top500_provider_type_mix_v3.csv", index=False)

top1000_mix = (
    df.head(1000)["rndrng_prvdr_type"]
    .value_counts()
    .reset_index()
)
top1000_mix.columns = ["rndrng_prvdr_type", "count_top1000"]
top1000_mix.to_csv("outputs/metrics/top1000_provider_type_mix_v3.csv", index=False)

print("Top-K summary:")
print(summary_df.to_string(index=False))

print("\nSaved:")
print("outputs/metrics/topk_summary_v3.csv")
print("outputs/metrics/top500_provider_type_mix_v3.csv")
print("outputs/metrics/top1000_provider_type_mix_v3.csv")
