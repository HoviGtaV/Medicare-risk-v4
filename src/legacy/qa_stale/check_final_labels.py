import pandas as pd

df = pd.read_parquet("data/interim/provider_all_years_labeled_temp.parquet")

print("Shape:", df.shape)
print("\nLabel counts:")
print(df["label_temp"].value_counts(dropna=False))

print("\nPositive rate:")
print(df["label_temp"].mean())

print("\nCounts by year:")
print(df.groupby("summary_year")["label_temp"].value_counts().unstack(fill_value=0))