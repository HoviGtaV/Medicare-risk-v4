import pandas as pd

cms = pd.read_csv("data/raw/cms/mup_provider_2023.csv", nrows=10000)
leie = pd.read_csv("data/raw/oig/UPDATED.csv", dtype=str)

print("CMS dtypes:\n")
print(cms.dtypes)

print("\nLEIE dtypes:\n")
print(leie.dtypes)

print("\nCMS null %")
print((cms.isna().mean() * 100).sort_values(ascending=False).head(20))

print("\nLEIE null %")
print((leie.isna().mean() * 100).sort_values(ascending=False).head(20))