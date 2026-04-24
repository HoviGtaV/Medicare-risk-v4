import pandas as pd
import re

def clean_columns(columns):
    cleaned = []
    for c in columns:
        c = c.strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        cleaned.append(c)
    return cleaned

cms = pd.read_csv("data/raw/cms/mup_provider_2023.csv", nrows=5, low_memory=False)
leie = pd.read_csv("data/raw/oig/UPDATED.csv", nrows=5, dtype=str)

cms.columns = clean_columns(cms.columns)
leie.columns = clean_columns(leie.columns)

print("CMS cleaned columns:")
print(cms.columns.tolist())

print("\nLEIE cleaned columns:")
print(leie.columns.tolist())

import pandas as pd

cms = pd.read_csv("data/raw/cms/mup_provider_2023.csv", low_memory=False)
leie = pd.read_csv("data/raw/oig/UPDATED.csv", dtype=str)

print([n for n in cms.columns if "npi" in n.lower()])
print([n for n in leie.columns if "npi" in n.lower()])