from importlib.metadata import version

pkgs = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "catboost",
    "shap",
    "matplotlib",
    "joblib",
]

lines = []
for p in pkgs:
    lines.append(f"{p}=={version(p)}")

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(open("requirements.txt", "r", encoding="utf-8").read())
