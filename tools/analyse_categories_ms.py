import pandas as pd

dataset_ms = pd.read_csv("bjarkarim.csv")

dataset_ms["Category"].value_counts().to_csv("categories_ms.csv")

dataset_ps = pd.read_csv("Clean-ContextualData22Values.csv")

dataset_ps["Genre"].value_counts().to_csv("categories_ms.csv")
