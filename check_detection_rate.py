import pandas as pd
df=pd.read_csv("data/landmarks/alphabet.csv")
counts=df["label"].value_counts().sort_index()
print(counts)
print(f"\nTotal classes with data: {len(counts)}")