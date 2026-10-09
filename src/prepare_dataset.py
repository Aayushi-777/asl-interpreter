import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle, os
df= pd.read_csv("data/landmarks/alphabet.csv")
df=df[df["label"] !="nothing"].reset_index(drop=True)
print(f"Rows after dropping 'nothing': {len(df)}")

def normalize_row(row):
    coords=row.values.reshape(-1, 3).astype(float)
    wrist=coords[0].copy()
    coords-=wrist
    max_val=np.linalg.norm(coords, axis=1).max()
    coords/=(max_val+1e-6)
    return coords.flatten()

feature_cols=[c for c in df.columns if c!="label"]
X_raw=df[feature_cols]
Y_raw=df["label"]
print("Normalizing landmarks...")
X=np.array([normalize_row(row) for _, row in X_raw.iterrows()])
le=LabelEncoder()
Y=le.fit_transform(Y_raw)

print("Classes:", list(le.classes_))
X_train, X_temp, Y_train, Y_temp=train_test_split(X, Y, test_size=0.30, stratify=Y, random_state=42)
X_val, X_test, Y_val, Y_test=train_test_split(X_temp, Y_temp, test_size=0.50, stratify=Y_temp, random_state=42)
print(f"Train {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.savez("data/processed/alphabet_split.npz", X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Saved: data/processed/alphabet_split.npz and models/label_encoder.pkl") 