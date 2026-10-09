import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data=np.load("data/processed/alphabet_split.npz")
X_test, Y_test=data["X_test"], data["Y_test"]
with open("models/label_encoder.pkl", "rb") as f:
    le=pickle.load(f)

model=tf.keras.models.load_model("models/best_model.keras")
Y_pred_probs=model.predict(X_test, verbose=0)
Y_pred=np.argmax(Y_pred_probs, axis=1)
print(classification_report(Y_test, Y_pred, target_names=le.classes_, digits=3))
cm=confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix- ASL Alphabet Classifier")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("\nSaved: models/confusion_matrix.png")
