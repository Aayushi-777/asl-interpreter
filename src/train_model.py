import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

data=np.load("data/processed/alphabet_split.npz")
X_train, Y_train=data["X_train"], data["Y_train"]
X_val, Y_val=data["X_val"], data["Y_val"]
X_test, Y_test=data["X_test"], data["Y_test"]

with open("models/label_encoder.pkl", "rb") as f:
    le=pickle.load(f)

num_classes=len(le.classes_)
input_dim=X_train.shape[1]
print(f"Input dim: {input_dim} | Classes: {num_classes}")
model=models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks=[
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor="val_accuracy"),
]

history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                  epochs=100, batch_size=32, callbacks=callbacks, verbose=1,)

test_loss, test_acc=model.evaluate(X_test, Y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.4f}")
model.save("models/final_model.keras")
print("Saved: models/best_model.keras and models/final_model.keras")