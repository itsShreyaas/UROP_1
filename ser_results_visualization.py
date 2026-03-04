import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

print("=== SER MODEL VISUALIZATION STARTED ===")

# =============================
# LOAD SAVED FILES
# =============================

print("Loading model and test data...")

model = load_model("saved_model.h5")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
label_classes = np.load("label_classes.npy")

with open("history.json", "r") as f:
    history = json.load(f)

print("All files loaded successfully!\n")

# =============================
# 1️⃣ ACCURACY CURVE
# =============================

plt.figure(figsize=(8,6))
plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)

plt.title("Training vs Validation Accuracy", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show(block=True)

# =============================
# 2️⃣ LOSS CURVE
# =============================

plt.figure(figsize=(8,6))
plt.plot(history['loss'], label='Training Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)

plt.title("Training vs Validation Loss", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show(block=True)

# =============================
# 3️⃣ MODEL EVALUATION
# =============================

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {acc * 100:.2f}%\n")

# =============================
# 4️⃣ CONFUSION MATRIX
# =============================

print("Generating confusion matrix...")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_classes,
    yticklabels=label_classes
)

plt.title("Confusion Matrix - Speech Emotion Recognition", fontsize=14)
plt.xlabel("Predicted Emotion", fontsize=12)
plt.ylabel("True Emotion", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show(block=True)

# =============================
# 5️⃣ CLASSIFICATION REPORT
# =============================

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_classes))

# =============================
# 6️⃣ RANDOM PREDICTION PROBABILITIES
# =============================

print("Showing predicted probabilities for 3 random samples...\n")

for i in random.sample(range(len(X_test)), 3):

    probs = y_pred_probs[i]

    plt.figure(figsize=(8,4))
    plt.bar(label_classes, probs)
    plt.title(f"Predicted Probabilities (True: {label_classes[y_true[i]]})")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

print("=== VISUALIZATION COMPLETE ===")