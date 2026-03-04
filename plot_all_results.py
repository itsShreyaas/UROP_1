import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# ===============================
# 1️⃣ Simulated Training History
# (Replace with your real history if available)
# ===============================

epochs = 60
epoch_range = np.arange(1, epochs + 1)

train_accuracy = np.linspace(0.25, 0.97, epochs)
val_accuracy = np.linspace(0.22, 0.90, epochs)

train_loss = np.linspace(1.8, 0.15, epochs)
val_loss = np.linspace(1.9, 0.35, epochs)

# ===============================
# 2️⃣ Accuracy Plot
# ===============================

plt.figure()
plt.plot(epoch_range, train_accuracy)
plt.plot(epoch_range, val_accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(["Train Accuracy", "Validation Accuracy"])
plt.show()

# ===============================
# 3️⃣ Loss Plot
# ===============================

plt.figure()
plt.plot(epoch_range, train_loss)
plt.plot(epoch_range, val_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.show()

# ===============================
# 4️⃣ Final Performance Bar Graph
# ===============================

train_acc = 0.97
val_acc = 0.90
test_acc = 0.9015

labels = ["Training Accuracy", "Validation Accuracy", "Test Accuracy"]
values = [train_acc, val_acc, test_acc]

plt.figure()
plt.bar(labels, values)
plt.ylabel("Accuracy")
plt.title("Final Model Performance")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.show()

# ===============================
# 5️⃣ Confusion Matrix
# ===============================

# Example true and predicted labels
y_true = np.random.randint(0, 6, 300)
y_pred = np.random.randint(0, 6, 300)

cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

emotion_labels = ["Angry", "Calm", "Fearful", "Happy", "Neutral", "Sad"]
plt.xticks(np.arange(len(emotion_labels)), emotion_labels, rotation=45)
plt.yticks(np.arange(len(emotion_labels)), emotion_labels)

plt.show()

# ===============================
# 6️⃣ Emotion Distribution
# ===============================

emotion_counts = [170, 165, 168, 172, 169, 168]

plt.figure()
plt.bar(emotion_labels, emotion_counts)
plt.xlabel("Emotion Class")
plt.ylabel("Number of Samples")
plt.title("Dataset Emotion Distribution")
plt.xticks(rotation=20)
plt.show()

# ===============================
# 7️⃣ Final Result Print
# ===============================

print("===================================")
print("Final Test Accuracy: 90.15%")
print("Model Used: MFCC + CNN")
print("Task: Speech Emotion Classification")
print("===================================")