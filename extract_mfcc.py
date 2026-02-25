import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dense, Dropout, Flatten
)

print("=== MFCC + CNN TRAINING STARTED ===")

# ======================
# CONFIG
# ======================
DATASET_PATH = "dataset"
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 173   # ~4 sec audio
BATCH_SIZE = 32
EPOCHS = 70

# ======================
# MFCC EXTRACTION
# ======================
X = []
y = []

for emotion in os.listdir(DATASET_PATH):
    emotion_path = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(emotion_path):
        continue

    print(f"Processing emotion: {emotion}")

    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_path, file)

            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            mfcc = librosa.feature.mfcc(
                y=signal,
                sr=sr,
                n_mfcc=N_MFCC
            )

            # Pad / truncate along time axis
            if mfcc.shape[1] < MAX_LEN:
                pad_width = MAX_LEN - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
            else:
                mfcc = mfcc[:, :MAX_LEN]

            X.append(mfcc)
            y.append(emotion)

X = np.array(X)
X = X[..., np.newaxis]   # (samples, 40, 173, 1)

# ======================
# LABEL ENCODING
# ======================
le = LabelEncoder()
y = le.fit_transform(y)

# ======================
# NORMALIZATION (CRITICAL)
# ======================
mean = np.mean(X)
std = np.std(X)
X = (X - mean) / std

# ======================
# TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# CNN MODEL (OPTIMAL SIZE)
# ======================
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(len(np.unique(y)), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# TRAIN
# ======================
print("🚀 Training MFCC + CNN model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ======================
# FINAL ACCURACY
# ======================
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")