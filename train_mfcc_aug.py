import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

print("=== MFCC TRAINING STARTED ===")

# =============================
# PATH (FIXED FOR YOUR PROJECT)
# =============================
DATA_PATH = "dataset"

# =============================
# PARAMETERS
# =============================
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174   # fixed length for MFCC
TEST_SIZE = 0.2
EPOCHS = 60
BATCH_SIZE = 32

# =============================
# LOAD + EXTRACT MFCC
# =============================
X, y = [], []

for emotion in os.listdir(DATA_PATH):
    emotion_path = os.path.join(DATA_PATH, emotion)
    if not os.path.isdir(emotion_path):
        continue

    print(f"Processing emotion: {emotion}")

    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_path, file)
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                mfcc = librosa.feature.mfcc(
                    y=signal,
                    sr=sr,
                    n_mfcc=N_MFCC
                )

                # Pad or truncate
                if mfcc.shape[1] < MAX_LEN:
                    pad_width = MAX_LEN - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)))
                else:
                    mfcc = mfcc[:, :MAX_LEN]

                X.append(mfcc)
                y.append(emotion)

            except Exception as e:
                print("Error:", e)

X = np.array(X)
X = X[..., np.newaxis]   # (samples, 40, 174, 1)

print("MFCC shape:", X.shape)

# =============================
# LABEL ENCODING
# =============================
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# =============================
# TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print("Train samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])

# =============================
# CNN MODEL (MFCC ONLY)
# =============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X.shape[1:]),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# TRAIN
# =============================
print("🚀 Training MFCC CNN model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =============================
# FINAL EVALUATION
# =============================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ FINAL TEST ACCURACY: {acc * 100:.2f}%")