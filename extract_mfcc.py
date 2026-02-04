import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

dataset_folder = r"C:\Users\Shrey\OneDrive\Desktop\UROP\dataset"
n_mfcc = 13
features = []
labels = []

for emotion in os.listdir(dataset_folder):
    emotion_path = os.path.join(dataset_folder, emotion)
    if os.path.isdir(emotion_path):
        for i, file in enumerate(os.listdir(emotion_path), 1):
            if file.endswith(".wav"):
                print(f"Processing {file} ({i}) in {emotion}...")
                file_path = os.path.join(emotion_path, file)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    features.append(mfcc_mean)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Could not process {file}: {e}")

X = np.array(features)
y = np.array(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

svm_probs = svm_model.predict_proba(X_test)
rf_probs = rf_model.predict_proba(X_test)

svm_preds = svm_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

svm_accuracy = np.mean(svm_preds == y_test) * 100
rf_accuracy = np.mean(rf_preds == y_test) * 100

print(f" SVM Accuracy: {svm_accuracy:.2f}%")
print(f" Random Forest Accuracy: {rf_accuracy:.2f}%")

df_features = pd.DataFrame(X_test, columns=[f"mfcc_{i}" for i in range(n_mfcc)])
df_features['true_label'] = le.inverse_transform(y_test)
df_features['svm_pred'] = le.inverse_transform(svm_preds)
df_features['rf_pred'] = le.inverse_transform(rf_preds)
df_features['svm_accuracy_percent'] = svm_accuracy
df_features['rf_accuracy_percent'] = rf_accuracy

output_csv = os.path.join(dataset_folder, "mfcc_with_predictions.csv")
df_features.to_csv(output_csv, index=False)

print(f" Saved MFCC features with predictions to {output_csv}")
