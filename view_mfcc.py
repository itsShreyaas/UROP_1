import os
import librosa
import numpy as np

dataset_folder = r"C:\Users\Shrey\OneDrive\Desktop\UROP\dataset"

for emotion in os.listdir(dataset_folder):
    emotion_folder = os.path.join(dataset_folder, emotion)
    for i, file in enumerate(os.listdir(emotion_folder), 1):
        file_path = os.path.join(emotion_folder, file)
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        print(f"\nFile: {file} | Emotion: {emotion}")
        print("MFCC shape:", mfcc.shape)
        print("MFCC mean values (13 numbers):", mfcc_mean)
