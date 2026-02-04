import os
import shutil
import pandas as pd

# Main project folder containing Actor_XX folders
source_folder = r"C:\Users\Shrey\OneDrive\Desktop\UROP"

# Destination folder where organized files will go
dataset_folder = os.path.join(source_folder, "dataset")
os.makedirs(dataset_folder, exist_ok=True)

# Path to labels.csv
labels_file = os.path.join(source_folder, "labels.csv")

# Read CSV
df = pd.read_csv(labels_file)

# Create a folder for each emotion
for emotion in df['emotion'].unique():
    os.makedirs(os.path.join(dataset_folder, emotion), exist_ok=True)

# Move files
for idx, row in df.iterrows():
    file_name = row['filename']
    emotion = row['emotion']
    file_found = False

    # Look inside all Actor_XX folders
    for folder in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder)
        if os.path.isdir(folder_path) and folder.startswith("Actor_"):
            src = os.path.join(folder_path, file_name)
            if os.path.exists(src):
                dst = os.path.join(dataset_folder, emotion, file_name)
                shutil.move(src, dst)
                file_found = True
                break

    if not file_found:
        print(f" File {file_name} not found!")

print(" All audio files have been organized automatically into dataset/ by emotion!")
