import os
import shutil
import pandas as pd

source_folder = "."  # current folder (all Actor_XX folders are here)
dataset_folder = "dataset"   # where organized folders will go
labels_file = "labels.csv"   # CSV with filename + emotion

df = pd.read_csv(labels_file)

# Create folders for each emotion
for emo in df['emotion'].unique():
    os.makedirs(os.path.join(dataset_folder, emo), exist_ok=True)

# Move files to corresponding emotion folder
for idx, row in df.iterrows():
    file_name = row['filename']
    emotion = row['emotion']
    file_found = False
    
    # Look in all Actor_XX folders in current folder
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
        print(f"⚠️ File {file_name} not found!")

print(" All audio files have been organized into folders by emotion!")
