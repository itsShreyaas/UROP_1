import os
import csv

source_folder = r"C:\Users\Shrey\OneDrive\Desktop\UROP"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

rows = [["filename", "emotion"]]

for folder in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder)
    if os.path.isdir(folder_path) and folder.startswith("Actor_"):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, "unknown")
                rows.append([file, emotion])

with open(os.path.join(source_folder, "labels.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(" labels.csv generated successfully!")
