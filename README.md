🎙️ Speech Emotion Recognition System

A comprehensive machine learning platform for detecting human emotions from speech signals using audio signal processing and machine learning models.

This project analyzes voice recordings, extracts MFCC (Mel Frequency Cepstral Coefficients) features, and classifies emotions such as happy, sad, angry, and neutral using supervised learning models.

🌟 Key Features

🎧 MFCC Feature Extraction from raw speech audio

🤖 Multi-Model Analysis using Support Vector Machine (SVM) and Random Forest

📂 Automatic Dataset Organization from raw RAVDESS audio files

📊 Real-Time Training & Accuracy Evaluation

🗂️ CSV Export of Predictions and Features

🎓 Academic & Research Focus for speech and signal processing

🏗️ Architecture
Core Python Pipeline

Main Scripts

generator_label.py – Extracts emotion labels from audio filenames

organize_automatically.py – Sorts audio files into emotion-wise folders

extract_mfcc.py – Extracts MFCC features, trains ML models, and evaluates accuracy

view_mfcc.py – Displays MFCC feature values for analysis

Machine Learning Models

Support Vector Machine (SVM)

Random Forest Classifier

Feature Type

MFCC (Mel Frequency Cepstral Coefficients)

📊 Model Training & Results (Image Section)

<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/2e95e138-e71d-454f-9a43-c7438d5a6968" />


The following screenshot shows the real execution of the system, including:

MFCC feature extraction

Audio file processing

Model training

Accuracy calculation

Output file generation

🧠 Model Performance
Model	Accuracy
Support Vector Machine (SVM)	~35%
Random Forest	~69%

Random Forest performs better because speech emotion data is non-linear and noisy, which tree-based ensemble models handle more effectively.

📁 Dataset Information

The system is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Emotions Supported

Neutral

Calm

Happy

Sad

Angry

Fearful

Disgust

Surprised

Each file represents emotional speech from professional actors.

🔬 Processing Pipeline
Raw Audio (.wav)
        ↓
Label Extraction
        ↓
Emotion-wise Organization
        ↓
MFCC Feature Extraction
        ↓
Machine Learning Training
        ↓
Emotion Prediction
        ↓
Accuracy Evaluation & CSV Output

🚀 Getting Started
Prerequisites

Python 3.9+

pip

Install Dependencies
pip install librosa numpy pandas scikit-learn

Step 1 – Generate Labels
python generator_label.py

Step 2 – Organize Dataset
python organize_automatically.py

Step 3 – Train Models and Predict
python extract_mfcc.py


This will:

Train SVM and Random Forest

Display accuracy

Generate mfcc_with_predictions.csv

📂 Output Files

labels.csv – Audio file to emotion mapping

mfcc_with_predictions.csv – MFCC features, true labels, and predictions

🛡️ Disclaimer

IMPORTANT: This project is for academic and educational purposes only.
