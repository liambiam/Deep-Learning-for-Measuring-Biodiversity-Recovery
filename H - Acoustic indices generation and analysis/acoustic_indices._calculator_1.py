import os
import numpy as np
import librosa
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

# Base path where spectrograms are stored
base_path = 'E:/Spectrograms64second' 

# Path where results will be saved
result_path = 'E:/Code/ResultsLR'
os.makedirs(result_path, exist_ok=True)

# Class labels for spectrograms
labels = ['PAM_1', 'PAM_2', 'PAM_3', 'PAM_4']

# Function to gather spectrogram file paths with a limit on number of files per folder
def gather_spectrogram_paths(base_path, labels, max_files_per_folder=10):
    spectrogram_files = []
    for label in labels:
        class_folder = os.path.join(base_path, label)
        for root, dirs, files in os.walk(class_folder):
            npy_files = [os.path.join(root, file) for file in files if file.endswith('.npy')]
            if len(npy_files) > max_files_per_folder:
                npy_files = random.sample(npy_files, max_files_per_folder)
            spectrogram_files.extend(npy_files)
    return spectrogram_files

# Function to get acoustic features from spectrogram
def get_acoustic_features(spectrogram):
    # Assuming spectrogram is already in magnitude format
    S = np.abs(spectrogram)

    # Compute acoustic features
    rms = np.mean(librosa.feature.rms(S=S))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S))

    return rms, spectral_centroid, spectral_bandwidth

# Gather spectrogram file paths
spectrogram_files = gather_spectrogram_paths(base_path, labels)

# Load spectrograms and extract features
feature_data = []

for file in spectrogram_files:
    spectrogram = np.load(file)
    class_label = file.split(os.sep)[2]  # Get class label from file path
    rms, centroid, bandwidth = get_acoustic_features(spectrogram)
    date_str = file.split(os.sep)[3]  # Get date from file path
    feature_data.append([class_label, date_str, rms, centroid, bandwidth])

# Create DataFrame with features
df_features = pd.DataFrame(feature_data, columns=['class', 'date', 'rms', 'spectral_centroid', 'spectral_bandwidth'])

# Save features to CSV file
df_features.to_csv(os.path.join(result_path, 'acoustic_features.csv'), index=False)

print("Acoustic features data has been saved.")
