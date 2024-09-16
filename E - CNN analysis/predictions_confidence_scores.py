import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model_file = 'E:/Results/CNN2secondFINAL2/cnn_model.h5'
print(f"Loading model from {model_file}")
model = tf.keras.models.load_model(model_file)

# Base folder for PAM_4 spectrograms
pam4_folder = 'E:/Spectrograms/PAM_4'

# Folder for results
results_folder = 'E:/Results/CNN2secondConf'
os.makedirs(results_folder, exist_ok=True)

def get_first_200_files(base_folder):
    paths = []
    for ym_folder in os.listdir(base_folder):
        ym_path = os.path.join(base_folder, ym_folder)
        if os.path.isdir(ym_path):
            subfolders = [d for d in os.listdir(ym_path) if os.path.isdir(os.path.join(ym_path, d))]
            if subfolders:
                first_folder = subfolders[0]
                folder_path = os.path.join(ym_path, first_folder)
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
                if files:
                    chosen_files = files[:200]
                    paths.extend(chosen_files)
                    print(f"Got {len(chosen_files)} files from {first_folder}")
    return paths

# Get spectrogram file paths
print(f"Getting spectrogram paths from {pam4_folder}")
file_paths = get_first_200_files(pam4_folder)
print(f"Got {len(file_paths)} spectrogram paths")

# Function to get date from folder name
def get_date_from_folder(name):
    parts = name.split('_')
    for part in parts:
        if len(part) == 8 and part.isdigit():
            return f'{part[:4]}-{part[4:6]}-{part[6:]}'
    return 'unknown'

# List to store results
results_list = []

# Process each spectrogram
print("Starting predictions...")
for idx, path in enumerate(file_paths):
    # Load spectrogram
    spectrogram_data = np.load(path)
    spectrogram_data = spectrogram_data[np.newaxis, ..., np.newaxis]  # Add dimensions
    
    # Predict
    preds = model.predict(spectrogram_data)
    confidences = preds[0]
    
    # Get date from folder name
    folder_name = os.path.basename(os.path.dirname(path))
    date_str = get_date_from_folder(folder_name)
    
    # Save results
    results_list.append([date_str, os.path.basename(path)] + list(confidences))
    
    # Show progress
    if (idx + 1) % 10 == 0 or (idx + 1) == len(file_paths):
        print(f"Processed {idx + 1}/{len(file_paths)} files")

# Find number of classes
class_count = model.output_shape[-1]

# Create DataFrame from results
cols = ['Date', 'Spectrogram name'] + [f'Conf{i+1}' for i in range(class_count)]
df_results = pd.DataFrame(results_list, columns=cols)

# Save DataFrame to CSV
csv_file = os.path.join(results_folder, 'predictions_PAM_4.csv')
print(f"Saving to {csv_file}")
df_results.to_csv(csv_file, index=False)

print("Done.")
