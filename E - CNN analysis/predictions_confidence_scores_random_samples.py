import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the trained model
model_file_path = 'E:/Results/CNN2second/cnn_model.h5'
print(f"Loading model from {model_file_path}")
model = tf.keras.models.load_model(model_file_path)

# Base folder for PAM_3 spectrograms
pam3_base_folder = 'E:/Spectrograms/PAM_3'

# Folder to save results
results_base_folder = 'E:/Results/CNN2secondConf'
os.makedirs(results_base_folder, exist_ok=True)

def get_random_500_samples(base_folder):
    paths = []
    for ym_folder in os.listdir(base_folder):
        ym_folder_path = os.path.join(base_folder, ym_folder)
        if os.path.isdir(ym_folder_path):
            subs = [d for d in os.listdir(ym_folder_path) if os.path.isdir(os.path.join(ym_folder_path, d))]
            if subs:
                first_sub = subs[0]
                sub_path = os.path.join(ym_folder_path, first_sub)
                files = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.endswith('.npy')]
                if files:
                    chosen_files = random.sample(files, min(500, len(files)))
                    paths.extend(chosen_files)
                    print(f"Got {len(chosen_files)} files from {first_sub}")
    return paths

# Get spectrogram file paths
print(f"Getting spectrogram paths from {pam3_base_folder}")
files_paths = get_random_500_samples(pam3_base_folder)
print(f"Got {len(files_paths)} spectrogram paths")

# Function to get date from folder name
def get_date_from_folder(name):
    parts = name.split('_')
    for part in parts:
        if len(part) == 8 and part.isdigit():
            return f'{part[:4]}-{part[4:6]}-{part[6:]}'
    return 'unknown'

# List for storing results
results_list = []

# Process each spectrogram
print("Starting predictions...")
for index, path in enumerate(files_paths):
    # Load the spectrogram
    data = np.load(path)
    data = data[np.newaxis, ..., np.newaxis]  # Add dimensions
    
    # Make prediction
    preds = model.predict(data)
    confidences = preds[0]
    
    # Get date from folder name
    folder_name = os.path.basename(os.path.dirname(path))
    date_str = get_date_from_folder(folder_name)
    
    # Save results
    results_list.append([date_str, os.path.basename(path)] + list(confidences))
    
    # Show progress
    if (index + 1) % 10 == 0 or (index + 1) == len(files_paths):
        print(f"Processed {index + 1}/{len(files_paths)} files")

# Determine number of classes
class_count = model.output_shape[-1]

# Create DataFrame from results
columns_list = ['Date', 'Spectrogram file'] + [f'Conf{i+1}' for i in range(class_count)]
df_results = pd.DataFrame(results_list, columns=columns_list)

# Save DataFrame to CSV
csv_file_path = os.path.join(results_base_folder, 'predictions_PAM_3.csv')
print(f"Saving to {csv_file_path}")
df_results.to_csv(csv_file_path, index=False)

print("Done.")
