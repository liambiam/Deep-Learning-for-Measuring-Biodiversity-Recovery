import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'E:/Results/CNN2secondFINAL2/cnn_model.h5'
print(f"Loading model from {model_path}")
model = tf.keras.models.load_model(model_path)

# Directory for PAM_2 spectrograms
pam1_dir = 'E:/Spectrograms/PAM_2'

# Results directory
results_dir = 'E:/Results/CNN2secondConf2'
os.makedirs(results_dir, exist_ok=True)

def load_second_200_samples(base_dir):
    spectrogram_paths = []
    for year_month_folder in os.listdir(base_dir):
        year_month_path = os.path.join(base_dir, year_month_folder)
        if os.path.isdir(year_month_path):
            subfolders = sorted([f for f in os.listdir(year_month_path) if os.path.isdir(os.path.join(year_month_path, f))])
            if len(subfolders) > 1:  # Check for a second subfolder
                second_subfolder = subfolders[1]
                subfolder_path = os.path.join(year_month_path, second_subfolder)
                files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                if files:
                    selected_files = files[:200]  # Up to 200 files
                    spectrogram_paths.extend(selected_files)
                    print(f"Collected {len(selected_files)} files from {second_subfolder}")
    return spectrogram_paths

# Collect spectrogram paths
print(f"Collecting spectrogram paths from {pam1_dir}")
spectrogram_paths = load_second_200_samples(pam1_dir)
print(f"Collected {len(spectrogram_paths)} spectrogram paths")

# Function to extract date from folder name
def extract_date(folder_name):
    parts = folder_name.split('_')
    for part in parts:
        if len(part) == 8 and part.isdigit():
            return f'{part[:4]}-{part[4:6]}-{part[6:]}'
    return 'unknown'

# Initialize results list
results = []

# Process each spectrogram
print("Making predictions...")
for i, file_path in enumerate(spectrogram_paths):
    # Load and preprocess the spectrogram
    spectrogram = np.load(file_path)
    spectrogram = spectrogram[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
    
    # Make predictions
    predictions = model.predict(spectrogram)
    confidence_scores = predictions[0]
    
    # Extract the date from the folder name
    folder_name = os.path.basename(os.path.dirname(file_path))
    date = extract_date(folder_name)
    
    # Store the results
    results.append([date, os.path.basename(file_path)] + list(confidence_scores))
    
    # Print progress
    if (i + 1) % 10 == 0 or (i + 1) == len(spectrogram_paths):
        print(f"Processed {i + 1}/{len(spectrogram_paths)} spectrograms")

# Create DataFrame and save to CSV
num_classes = model.output_shape[-1]
df_columns = ['Date', 'Spectrogram number'] + [f'Conf{i + 1}' for i in range(num_classes)]
df = pd.DataFrame(results, columns=df_columns)

csv_path = os.path.join(results_dir, 'predictions_PAM_2.csv')
print(f"Saving results to {csv_path}")
df.to_csv(csv_path, index=False)

print("Processing complete.")
