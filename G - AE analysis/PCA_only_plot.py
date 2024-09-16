import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

def load_spectrograms(folder_paths, sample_size):
    all_spectrograms = []
    labels = []
    
    for i, folder in enumerate(folder_paths):
        print(f"Processing folder: {folder}")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        
        sampled_files = random.sample(files, min(sample_size, len(files)))
        
        for file_path in sampled_files:
            spectrogram = np.load(file_path)
            all_spectrograms.append(spectrogram)
            labels.append(i)
    
    return np.array(all_spectrograms), np.array(labels)

def plot_pca(features_pca, labels, day, save_dir):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA of Raw Spectrograms ({day})')
    plt.colorbar(scatter, label='PAM Region')
    plt.savefig(os.path.join(save_dir, f'raw_pca_{day}.png'))
    plt.close()

def get_day_from_path(path):
    """ Extract the date (YYYY-MM-DD) from the folder path. """
    folder_name = os.path.basename(path)
    date_str = folder_name.split('_')[1][:8]
    return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')

# Base paths for the different rainforest regions
base_path = r'E:\Spectrograms'
locations = ['PAM_1', 'PAM_3']

# Selected dates for processing
selected_dates = [
    '2020-03-18', '2020-06-23', '2022-03-19',
    '2022-08-19', '2023-03-20', '2023-07-21',
    '2024-03-23', '2024-04-23'
]

# Convert selected dates to datetime format for comparison
selected_dates = pd.to_datetime(selected_dates)

# Create directory for saving outputs
save_dir = r'E:\Results\PCA_only_results'
os.makedirs(save_dir, exist_ok=True)

# Process each selected day
for day in selected_dates:
    day_str = day.strftime('%Y-%m-%d')
    print(f"Processing day: {day_str}")

    # Find the corresponding folders for PAM_1 and PAM_3
    pam1_path = None
    pam3_path = None
    
    for location in locations:
        year_month = day.strftime('%Y-%m')
        location_path = os.path.join(base_path, location, year_month)
        
        if os.path.exists(location_path):
            for subfolder in os.listdir(location_path):
                if get_day_from_path(subfolder) == day_str:
                    if location == 'PAM_1':
                        pam1_path = os.path.join(location_path, subfolder)
                    elif location == 'PAM_3':
                        pam3_path = os.path.join(location_path, subfolder)
                    break

    if not pam1_path or not pam3_path:
        print(f"No data for {day_str}")
        continue

    # Load spectrograms for the day
    day_paths = [pam1_path, pam3_path]
    spectrograms, labels = load_spectrograms(day_paths, sample_size=1000)

    # Flatten the spectrograms
    spectrograms_flat = spectrograms.reshape(len(spectrograms), -1)

    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(spectrograms_flat)

    # Plot PCA results
    plot_pca(features_pca, labels, day_str, save_dir)

    print(f"Processed and saved PCA plot for {day_str}")

print("All PCA plots have been saved successfully.")