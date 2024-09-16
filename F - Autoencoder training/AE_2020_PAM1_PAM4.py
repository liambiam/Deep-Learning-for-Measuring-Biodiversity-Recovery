import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
import pandas as pd

def load_spectrograms_from_folders(folder_list, sample_size):
    all_spectrograms = []
    all_labels = []
    
    for index, folder in enumerate(folder_list):
        print(f"Processing folder: {folder}")
        files_in_folder = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.npy')]
        
        sampled_files = random.sample(files_in_folder, min(sample_size, len(files_in_folder)))
        
        for file_path in sampled_files:
            spectrogram = np.load(file_path)
            all_spectrograms.append(spectrogram)
            all_labels.append(index)
    
    if not all_spectrograms:
        raise ValueError("No spectrograms found. Check folder structure and file availability.")

    return np.array(all_spectrograms), np.array(all_labels)

def create_autoencoder_model(input_shape):
    input_image = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded_image = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded_image)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded_image = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder_model = Model(input_image, decoded_image)
    autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    
    return autoencoder_model

def analyze_monthly_data(folder_list, sample_size, encoder_model):
    spectrograms, labels = load_spectrograms_from_folders(folder_list, sample_size)
    if spectrograms.ndim == 3:
        spectrograms = np.expand_dims(spectrograms, axis=-1)
    
    features_from_encoder = encoder_model.predict(spectrograms)
    features_flattened = features_from_encoder.reshape(len(features_from_encoder), -1)
    
    pca_model = PCA(n_components=2)
    features_pca = pca_model.fit_transform(features_flattened)
    
    kmeans_model = KMeans(n_clusters=2, random_state=42)  # 2 clusters for PAM_1 and PAM_4
    cluster_assignments = kmeans_model.fit_predict(features_pca)
    
    return features_pca, labels, cluster_assignments, kmeans_model.cluster_centers_

# Base directory for spectrogram files
base_dir = r'E:\Spectrograms'

# List all months in 2020 starting from March
months_in_2020 = [f"2020-{str(month).zfill(2)}" for month in range(3, 13)]  # Start from March

# Prepare data for training the autoencoder on March 2020
march_2020_paths = []
for i in [1, 4]:  # Only PAM_1 and PAM_4
    pam_folder_path = os.path.join(base_dir, f'PAM_{i}', '2020-03')
    subdirectories = [d for d in os.listdir(pam_folder_path) if os.path.isdir(os.path.join(pam_folder_path, d))]
    if subdirectories:
        march_2020_paths.append(os.path.join(pam_folder_path, subdirectories[0]))
    else:
        print(f"Warning: No subdirectory found for PAM_{i} in 2020-03")

if len(march_2020_paths) != 2:
    raise ValueError("Cannot find data for both PAM_1 and PAM_4 in March 2020")

sample_size = 500
spectrograms_data, _ = load_spectrograms_from_folders(march_2020_paths, sample_size)

if spectrograms_data.ndim == 3:
    spectrograms_data = np.expand_dims(spectrograms_data, axis=-1)

input_shape_for_autoencoder = spectrograms_data.shape[1:]

# Train the autoencoder
autoencoder_model = create_autoencoder_model(input_shape_for_autoencoder)
training_history = autoencoder_model.fit(spectrograms_data, spectrograms_data, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)
print("Autoencoder trained on March 2020 data.")

# Create encoder model to extract features
encoder_model = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer(index=4).output)

# Process data for each month
centroid_data = []

for month in months_in_2020:
    print(f"Processing month: {month}")
    month_paths = []
    for i in [1, 4]:  # Only PAM_1 and PAM_4
        pam_month_path = os.path.join(base_dir, f'PAM_{i}', month)
        if os.path.exists(pam_month_path):
            subdirectories = [d for d in os.listdir(pam_month_path) if os.path.isdir(os.path.join(pam_month_path, d))]
            if subdirectories:
                month_paths.append(os.path.join(pam_month_path, subdirectories[0]))
            else:
                print(f"Warning: No subdirectory found for PAM_{i} in {month}")
        else:
            print(f"Warning: Path not found for PAM_{i} in {month}")
    
    if len(month_paths) == 2:  # Only process if data for both PAM locations is available
        features_pca_data, labels_data, clusters_data, centroids_data = analyze_monthly_data(month_paths, sample_size, encoder_model)
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        scatter_plot = plt.scatter(features_pca_data[:, 0], features_pca_data[:, 1], c=labels_data, cmap='viridis')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'Feature Clusters from Rainforest Regions ({month})')
        plt.colorbar(scatter_plot, label='PAM Region')
        plt.savefig(os.path.join(base_dir, f'clusters_{month}.png'))
        plt.close()
        
        # Save centroid positions
        for idx, centroid in enumerate(centroids_data):
            centroid_data.append({
                'Month': month,
                'PAM_Region': f'PAM_{1 if idx == 0 else 4}',  # Assign PAM_1 to first centroid, PAM_4 to second
                'Centroid_X': centroid[0],
                'Centroid_Y': centroid[1]
            })
    else:
        print(f"Skipping month {month} due to missing data")

# Save centroid positions to CSV
centroid_dataframe = pd.DataFrame(centroid_data)
centroid_dataframe.to_csv(os.path.join(base_dir, 'centroid_positions_2020_PAM1_PAM4.csv'), index=False)

print("Completed processing for all available months in 2020 for PAM_1 and PAM_4.")
