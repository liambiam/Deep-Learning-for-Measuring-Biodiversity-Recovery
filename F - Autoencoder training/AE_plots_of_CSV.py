import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from scipy.spatial.distance import cdist

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

def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    
    return autoencoder

def plot_kmeans(features, labels, day, save_dir, centroids):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA + K-means of Encoded Features ({day})')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Label')
    
    save_path = os.path.join(save_dir, f'kmeans_plot_{day}.png')
    plt.savefig(save_path)
    plt.close()

def calculate_centroids(features, labels):
    centroids = []
    for label in np.unique(labels):
        centroid = features[labels == label].mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids)

def calculate_cluster_distances(features, labels, centroids):
    distances = {}
    for i, label in enumerate(np.unique(labels)):
        cluster_points = features[labels == label]
        centroid = centroids[i]
        dist = cdist(cluster_points, [centroid], 'euclidean')
        distances[label] = dist.mean()
    return distances

def get_day_from_path(path):
    """ Extract the date (YYYY-MM-DD) from the folder path. """
    folder_name = os.path.basename(path)
    date_str = folder_name.split('_')[1][:8]
    return pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')

# Base paths for the different rainforest regions
base_path = r'E:\Spectrograms'
locations_combinations = [
    ('PAM_1', 'PAM_3'),
    ('PAM_2', 'PAM_3'),
    ('PAM_4', 'PAM_3')
]

for loc1, loc2 in locations_combinations:
    # Get all available months
    months = os.listdir(os.path.join(base_path, loc1))
    months = [month for month in months if os.path.isdir(os.path.join(base_path, loc1, month))]

    # Filter out common months for both locations
    months = [month for month in months if os.path.isdir(os.path.join(base_path, loc2, month))]

    # Create directory for saving outputs
    save_dir = os.path.join(r'E:\Results\AE6c', f'{loc1}_vs_{loc2}')
    os.makedirs(save_dir, exist_ok=True)

    # CSV files to store centroid positions and distances
    centroid_positions_csv = os.path.join(save_dir, 'centroid_positions.csv')
    cluster_distances_csv = os.path.join(save_dir, 'cluster_distances.csv')

    centroid_positions_data = []
    cluster_distances_data = []

    # Train autoencoder on the full dataset from the first common month
    month = months[0]
    pam1_paths = [os.path.join(base_path, loc1, month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, loc1, month)))]

    pam3_paths = [os.path.join(base_path, loc2, month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, loc2, month)))]

    # Filter only the first occurrence of each date
    pam1_paths_filtered = {get_day_from_path(p): p for p in pam1_paths}
    pam3_paths_filtered = {get_day_from_path(p): p for p in pam3_paths}

    # Combine paths and load data
    all_paths = list(pam1_paths_filtered.values()) + list(pam3_paths_filtered.values())
    spectrograms, labels = load_spectrograms(all_paths, sample_size=200)

    if spectrograms.ndim == 3:
        spectrograms = np.expand_dims(spectrograms, axis=-1)

    input_shape = spectrograms.shape[1:]

    # Build and train the autoencoder
    autoencoder = build_autoencoder(input_shape)
    history = autoencoder.fit(spectrograms, spectrograms, epochs=1, batch_size=32, shuffle=True, validation_split=0.2)

    # Feature extraction using the encoder
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)

    # Iterate over each month and process
    for month in months:
        pam1_paths = [os.path.join(base_path, loc1, month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, loc1, month)))]
        pam3_paths = [os.path.join(base_path, loc2, month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, loc2, month)))]

        pam1_paths_filtered = {get_day_from_path(p): p for p in pam1_paths}
        pam3_paths_filtered = {get_day_from_path(p): p for p in pam3_paths}

        common_days = sorted(list(set(pam1_paths_filtered.keys()).intersection(set(pam3_paths_filtered.keys()))))

        for day in common_days:
            # Load spectrograms for the day
            day_paths = [pam1_paths_filtered[day], pam3_paths_filtered[day]]
            spectrograms, labels = load_spectrograms(day_paths, sample_size=200)
            
            if spectrograms.ndim == 3:
                spectrograms = np.expand_dims(spectrograms, axis=-1)
            
            features = encoder.predict(spectrograms)
            features_flat = features.reshape(len(features), -1)
            
            # Dimensionality reduction using PCA
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_flat)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans_labels = kmeans.fit_predict(features_pca)
            centroids = kmeans.cluster_centers_

            # Calculate cluster distances
            cluster_distances = calculate_cluster_distances(features_pca, kmeans_labels, centroids)
            
            # Save the PCA + k-means plot with centroids
            plot_kmeans(features_pca, kmeans_labels, day, save_dir, centroids)
            
            # Save centroid positions and distances to CSV
            centroid_positions_data.append({
                'date': day,
                'centroid_1_x': centroids[0, 0],
                'centroid_1_y': centroids[0, 1],
                'centroid_2_x': centroids[1, 0],
                'centroid_2_y': centroids[1, 1],
                'centroid_distance': np.linalg.norm(centroids[0] - centroids[1])
            })
            
            cluster_distances_data.append({
                'date': day,
                'cluster_1_mean_distance': cluster_distances[0],
                'cluster_2_mean_distance': cluster_distances[1]
            })

    # Save CSV files
    pd.DataFrame(centroid_positions_data).to_csv(centroid_positions_csv, index=False)
    pd.DataFrame(cluster_distances_data).to_csv(cluster_distances_csv, index=False)

    print(f"PCA + K-means plots, centroid positions, and cluster distances saved successfully for {loc1} vs {loc2}.")
