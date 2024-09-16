import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
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
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['mse'])
    
    # Create a separate encoder model
    encoder = Model(input_img, encoded)
    
    return autoencoder, encoder

def plot_loss(history, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_vs_epochs.png'))
    plt.close()

def plot_mse(history, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mse_vs_epochs.png'))
    plt.close()

def plot_ae_clustering(features, labels, day, save_dir):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='cividis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Autoencoder Clustering ({day})')
    plt.colorbar(scatter, label='PAM Region')
    plt.savefig(os.path.join(save_dir, f'ae_clustering_{day}.png'))
    plt.close()

def plot_ae_pca(features_pca, labels, day, save_dir):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='magma', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Autoencoder + PCA ({day})')
    plt.colorbar(scatter, label='PAM Region')
    plt.savefig(os.path.join(save_dir, f'ae_pca_{day}.png'))
    plt.close()

def plot_ae_kmeans(features, kmeans_labels, day, save_dir):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Autoencoder + KMeans ({day})')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(os.path.join(save_dir, f'ae_kmeans_{day}.png'))
    plt.close()

def plot_ae_pca_kmeans(features_pca, kmeans_labels, day, save_dir):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=kmeans_labels, cmap='plasma', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'Autoencoder + PCA + KMeans ({day})')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(os.path.join(save_dir, f'ae_pca_kmeans_{day}.png'))
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
save_dir = r'E:\Results\AEDresults'
os.makedirs(save_dir, exist_ok=True)

# CSV files to store centroid positions and distances
centroid_positions_csv = os.path.join(save_dir, 'centroid_positions.csv')
cluster_distances_csv = os.path.join(save_dir, 'cluster_distances.csv')

centroid_positions_data = []
cluster_distances_data = []

# Train autoencoder on the full dataset from the first common month
month = '2020-03'  # First common month or any month to initialize the training
pam1_paths = [os.path.join(base_path, 'PAM_1', month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, 'PAM_1', month)))]
pam3_paths = [os.path.join(base_path, 'PAM_3', month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, 'PAM_3', month)))]

# Filter only the first occurrence of each date
pam1_paths_filtered = {get_day_from_path(p): p for p in pam1_paths}
pam3_paths_filtered = {get_day_from_path(p): p for p in pam3_paths}

# Combine paths and load data
all_paths = list(pam1_paths_filtered.values()) + list(pam3_paths_filtered.values())
spectrograms, labels = load_spectrograms(all_paths, sample_size=500)

if spectrograms.ndim == 3:
    spectrograms = np.expand_dims(spectrograms, axis=-1)

input_shape = spectrograms.shape[1:]

# Build and train the autoencoder
autoencoder, encoder = build_autoencoder(input_shape)
history = autoencoder.fit(spectrograms, spectrograms, epochs=20, batch_size=32, shuffle=True, validation_split=0.2)

# Plot loss and MSE graphs
plot_loss(history, save_dir)
plot_mse(history, save_dir)

# Iterate over each selected day and process
for day in selected_dates:
    day_str = day.strftime('%Y-%m-%d')
    print(f"Processing day: {day_str}")

    pam1_path = pam1_paths_filtered.get(day_str)
    pam3_path = pam3_paths_filtered.get(day_str)

    if not pam1_path or not pam3_path:
        print(f"No data for {day_str}")
        continue

    # Load spectrograms for the day
    day_paths = [pam1_path, pam3_path]
    spectrograms, labels = load_spectrograms(day_paths, sample_size=500)

    if spectrograms.ndim == 3:
        spectrograms = np.expand_dims(spectrograms, axis=-1)

    features = encoder.predict(spectrograms)
    features_flat = features.reshape(len(features), -1)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_flat)

    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(features_flat)

    # Plot various graphs
    plot_ae_clustering(features_flat[:, :2], labels, day_str, save_dir)
    plot_ae_pca(features_pca, labels, day_str, save_dir)
    plot_ae_kmeans(features_flat[:, :2], kmeans_labels, day_str, save_dir)
    plot_ae_pca_kmeans(features_pca, kmeans_labels, day_str, save_dir)


# Save CSV files
pd.DataFrame(centroid_positions_data).to_csv(centroid_positions_csv, index=False)
pd.DataFrame(cluster_distances_data).to_csv(cluster_distances_csv, index=False)

print("PCA plots, centroid positions, and cluster distances saved successfully.")
