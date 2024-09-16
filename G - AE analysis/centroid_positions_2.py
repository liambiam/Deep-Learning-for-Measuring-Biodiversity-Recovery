import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
import pandas as pd  # For saving centroid positions

def load_spectrograms(folder_paths, sample_size):
    all_spectrograms = []
    labels = []
    
    for i, folder in enumerate(folder_paths):
        print(f"Processing folder: {folder}")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        
        # Randomly sample 'sample_size' files or all if less than sample_size
        sampled_files = random.sample(files, min(sample_size, len(files)))
        
        for file_path in sampled_files:
            spectrogram = np.load(file_path)
            all_spectrograms.append(spectrogram)
            labels.append(i)
    
    if not all_spectrograms:
        raise ValueError("No spectrograms were loaded. Please check the folder structure and file availability.")

    return np.array(all_spectrograms), np.array(labels)

# Paths to the different rainforest regions for the initial training month/year
initial_folder_paths = [
    r'E:\Spectrograms\PAM_1\2020-03\S4A07754_20200318_051400',
    r'E:\Spectrograms\PAM_2\2020-03\S4A07714_20200317_051400',
    r'E:\Spectrograms\PAM_3\2020-03\S4A07576_20200318_041457',
    r'E:\Spectrograms\PAM_4\2020-03\S4A10711_20200317_041400'
]

# Set sample size and load spectrograms for initial training
sample_size = 200  # Adjust this value as needed
spectrograms, labels = load_spectrograms(initial_folder_paths, sample_size)

if spectrograms.ndim == 3:
    spectrograms = np.expand_dims(spectrograms, axis=-1)
elif spectrograms.ndim != 4:
    raise ValueError("Loaded spectrograms have an unexpected shape.")

input_shape = spectrograms.shape[1:]

print(f"Loaded {len(spectrograms)} spectrograms.")

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

# Training the autoencoder on the initial data
autoencoder = build_autoencoder(input_shape)
history = autoencoder.fit(spectrograms, spectrograms, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)
print("Autoencoder trained on initial data.")

# Save the trained autoencoder
autoencoder.save('trained_autoencoder.h5')

# Feature extraction using the encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)

def process_and_cluster(folder_paths, sample_size, encoder, kmeans=None):
    spectrograms, _ = load_spectrograms(folder_paths, sample_size)
    if spectrograms.ndim == 3:
        spectrograms = np.expand_dims(spectrograms, axis=-1)
    features = encoder.predict(spectrograms)
    features_flat = features.reshape(len(features), -1)
    
    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_flat)
    print("Dimensionality reduction with PCA completed.")
    
    # Clustering using K-means (either fit new or predict with existing)
    if kmeans is None:
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(features_pca)
    else:
        clusters = kmeans.predict(features_pca)
    
    print("Clustering completed.")
    
    return kmeans, features_pca, clusters

# Process and cluster the initial data to establish the centroids
initial_kmeans, initial_features_pca, initial_clusters = process_and_cluster(initial_folder_paths, sample_size, encoder)

# Function to get all subfolder paths for subsequent months
def get_subfolder_paths(base_path):
    subfolder_paths = []
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            subfolder_paths.append(os.path.join(root, dir_name))
    return subfolder_paths

# Function to get all month folder paths
def get_all_months(base_path):
    months = []
    for pam in range(1, 5):  # Assuming PAM_1 to PAM_4
        pam_folder = f'PAM_{pam}'
        pam_path = os.path.join(base_path, pam_folder)
        for year in os.listdir(pam_path):
            year_path = os.path.join(pam_path, year)
            if os.path.isdir(year_path):
                for month in os.listdir(year_path):
                    month_path = os.path.join(year_path, month)
                    if os.path.isdir(month_path):
                        months.append(month_path)
    return months

# Base path for the spectrograms
base_path = r'E:\Spectrograms'

# Get all month folder paths
month_folders = get_all_months(base_path)

# Prepare to save centroid positions
centroid_positions = []

# Loop over each month and process the first subfolder only
for month_path in month_folders:
    print(f"Processing data for month: {month_path}")
    folder_paths = get_subfolder_paths(month_path)  # Get all subfolders under each month
    if folder_paths:  # Only process if there are subfolders
        first_subfolder = folder_paths[0]  # Process only the first subfolder
        current_kmeans, features_pca, clusters = process_and_cluster([first_subfolder], sample_size, encoder, kmeans=initial_kmeans)
        
        # Save the cluster plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'K-means Clusters for {os.path.basename(month_path)}')
        plt.colorbar(scatter, label='Cluster')
        
        plot_filename = os.path.join(month_path, 'cluster_plot.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        print(f"Cluster plot saved to {plot_filename}.")
        
        # Get the current centroids
        centroids = current_kmeans.cluster_centers_
        centroid_positions.append((os.path.basename(month_path), centroids))

# Save centroid positions to a CSV file
centroid_df = pd.DataFrame(columns=['Month', 'Centroid 1', 'Centroid 2'])
for month, centroids in centroid_positions:
    for i, centroid in enumerate(centroids):
        centroid_df = centroid_df.append({'Month': month, f'Centroid {i + 1}': centroid}, ignore_index=True)

centroid_csv_path = os.path.join(base_path, 'centroid_positions.csv')
centroid_df.to_csv(centroid_csv_path, index=False)
print(f"Centroid positions saved to {centroid_csv_path}.")

print("Completed processing all months/years.")
