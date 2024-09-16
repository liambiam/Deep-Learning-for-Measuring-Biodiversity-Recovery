import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
import pandas as pd

def load_spectrograms_from_dirs(dir_paths, num_samples):
    all_spectros = []  # List to store all spectrograms
    all_labels = []    # List to store labels for each spectrogram
    
    # Go through each directory path
    for idx, path in enumerate(dir_paths):
        print(f"Processing directory: {path}")
        all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.npy')]
        
        # Randomly sample files
        sampled_files = random.sample(all_files, min(num_samples, len(all_files)))
        
        # Load and append each sampled file
        for file in sampled_files:
            spectrogram = np.load(file)
            all_spectros.append(spectrogram)
            all_labels.append(idx)
    
    if not all_spectros:
        raise ValueError("No spectrograms were loaded. Check folder structure and files.")

    return np.array(all_spectros), np.array(all_labels)

def create_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder part
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder part
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    
    return autoencoder

def process_and_cluster_data(dir_paths, num_samples, encoder_model):
    spectros, _ = load_spectrograms_from_dirs(dir_paths, num_samples)
    if spectros.ndim == 3:
        spectros = np.expand_dims(spectros, axis=-1)  # Add channel dimension if missing
    features = encoder_model.predict(spectros)
    flat_features = features.reshape(len(features), -1)  # Flatten features for PCA
    
    # Apply PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(flat_features)
    
    # Apply KMeans clustering to group features into clusters
    kmeans = KMeans(n_clusters=4, random_state=42)  # 4 clusters for 4 PAM locations
    cluster_labels = kmeans.fit_predict(reduced_features)
    
    return reduced_features, cluster_labels, kmeans.cluster_centers_

def get_monthly_folders(base_directory):
    month_dirs = {}
    for pam in range(1, 5):
        pam_path = os.path.join(base_directory, f'PAM_{pam}')
        for year in os.listdir(pam_path):
            year_path = os.path.join(pam_path, year)
            if os.path.isdir(year_path):
                for month in os.listdir(year_path):
                    month_path = os.path.join(year_path, month)
                    if os.path.isdir(month_path):
                        key = f"{year}-{month}"
                        if key not in month_dirs:
                            month_dirs[key] = []
                        month_dirs[key].append(month_path)
    return month_dirs

# Initialize and train autoencoder
initial_dirs = [
    r'E:\Spectrograms\PAM_1\2020-03\S4A07754_20200318_051400',
    r'E:\Spectrograms\PAM_2\2020-03\S4A07714_20200317_051400',
    r'E:\Spectrograms\PAM_3\2020-03\S4A07576_20200318_041457',
    r'E:\Spectrograms\PAM_4\2020-03\S4A10711_20200317_041400'
]

sample_count = 200
spectros, _ = load_spectrograms_from_dirs(initial_dirs, sample_count)

if spectros.ndim == 3:
    spectros = np.expand_dims(spectros, axis=-1)

input_shape = spectros.shape[1:]

# Train or load autoencoder
autoencoder_file = 'trained_autoencoder.h5'
if os.path.exists(autoencoder_file):
    autoencoder = load_model(autoencoder_file)
    print("Loaded existing autoencoder model.")
else:
    autoencoder = create_autoencoder(input_shape)
    history = autoencoder.fit(spectros, spectros, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)
    autoencoder.save(autoencoder_file)
    print("Trained and saved new autoencoder model.")

# Create encoder model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)

# Process and cluster data for each month
base_dir = r'E:\Spectrograms'
month_dirs = get_monthly_folders(base_dir)

centroid_data = []

for year_month, paths in month_dirs.items():
    if len(paths) == 4:  # Ensure all 4 PAM locations are available
        print(f"Processing {year_month}")
        pca_features, clusters, centroids = process_and_cluster_data(paths, sample_count, encoder)
        
        # Plot and save cluster results
        plt.figure(figsize=(12, 10))
        scatter_plot = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'K-means Clustering for {year_month}')
        plt.colorbar(scatter_plot, label='Cluster')
        
        plot_filename = os.path.join(base_dir, f'cluster_plot_{year_month}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Cluster plot saved as {plot_filename}.")
        
        # Save centroid coordinates
        for i, centroid in enumerate(centroids):
            centroid_data.append({
                'Month': year_month,
                'PAM': f'PAM_{i+1}',  # Map centroid to PAM locations
                'Centroid_X': centroid[0],
                'Centroid_Y': centroid[1]
            })

# Save centroid positions to a CSV file
centroid_df = pd.DataFrame(centroid_data)
centroid_csv = os.path.join(base_dir, 'centroid_positions.csv')
centroid_df.to_csv(centroid_csv, index=False)
print(f"Centroid positions saved to {centroid_csv}.")

print("Completed processing for all available months and years.")
