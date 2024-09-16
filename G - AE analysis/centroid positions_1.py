import os
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam

def load_spectrograms(folder_paths, sample_size):
    all_spectrograms = []
    labels = []
    
    for i, folder in enumerate(folder_paths):
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}, skipping...")
            continue
        
        print(f"Processing folder: {folder}")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        
        if not files:
            print(f"No .npy files found in {folder}, skipping...")
            continue
        
        # Randomly sample 'sample_size' files or all if less than sample_size
        sampled_files = random.sample(files, min(sample_size, len(files)))
        
        for file_path in sampled_files:
            spectrogram = np.load(file_path)
            all_spectrograms.append(spectrogram)
            labels.append(i)
    
    if not all_spectrograms:
        raise ValueError("No spectrograms were loaded. Please check the folder structure and file availability.")

    all_spectrograms = np.array(all_spectrograms)
    
    # Ensure the data has 4 dimensions
    if all_spectrograms.ndim == 3:
        all_spectrograms = np.expand_dims(all_spectrograms, axis=-1)

    return all_spectrograms, np.array(labels)

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

# Paths to the different rainforest regions (adjust paths as needed)
folder_paths = [
    r'E:\Spectrograms\PAM_1',
    r'E:\Spectrograms\PAM_2',
    r'E:\Spectrograms\PAM_3',
    r'E:\Spectrograms\PAM_4'
]

# Set sample size
sample_size = 200  # Adjust this value as needed

# Dictionary to store centroids for each month/year
centroid_data = {
    'Date': [],
    'PAM_1_x': [], 'PAM_1_y': [],
    'PAM_2_x': [], 'PAM_2_y': [],
    'PAM_3_x': [], 'PAM_3_y': [],
    'PAM_4_x': [], 'PAM_4_y': []
}

# Loop through each month/year and process spectrograms
for year in range(2020, 2024):  # Adjust year range as needed
    for month in range(1, 13):
        folder_paths_month = [
            os.path.join(folder_paths[0], f"{year}-{str(month).zfill(2)}"),
            os.path.join(folder_paths[1], f"{year}-{str(month).zfill(2)}"),
            os.path.join(folder_paths[2], f"{year}-{str(month).zfill(2)}"),
            os.path.join(folder_paths[3], f"{year}-{str(month).zfill(2)}")
        ]
        
        try:
            spectrograms, labels = load_spectrograms(folder_paths_month, sample_size)
        except ValueError:
            # If no data is found for the month/year, skip to the next
            centroid_data['Date'].append(f"{year}-{str(month).zfill(2)}")
            for loc in ['PAM_1', 'PAM_2', 'PAM_3', 'PAM_4']:
                centroid_data[f'{loc}_x'].append(np.nan)
                centroid_data[f'{loc}_y'].append(np.nan)
            continue

        input_shape = spectrograms.shape[1:]

        # Train the autoencoder
        autoencoder = build_autoencoder(input_shape)
        autoencoder.fit(spectrograms, spectrograms, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)

        # Feature extraction using the encoder
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)
        features = encoder.predict(spectrograms)
        features_flat = features.reshape(len(features), -1)

        # Dimensionality reduction using PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_flat)

        # Clustering using K-means
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(features_pca)
        centroids = kmeans.cluster_centers_

        # Store centroid positions for this month/year
        centroid_data['Date'].append(f"{year}-{str(month).zfill(2)}")
        for i in range(4):
            centroid_data[f'PAM_{i+1}_x'].append(centroids[i, 0])
            centroid_data[f'PAM_{i+1}_y'].append(centroids[i, 1])

# Convert centroid data to a DataFrame and save to CSV
centroid_df = pd.DataFrame(centroid_data)
centroid_df.to_csv(r'E:\Results\AE_all\centroid_positions.csv', index=False)

# Plot and save the centroid movements over time
for i in range(1, 5):
    plt.figure(figsize=(10, 6))
    plt.plot(centroid_df['Date'], centroid_df[f'PAM_{i}_x'], label='X Position')
    plt.plot(centroid_df['Date'], centroid_df[f'PAM_{i}_y'], label='Y Position')
    plt.title(f'Centroid Movement for PAM_{i} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Centroid Position')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'E:/Results/AE_all/PAM_{i}_centroid_movement.png')
    plt.close()

print("Centroid positions saved and plots generated.")
