import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam

# Function to load spectrograms from all subfolders of the first month of each region
def load_spectrograms(data_paths, sample_size=10):
    all_spectrograms = []
    
    for data_path in data_paths:
        print(f"Processing folder: {data_path}")
        
        # Check for month directory (e.g., 2020-03)
        month_folder = os.path.join(data_path, "2020-03")
        if os.path.isdir(month_folder):
            spectrogram_folders = [os.path.join(month_folder, f) for f in os.listdir(month_folder) if os.path.isdir(os.path.join(month_folder, f))]
            
            for folder in spectrogram_folders:
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
                sampled_files = random.sample(files, min(sample_size, len(files)))
                
                for file_path in sampled_files:
                    spectrogram = np.load(file_path)
                    all_spectrograms.append(spectrogram)
    
    if not all_spectrograms:
        raise ValueError("No spectrograms were loaded. Please check the folder structure and file availability.")

    return np.array(all_spectrograms)

# Paths to the different rainforest regions
data_paths = [
    'E:/Spectrograms64second/PAM_1',
   # 'E:/Spectrograms64second/PAM_2',
    'E:/Spectrograms64second/PAM_3'
    # 'E:/Spectrograms64second/PAM_4'
]

# Load spectrograms
sample_size = 100
spectrograms = load_spectrograms(data_paths, sample_size)

# Check the shape of the loaded spectrograms
if spectrograms.ndim == 3:
    spectrograms = np.expand_dims(spectrograms, axis=-1)  # Add channel dimension if needed
elif spectrograms.ndim == 4:
    print("Spectrograms already have the correct shape.")
else:
    raise ValueError("Loaded spectrograms have an unexpected shape.")

input_shape = spectrograms.shape[1:]

print("Loaded and sampled spectrograms.")

# Build a convolutional autoencoder
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

# Training the autoencoder
autoencoder = build_autoencoder(input_shape)
autoencoder.fit(spectrograms, spectrograms, epochs=1, batch_size=16, shuffle=True)
print("Autoencoder trained.")

# Feature extraction using the encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)
features = encoder.predict(spectrograms)
features_flat = features.reshape(len(features), -1)
print("Features extracted.")

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_flat)
print("Dimensionality reduction with PCA completed.")

# Clustering using K-means
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(features_pca)
print("Clustering completed.")

# Plot clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Feature Clusters from Rainforest Regions (PCA)')
plt.colorbar(scatter)
plt.show()
print("Feature clusters plotted using PCA and K-means.")