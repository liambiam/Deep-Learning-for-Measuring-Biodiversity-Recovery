import os
import random
import numpy as np
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

# Paths to the different rainforest regions
folder_paths = [
    r'E:\Spectrograms\PAM_1\2020-03\S4A07754_20200318_051400',
 #   r'E:\Spectrograms\PAM_2\2020-03\S4A07714_20200317_051400'
    r'E:\Spectrograms\PAM_3\2020-03\S4A07576_20200318_041457'
  #  r'E:\Spectrograms\PAM_4\2020-03\S4A10711_20200317_041400'
]

# Set sample size and load spectrograms
sample_size = 1000  # Adjust this value as needed
spectrograms, labels = load_spectrograms(folder_paths, sample_size)

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

# Training the autoencoder
autoencoder = build_autoencoder(input_shape)
history = autoencoder.fit(spectrograms, spectrograms, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)
print("Autoencoder trained.")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

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

# Plot clusters based on original labels
plt.figure(figsize=(12, 10))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'Feature Clusters from Rainforest Regions (Sample Size: {sample_size})')
plt.colorbar(scatter, label='PAM Region')
plt.show()

# Plot K-means clusters
plt.figure(figsize=(12, 10))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'K-means Clusters of Rainforest Regions (Sample Size: {sample_size})')
plt.colorbar(scatter, label='Cluster')
plt.show()

print("Feature clusters plotted.")