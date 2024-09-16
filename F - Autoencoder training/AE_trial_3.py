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

# Function to load spectrograms
def load_spectrograms(data_path, sample_size=50):
    spectrograms = []
    months = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
    monthly_data = {}

    for month in months:
        print(f"Processing month: {month}")
        month_path = os.path.join(data_path, month)
        spectrogram_folders = [os.path.join(month_path, f) for f in os.listdir(month_path) if os.path.isdir(os.path.join(month_path, f))]
        
        if spectrogram_folders:
            first_folder = spectrogram_folders[0]
            files = [os.path.join(first_folder, f) for f in os.listdir(first_folder) if f.endswith('.npy')]
            sampled_files = random.sample(files, min(sample_size, len(files)))
            
            sampled_spectrograms = []
            for file_path in sampled_files:
                spectrogram = np.load(file_path)
                sampled_spectrograms.append(spectrogram)
            
            monthly_data[month] = np.array(sampled_spectrograms)
            spectrograms.extend(sampled_spectrograms)
    
    return np.array(spectrograms), monthly_data

data_path = 'E:/Spectrograms64second/PAM_1'
sample_size = 50
spectrograms, monthly_data = load_spectrograms(data_path, sample_size)
input_shape = spectrograms.shape[1:]

print("Loaded and sampled spectrograms.")

# Build a convolutional autoencoder
def build_autoencoder(input_shape):
    input_img = Input(shape=(input_shape[0], input_shape[1], 1))
    
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

# Preprocessing spectrograms
spectrograms = np.expand_dims(spectrograms, axis=-1)
print("Spectrograms reshaped for training.")

# Training the autoencoder
autoencoder = build_autoencoder((input_shape[0], input_shape[1], 1))
autoencoder.fit(spectrograms, spectrograms, epochs=1, batch_size=16, shuffle=True)
print("Autoencoder trained.")

# Feature extraction using the encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=4).output)
features = encoder.predict(spectrograms)
features_flat = features.reshape(len(features), -1)
print("Features extracted.")

# Dimensionality reduction
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_flat)
print("Dimensionality reduction with PCA completed.")

# Clustering
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(features_pca)
print("Clustering completed.")

# Plot clusters
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Feature Clusters')
plt.show()
print("Feature clusters plotted.")

# Track features over time
monthly_clusters = {}
for month, data in monthly_data.items():
    print(f"Tracking features for month: {month}")
    data = np.expand_dims(data, axis=-1)
    features = encoder.predict(data)
    features_flat = features.reshape(len(features), -1)
    features_pca = pca.transform(features_flat)
    clusters = kmeans.predict(features_pca)
    monthly_clusters[month] = clusters

# Convert to DataFrame for easier manipulation and plotting
df = pd.DataFrame.from_dict(monthly_clusters, orient='index').fillna(-1)  # Fill missing values with -1
print("Monthly clusters dataframe created.")

# Plot trends over time
df.T.plot()
plt.xlabel('Month')
plt.ylabel('Cluster Frequency')
plt.title('Feature Cluster Trends Over Time')
plt.legend(title='Clusters')
plt.show()
print("Feature cluster trends over time plotted.")
