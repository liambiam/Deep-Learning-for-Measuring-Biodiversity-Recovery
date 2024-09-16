import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.models import load_model  # Import load_model
import random

# Function to load spectrograms
def load_spectrograms(data_folder, file_limit=10):
    spectrograms = []
    dates = []
    
    # List subfolders
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders.")
    
    for subfolder in subfolders:
        print(f"Processing subfolder: {subfolder}")
        files = [f for f in os.listdir(subfolder) if f.endswith('.npy')]
        print(f"Found {len(files)} .npy files in {subfolder}.")
        random.shuffle(files)
        
        # Apply file limit within each subfolder
        for file_name in files[:file_limit]:
            file_path = os.path.join(subfolder, file_name)
            
            # Get the parent directory name
            folder_name = os.path.basename(subfolder)
            
            # Extract date from folder name (assuming format: 'S4A10711_20220822_041100')
            date_str = folder_name.split('_')[1]  # Date is the second part after splitting by '_'
            date = pd.to_datetime(date_str, format='%Y%m%d')
            
            # Load spectrogram
            spectrogram = np.load(file_path)
            spectrograms.append(spectrogram)
            dates.append(date)
            
            print(f"Loaded file {file_name} with date {date_str}.")
            
    print(f"Total spectrograms loaded: {len(spectrograms)}")
    return np.array(spectrograms), dates

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

# Load the data
data_folder = 'E:/Spectrograms64second/PAGI_1'
print(f"Loading spectrograms from {data_folder}...")
spectrograms, dates = load_spectrograms(data_folder)

# Ensure the spectrograms have a consistent shape
print("Padding spectrograms to ensure consistent shape...")
max_height = max([s.shape[0] for s in spectrograms])
max_width = max([s.shape[1] for s in spectrograms])
spectrograms_padded = np.array([np.pad(s, ((0, max_height - s.shape[0]), (0, max_width - s.shape[1])), 'constant') for s in spectrograms])
spectrograms_padded = spectrograms_padded[..., np.newaxis]  # Add channel dimension

# Path to save/load the model
model_path = 'autoencoder_model.h5'

# Check if the model exists
if os.path.exists(model_path):
    print(f"Loading autoencoder model from {model_path}...")
    autoencoder = load_model(model_path)
    print("Model loaded successfully.")
else:
    print("Building autoencoder model...")
    autoencoder = build_autoencoder(input_shape=spectrograms_padded.shape[1:])
    print("Training autoencoder...")
    autoencoder.fit(spectrograms_padded, spectrograms_padded, epochs=10, batch_size=16, shuffle=True)
    print("Autoencoder training complete.")
    
    # Save the trained model
    print(f"Saving autoencoder model to {model_path}...")
    autoencoder.save(model_path)
    print("Model saved successfully.")

# Extract features using the encoder part of the autoencoder
print("Extracting features using encoder...")
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_1').output)
encoded_features = encoder.predict(spectrograms_padded)

# Flatten the encoded features for clustering
print("Flattening encoded features for clustering...")
n_samples, height, width, n_channels = encoded_features.shape
encoded_features_flat = encoded_features.reshape(n_samples, height * width * n_channels)

# Apply K-means clustering
print("Applying K-means clustering...")
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(encoded_features_flat)
print("Clustering complete.")

# Prepare DataFrame for temporal analysis
print("Preparing DataFrame for temporal analysis...")
df = pd.DataFrame({'date': dates, 'cluster': clusters})

# Aggregate by month
df['month'] = df['date'].dt.to_period('M')
monthly_cluster_counts = df.groupby(['month', 'cluster']).size().unstack(fill_value=0)

# Plot heatmap of cluster distribution over time
print("Plotting heatmap of cluster distribution over time...")
plt.figure(figsize=(12, 8))
plt.imshow(monthly_cluster_counts.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Month')
plt.ylabel('Cluster')
plt.title('Cluster Distribution Over Time')
plt.show()

# Plotting the clusters
print("Plotting clusters...")
plt.figure(figsize=(10, 7))
scatter = plt.scatter(encoded_features_flat[:, 0], encoded_features_flat[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('Clusters of Audio Segments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Calculate and plot the centroids over time
print("Calculating and plotting centroids over time...")
centroids = df.groupby('month')[['cluster']].mean().reset_index()

plt.figure(figsize=(10, 7))
plt.plot(centroids['month'].astype(str), centroids['cluster'])
plt.title('Centroid Movement Over Time')
plt.xlabel('Month')
plt.ylabel('Cluster Value')
plt.legend()
plt.show()
