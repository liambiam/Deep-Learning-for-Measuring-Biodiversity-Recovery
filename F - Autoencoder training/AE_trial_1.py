import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.manifold import TSNE
import cv2
import pandas as pd
import random

def load_spectrograms(data_folder, num_per_subfolder, target_width=256):
    X = []
    y = []
    dates = []
    categories = os.listdir(data_folder)

    for category in categories:
        category_path = os.path.join(data_folder, category)
        label = categories.index(category)
        
        for subfolder in os.listdir(category_path):
            if '2020' in subfolder:  # Check if '2020' is in the subfolder name
                subfolder_path = os.path.join(category_path, subfolder)
                spectrogram_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                
                # Randomly sample spectrograms from the subfolder
                sampled_files = random.sample(spectrogram_files, min(len(spectrogram_files), num_per_subfolder))

                for file_name in sampled_files:
                    file_path = os.path.join(subfolder_path, file_name)
                    spectrogram = np.load(file_path)
                    spectrogram = cv2.resize(spectrogram, (target_width, spectrogram.shape[0]))  # Resize the spectrogram
                    X.append(spectrogram)
                    y.append(label)
                    
                    # Extract date from the subfolder name
                    date_str = subfolder.split('_')[1]  # Assumes the date is the second part after splitting by '_'
                    date = pd.to_datetime(date_str, format='%Y%m%d')
                    dates.append(date)

    X = np.array(X)
    y = np.array(y)
    dates = np.array(dates)
    return X, y, dates

# Load the data
data_folder = 'E:/Spectrograms/PAM_1'  # Specify the path to the data folder
num_per_subfolder = 10  # Specify the number of spectrograms per subfolder

X, y, dates = load_spectrograms(data_folder, num_per_subfolder)

# Ensure the input data has the correct shape (num_samples, height, width, channels)
X = X[..., np.newaxis]  # Add channel dimension

# Define the autoencoder model
input_shape = X.shape[1:]

# Reshape the input data
X = X.reshape((-1, *input_shape))

input_img = Input(shape=input_shape)

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X, X, epochs=10, batch_size=32, shuffle=True, validation_split=0.2)

# Define the encoder model to extract the encoded representations
encoder = Model(input_img, encoded)

# Encode the spectrograms
encoded_spectrograms = encoder.predict(X)

# Flatten the encoded representations for t-SNE
encoded_spectrograms_flat = encoded_spectrograms.reshape((encoded_spectrograms.shape[0], -1))

# Apply t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(encoded_spectrograms_flat)

# Create a DataFrame with the t-SNE results and dates
tsne_df = pd.DataFrame(tsne_result, columns=['Component 1', 'Component 2'])
tsne_df['Date'] = dates
tsne_df['Label'] = y

# Plot the t-SNE results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_df['Component 1'], tsne_df['Component 2'], c=tsne_df['Label'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE of Encoded Spectrograms')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# Plot the clusters over time
for date in tsne_df['Date'].unique():
    subset = tsne_df[tsne_df['Date'] == date]
    plt.scatter(subset['Component 1'], subset['Component 2'], label=str(date.date()), alpha=0.7)

plt.legend()
plt.title('Clusters Over Time')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

# Calculate and plot the centroids over time
centroids = tsne_df.groupby('Date')[['Component 1', 'Component 2']].mean().reset_index()

plt.figure(figsize=(10, 7))
plt.plot(centroids['Date'], centroids['Component 1'], label='Component 1')
plt.plot(centroids['Date'], centroids['Component 2'], label='Component 2')
plt.title('Centroid Movement Over Time')
plt.xlabel('Date')
plt.ylabel('Component Value')
plt.legend()
plt.show()
