import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Base paths for the different rainforest regions
base_path = r'E:\Spectrograms'
locations = ['PAM_1', 'PAM_3']

# Training month
month = '2020-03'
pam1_paths = [os.path.join(base_path, 'PAM_1', month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, 'PAM_1', month)))]
pam3_paths = [os.path.join(base_path, 'PAM_3', month, subfolder) for subfolder in sorted(os.listdir(os.path.join(base_path, 'PAM_3', month)))]

# Combine paths and load data
all_paths = pam1_paths + pam3_paths
spectrograms, _ = load_spectrograms(all_paths, sample_size=1000)

if spectrograms.ndim == 3:
    spectrograms = np.expand_dims(spectrograms, axis=-1)

input_shape = spectrograms.shape[1:]

# Build and train the autoencoder
autoencoder = build_autoencoder(input_shape)
history = autoencoder.fit(spectrograms, spectrograms, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Evaluate the autoencoder
reconstructed_spectrograms = autoencoder.predict(spectrograms)
mse = mean_squared_error(spectrograms.flatten(), reconstructed_spectrograms.flatten())
mae = mean_absolute_error(spectrograms.flatten(), reconstructed_spectrograms.flatten())

print(f"Mean Squared Error (MSE) on Training Data: {mse:.4f}")
print(f"Mean Absolute Error (MAE) on Training Data: {mae:.4f}")

# Save the training history
history_csv = r'E:\Results\AE6results\training_history.csv'
history_df = pd.DataFrame(history.history)
history_df.to_csv(history_csv, index=False)

# Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = r'E:\Results\AE6results\training_loss_plot.png'
plt.savefig(loss_plot_path)
plt.show()

print("Autoencoder training complete. Evaluation metrics and loss plot saved successfully.")
