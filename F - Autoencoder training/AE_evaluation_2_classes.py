import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization
from keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

# 1. Load and preprocess the audio files
def load_and_segment_audio(file_path, segment_duration=5):
    audio, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    
    segments = []
    for start in np.arange(0, total_duration, segment_duration):
        end = start + segment_duration
        if end > total_duration:
            break
        segment = audio[int(start * sr):int(end * sr)]
        segments.append(segment)
    
    return segments, sr

# 2. Convert audio segments to spectrograms
def audio_to_spectrograms(segments, sr):
    spectrograms = []
    for segment in segments:
        spect = librosa.feature.melspectrogram(y=segment, sr=sr)
        spect_db = librosa.power_to_db(spect, ref=np.max)
        spectrograms.append(spect_db)
    
    # Find the minimum width and crop all spectrograms to this width
    min_width = min(spect.shape[1] for spect in spectrograms)
    cropped_spectrograms = [spect[:, :min_width] for spect in spectrograms]
    
    cropped_spectrograms = np.array(cropped_spectrograms)
    # Changed normalization method
    cropped_spectrograms = (cropped_spectrograms - np.min(cropped_spectrograms)) / (np.max(cropped_spectrograms) - np.min(cropped_spectrograms))
    cropped_spectrograms = np.expand_dims(cropped_spectrograms, axis=-1)  # Add channel dimension
    return cropped_spectrograms

# 3. Define the autoencoder
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = BatchNormalization()(x)
    
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    
    return autoencoder

# 4. Load and preprocess audio
file_paths = [
    r"E:\Audio\PAM_1\2020_PAM_1_POS_2_PAGI\PAM 1_POS 2_Pagi_Data_download_2020-03-26\S4A07754_20200318_051400.wav",
    r"E:\Audio\PAM_3\2020_PAM_3_POS_3_PAGI\PAM 3_POS 3_Pagi_Data_Download_2020-03-24\S4A07576_20200318_041457.wav"
]

spectrograms = []
for file_path in file_paths:
    segments, sr = load_and_segment_audio(file_path)
    spectrograms.append(audio_to_spectrograms(segments, sr))

# Flatten the list of spectrograms
spectrograms = np.vstack(spectrograms)

# Set the input shape
input_shape = spectrograms.shape[1:]

# Visualize input data
plt.figure(figsize=(10, 5))
plt.imshow(spectrograms[0][:, :, 0], aspect='auto', cmap='viridis')
plt.title("Sample Input Spectrogram")
plt.colorbar()
plt.show()

# 5. Train the autoencoder
autoencoder = build_autoencoder(input_shape)
history = autoencoder.fit(spectrograms, spectrograms, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
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

# 6. Evaluate the autoencoder
decoded_spectrograms = autoencoder.predict(spectrograms)

# Calculate MSE and SSIM for each spectrogram
mse_values = []
ssim_values = []

for i in range(len(spectrograms)):
    mse_value = tf.keras.losses.MeanSquaredError()(spectrograms[i], decoded_spectrograms[i]).numpy()
    ssim_value = ssim(spectrograms[i][:, :, 0], decoded_spectrograms[i][:, :, 0], data_range=1.0)

    mse_values.append(mse_value)
    ssim_values.append(ssim_value)

    # Plot comparison for some samples
    if i < 10:  # Limiting to the first 10 for visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im1 = axes[0].imshow(spectrograms[i][:, :, 0], aspect='auto', cmap='viridis')
        axes[0].set_title("Original")
        plt.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(decoded_spectrograms[i][:, :, 0], aspect='auto', cmap='viridis')
        axes[1].set_title("Reconstructed")
        plt.colorbar(im2, ax=axes[1])
        plt.suptitle(f'Sample {i+1}: MSE={mse_value:.4f}, SSIM={ssim_value:.4f}')
        plt.show()

# Print average MSE and SSIM across the dataset
print(f'Average MSE: {np.mean(mse_values)}')
print(f'Average SSIM: {np.mean(ssim_values)}')