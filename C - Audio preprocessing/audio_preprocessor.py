import os
import librosa
import numpy as np
import pyloudnorm as pyln

# Normalize audio to EBU R128 loudness
def normalize_audio(audio, sr):
    meter = pyln.Meter(sr)  # create meter
    loudness = meter.integrated_loudness(audio)
    norm_audio = pyln.normalize.loudness(audio, loudness, -23.0)  # target loudness -23 LUFS
    return norm_audio

# Generate spectrogram (not Mel spectrogram)
def create_spectrogram(segment, sr):
    S = librosa.stft(segment)
    S_DB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_DB

# Resize spectrogram to 256x256
def resize_spectrogram(spec, size=(256, 256)):
    return np.resize(spec, size)

# Base directory for input
base_input_dir = 'E:/'

# Find all folders starting with 'PAM'
input_folders = [os.path.join(base_input_dir, folder) for folder in os.listdir(base_input_dir) if folder.startswith('PAM') and os.path.isdir(os.path.join(base_input_dir, folder))]

# Create corresponding output folders
output_base_folders = [os.path.join('E:/Spectrograms64second', os.path.basename(folder)) for folder in input_folders]

# Ensure output folders exist, create if missing
for output_base_folder in output_base_folders:
    os.makedirs(output_base_folder, exist_ok=True)

# Process each input folder and output folder
for input_folder, output_base_folder in zip(input_folders, output_base_folders):
    # Process all WAV files in each subfolder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                
                try:
                    # Load audio file
                    y, sr = librosa.load(file_path, sr=16000, mono=True)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                
                try:
                    # Normalize audio
                    y = normalize_audio(y, sr)
                except Exception as e:
                    print(f"Error normalizing {file_path}: {e}")
                    continue
                
                # Calculate number of sixty-four-second segments
                seg_duration = 64  # in seconds
                seg_length = seg_duration * sr
                
                # Determine output folder structure
                rel_path = os.path.relpath(root, input_folder)
                file_output_folder = os.path.join(output_base_folder, rel_path, os.path.splitext(filename)[0])
                os.makedirs(file_output_folder, exist_ok=True)
                
                # Generate and save spectrograms for each segment, up to 1000 spectrograms per file
                for i in range(min(len(y) // seg_length, 1000)):
                    out_filename = f'segment_{i}.npy'
                    out_path = os.path.join(file_output_folder, out_filename)
                    
                    # Skip if file exists
                    if os.path.exists(out_path):
                        print(f'Skipping existing spectrogram {out_filename} in folder {file_output_folder}')
                        continue
                    
                    segment = y[i * seg_length : (i + 1) * seg_length]
                    
                    try:
                        spec = create_spectrogram(segment, sr)
                    except Exception as e:
                        print(f"Error generating spectrogram for {file_path} segment {i}: {e}")
                        continue
                    
                    resized_spec = resize_spectrogram(spec)
                    
                    try:
                        # Save spectrogram as NumPy array
                        np.save(out_path, resized_spec)
                        print(f'Saved spectrogram {out_filename} in folder {file_output_folder}')
                    except Exception as e:
                        print(f"Error saving spectrogram {out_filename} in folder {file_output_folder}: {e}")

print('Spectrogram generation and saving complete.')
