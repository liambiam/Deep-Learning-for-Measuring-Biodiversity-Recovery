import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Folder where spectrograms are stored
base_folder = r"E:/PAM_1/2020_PAM_1_POS_2_PAGI/"
# Folder where results will be stored
results_path = r"E:/FeaturesResults/"

# Ensure results folder exists
os.makedirs(results_path, exist_ok=True)

def compute_acoustic_indices(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        # Compute Short-Time Fourier Transform (STFT) for spectral info
        S = np.abs(librosa.stft(y))**2
        freqs = librosa.fft_frequencies(sr=sr)
        times = librosa.times_like(S)
        
        # Calculate various acoustic measures
        aci = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        adi = np.mean(np.log(np.sum(S, axis=0)))
        bi = np.mean(librosa.feature.rms(y=y))
        te = entropy(np.histogram(np.mean(S, axis=0), bins=50)[0])
        ndsi = np.mean(librosa.feature.spectral_flatness(y=y))
        sdi = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_entropy = entropy(np.mean(S, axis=1))
        evenness = np.sum(S, axis=0) / np.sum(np.sum(S, axis=0))
        sbi = np.mean(librosa.feature.zero_crossing_rate(y=y))
        ari = np.sum(np.mean(S, axis=0) > np.median(np.mean(S, axis=0)))
        dominance_index = np.max(np.mean(S, axis=0)) / np.sum(np.mean(S, axis=0))
        
        return {
            'Acoustic Complexity Index (ACI)': aci,
            'Acoustic Diversity Index (ADI)': adi,
            'Bioacoustic Index (BI)': bi,
            'Temporal Entropy (TE)': te,
            'Normalized Difference Soundscape Index (NDSI)': ndsi,
            'Soundscape Diversity Index (SDI)': sdi,
            'Spectral Entropy': spectral_entropy,
            'Acoustic Evenness Index': evenness,
            'Songbird Diversity Index (SDI)': sbi,
            'Acoustic Richness Index (ARI)': ari,
            'Dominance Index': dominance_index
        }
    except Exception as e:
        print(f"Something went wrong with file {file_path}: {e}")
        return None

def analyze_folder(folder_path):
    print(f"Looking at folder: {folder_path}")
    results_for_month = {}
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            print(f"Processing directory: {directory}")
            folder_path = os.path.join(root, directory)
            wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
            
            if len(wav_files) < 2:
                print(f"Not enough .wav files in {folder_path}")
                continue
            
            indices_for_files = []
            for i, wav_file in enumerate(wav_files[:2]):  # Limit to first 2 files
                file_path = os.path.join(folder_path, wav_file)
                print(f"Working on file: {file_path}")
                indices = compute_acoustic_indices(file_path)
                if indices:
                    indices_for_files.append(indices)
            
            if indices_for_files:
                average_indices = {key: np.mean([indices[key] for indices in indices_for_files]) for key in indices_for_files[0]}
                results_for_month[directory] = average_indices
                print(f"Done with {directory}: {average_indices}")

    return results_for_month

def plot_and_save_results(results, result_path):
    print(f"Saving plots to {result_path}")
    indices = list(next(iter(results.values())).keys())
    
    for index in indices:
        values = [result[index] for result in results.values()]
        months = list(results.keys())
        
        plt.figure(figsize=(10, 6))
        plt.bar(months, values, color='skyblue')
        plt.xlabel('Month')
        plt.ylabel(index)
        plt.title(f'{index} Over Time')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        file_name = os.path.join(result_path, f"{index}.png")
        plt.savefig(file_name)
        plt.close()
        print(f"Plot for {index} saved at {file_name}")

def save_results_to_csv(results, result_path):
    print(f"Saving results to CSV at {result_path}")
    # Convert results to DataFrame
    df = pd.DataFrame(results).T
    csv_file = os.path.join(result_path, "acoustic_indices_results.csv")
    df.to_csv(csv_file, index=True)
    print(f"Results saved to {csv_file}")

def main():
    print("Starting analysis now...")
    all_results = analyze_folder(base_folder)
    print("Analysis done. Making plots and saving results...")
    plot_and_save_results(all_results, results_path)
    save_results_to_csv(all_results, results_path)
    print("All results are saved in", results_path)

if __name__ == "__main__":
    main()
