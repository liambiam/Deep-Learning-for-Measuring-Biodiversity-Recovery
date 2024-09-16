import os
import numpy as np
import pandas as pd
import random
from scipy.stats import entropy

# Base directories for PAM spectrograms
paths = {
    'PAM_1': 'E:/Spectrograms/PAM_1',
    'PAM_2': 'E:/Spectrograms/PAM_2',
    'PAM_3': 'E:/Spectrograms/PAM_3',
    'PAM_4': 'E:/Spectrograms/PAM_4'
}

# Results directory
res_dir = 'E:/Results/acoustic_indices'
os.makedirs(res_dir, exist_ok=True)

def get_random_files(base_path):
    all_files = []
    # Go through each year/month folder
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subfolders:
                # Take the first subfolder
                first_subfolder = subfolders[0]
                subfolder_path = os.path.join(folder_path, first_subfolder)
                # Find .npy files in the subfolder
                files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                if files:
                    # Randomly pick up to 500 files
                    chosen_files = random.sample(files, min(500, len(files)))
                    all_files.extend(chosen_files)
                    print(f"Picked {len(chosen_files)} files from {first_subfolder}")
    return all_files

def get_date_from_folder(folder):
    pieces = folder.split('_')
    for piece in pieces:
        if len(piece) == 8 and piece.isdigit():
            return f'{piece[:4]}-{piece[4:6]}-{piece[6:]}'
    return 'unknown'

def aci_calc(spec):
    # Calculate Acoustic Complexity Index
    return np.sum(np.abs(np.diff(spec, axis=1)))

def adi_calc(spec):
    # Calculate Acoustic Diversity Index
    freq_bins = np.mean(spec, axis=1)
    norm_bins = freq_bins / np.sum(freq_bins)
    return entropy(norm_bins)

def bio_index_calc(spec):
    # Calculate Bioacoustic Index
    return np.sum(spec)

def ndsi_calc(spec, bio_range=(2000, 8000), anthro_range=(1000, 2000)):
    # Calculate Normalized Difference Soundscape Index
    bin_count = spec.shape[0]
    bin_width = 22050 / bin_count
    bio_idx = (int(bio_range[0] / bin_width), int(bio_range[1] / bin_width))
    anthro_idx = (int(anthro_range[0] / bin_width), int(anthro_range[1] / bin_width))
    
    E_bio = np.sum(spec[bio_idx[0]:bio_idx[1], :])
    E_anthro = np.sum(spec[anthro_idx[0]:anthro_idx[1], :])
    
    return (E_bio - E_anthro) / (E_bio + E_anthro) if (E_bio + E_anthro) != 0 else 0

def process_directory(name, path):
    print(f"Processing {name}...")
    files = get_random_files(path)
    print(f"Collected {len(files)} file paths from {name}")
    
    result_list = []
    for idx, file in enumerate(files):
        spec = np.load(file)
        
        aci = aci_calc(spec)
        adi = adi_calc(spec)
        ndsi = ndsi_calc(spec)
        bio_index = bio_index_calc(spec)
        
        combined = np.mean([aci, adi, ndsi, bio_index])
        
        folder_name = os.path.basename(os.path.dirname(file))
        date = get_date_from_folder(folder_name)
        
        result_list.append([date, os.path.basename(file), aci, adi, ndsi, bio_index, combined])
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(files):
            print(f"Processed {idx + 1}/{len(files)} files from {name}")
    
    df_cols = ['Date', 'FileName', 'ACI', 'ADI', 'NDSI', 'BioIndex', 'Combined']
    df = pd.DataFrame(result_list, columns=df_cols)
    
    save_file = os.path.join(res_dir, f'indices_{name}.csv')
    print(f"Saving results for {name} to {save_file}")
    df.to_csv(save_file, index=False)

    print(f"Finished processing {name}.\n")

# Process each PAM directory
for name, path in paths.items():
    process_directory(name, path)

print("All processing complete.")
