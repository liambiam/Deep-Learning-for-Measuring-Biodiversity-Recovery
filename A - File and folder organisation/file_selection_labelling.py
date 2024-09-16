import os
import random

# Directory where spectrograms are located
data_dir = 'E:/Spectrograms'

# Results folder
out_dir = 'E:/Code/Results'
os.makedirs(out_dir, exist_ok=True)

# Class labels
labels = ['PAM_1', 'PAM_2', 'PAM_3', 'PAM_4']

# Collect spectrogram paths from subfolders, limit files per folder
def gather_spectrograms(data_dir, labels, max_files=100):
    folder_files = {}
    
    for label in labels:
        class_path = os.path.join(data_dir, label)
        for root, dirs, files in os.walk(class_path):
            # Only process folders with files (ultimate subfolders)
            if not dirs and files:
                if not os.path.basename(root) in folder_files:
                    folder_files[os.path.basename(root)] = []
                
                # Add all .npy files to dictionary
                for file in files:
                    if file.endswith('.npy'):
                        folder_files[os.path.basename(root)].append(os.path.join(root, file))
    
    # Sample files from each subfolder if there are too many
    selected_files = []
    for subfolder, files in folder_files.items():
        if len(files) > max_files:
            selected_files.extend(random.sample(files, max_files))
        else:
            selected_files.extend(files)
    
    return selected_files

# Get spectrogram paths
sampled_files = gather_spectrograms(data_dir, labels)

# Verify and save paths with labels
output_file = os.path.join(out_dir, 'verified_spectrogram_paths.txt')
with open(output_file, 'w') as f:
    for file_path in sampled_files:
        # Extract class label from directory structure
        parts = file_path.split(os.sep)
        # Class label assumed to be 4th directory from end (adjust as needed)
        label = parts[-4]
        
        if label in labels:
            f.write(f'{file_path} - Label: {label}\n')
        else:
            f.write(f'{file_path} - **Incorrect Label: {label}**\n')
            print(f'Warning: Incorrect label for {file_path}. Found: {label}, Expected one of: {labels}')

print(f'Paths saved to {output_file}')
