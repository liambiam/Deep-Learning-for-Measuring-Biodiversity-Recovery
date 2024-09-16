import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Set paths
data_dir = 'E:/Spectrograms'
train_out_dir = 'E:/Spectrograms_Train'
predict_out_dir = 'E:/Spectrograms_Predict'

# Proportion of files for training
train_ratio = 0.8  # 80% for training, 20% for prediction

# Create folder structure if it doesn't exist
def make_dirs(new_base_dir, original_base_dir):
    for root, dirs, _ in os.walk(original_base_dir):
        for folder in dirs:
            new_folder_path = os.path.join(root.replace(original_base_dir, new_base_dir), folder)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

# Create folders for train and predict directories
make_dirs(train_out_dir, data_dir)
make_dirs(predict_out_dir, data_dir)

# Split and copy files to respective directories
def move_files(source_dir, train_out_dir, predict_out_dir, train_ratio):
    for root, _, files in os.walk(source_dir):
        if files:
            npy_files = [os.path.join(root, f) for f in files if f.endswith('.npy')]
            train_files, predict_files = train_test_split(npy_files, train_size=train_ratio, random_state=42)
            
            for file in train_files:
                train_dest = file.replace(source_dir, train_out_dir)
                os.makedirs(os.path.dirname(train_dest), exist_ok=True)
                shutil.copy(file, train_dest)
                print(f"Copied {file} to {train_dest}")  # Debug info
                
            for file in predict_files:
                predict_dest = file.replace(source_dir, predict_out_dir)
                os.makedirs(os.path.dirname(predict_dest), exist_ok=True)
                shutil.copy(file, predict_dest)
                print(f"Copied {file} to {predict_dest}")  # Debug info

# Perform file split and copying
move_files(data_dir, train_out_dir, predict_out_dir, train_ratio)

print("File splitting and copying complete!")
