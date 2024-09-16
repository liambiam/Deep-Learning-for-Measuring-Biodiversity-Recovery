import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Set paths
data_dir = 'E:/Spectrograms64second'
out_dir = 'E:/Code/ResultsCNN64second_features'
cnn_model = os.path.join(out_dir, 'cnn_model_no_classes.h5')
out_csv = os.path.join(out_dir, 'features_by_month_no_classes.csv')

# Load CNN model
print("Loading model...")
model = tf.keras.models.load_model(cnn_model)
print("Model loaded.")

def get_features(file_path, model):
    """Extract features from file using loaded model."""
    spec = np.load(file_path)
    spec = spec[np.newaxis, ..., np.newaxis]  # Add extra dimensions for model input
    feats = model.predict(spec)
    return feats.flatten()

def extract_all_features(data_dir, model, max_files_per_sub=10):
    """Go through directories, extract features, and store by month."""
    print("Starting feature extraction...")
    feature_data = {}
    
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            for subfolder in os.listdir(class_path):
                subfolder_path = os.path.join(class_path, subfolder)
                if os.path.isdir(subfolder_path):
                    feats = []
                    files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                    
                    # Randomly select files from subfolder
                    selected_files = np.random.choice(files, min(max_files_per_sub, len(files)), replace=False)
                    
                    for f in selected_files:
                        file_path = os.path.join(subfolder_path, f)
                        feature_vec = get_features(file_path, model)
                        feats.append(feature_vec)
                    
                    feature_data[subfolder] = np.mean(feats, axis=0)  # Average features for subfolder
                    print(f"Processed {len(selected_files)} files in {subfolder}.")
    
    print("Feature extraction complete.")
    return feature_data

def save_features_to_csv(feature_dict, out_file):
    """Save features to CSV."""
    print(f"Saving features to {out_file}...")
    df = pd.DataFrame(feature_dict).T
    df.to_csv(out_file, index=True)
    print("Features saved to CSV.")

# Extract features and save to CSV
features = extract_all_features(data_dir, model, max_files_per_sub=10)
save_features_to_csv(features, out_csv)
