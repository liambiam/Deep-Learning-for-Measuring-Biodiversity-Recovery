import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Main directory for spectrogram data
data_dir = 'E:/Spectrograms'

# Class folders with data
folders = ['PAM_1', 'PAM_2', 'PAM_3', 'PAM_4']

# Directory where results will go
out_dir = 'E:/Results/CNN2secondFINAL3'
os.makedirs(out_dir, exist_ok=True)

def grab_data(data_dir, folders, files_per_folder=1500):
    """Grab spectrogram data from folder structure."""
    features = []
    labels = []
    paths = []
    loaded_count = 0

    for folder in folders:
        print(f"Checking folder: {folder}")
        folder_path = os.path.join(data_dir, folder)
        
        month = '2020-03'
        month_path = os.path.join(folder_path, month)
        
        # Get subfolders (sorted for consistency)
        subs = sorted([f for f in os.listdir(month_path) if os.path.isdir(os.path.join(month_path, f))])
        
        if subs:
            first_sub = subs[0]
            print(f"    Looking in subfolder: {first_sub}")
            sub_path = os.path.join(month_path, first_sub)
            files = [f for f in os.listdir(sub_path) if f.endswith('.npy')]
            
            if files:
                selected_files = np.random.choice(files, min(files_per_folder, len(files)), replace=False)
                
                for file in selected_files:
                    file_path = os.path.join(sub_path, file)
                    spectrogram = np.load(file_path)
                    features.append(spectrogram)
                    labels.append(folder)
                    paths.append(file_path)
                    loaded_count += 1
                    
                    if loaded_count % 10 == 0:
                        print(f"      Loaded {loaded_count} files so far...")
                        
    print(f"Total loaded files: {loaded_count}")
    return np.array(features), np.array(labels), np.array(paths)

# Load up data
X, y, file_paths = grab_data(data_dir, folders)

# Encode labels into numeric format
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
    X, y_encoded, file_paths, test_size=0.2, random_state=42, stratify=y_encoded
)

# Reshape data to fit CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Class count: {len(encoder.classes_)}")

def make_cnn_model(input_shape, num_classes):
    """Create basic CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Adding dropout to avoid overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Make model
input_shape = X_train.shape[1:]
num_classes = len(encoder.classes_)
model = make_cnn_model(input_shape, num_classes)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',       # Watching validation loss
    patience=3,               # Stop after 3 bad epochs
    restore_best_weights=True # Roll back to best weights
)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[stopper])

# Save trained model
model.save(os.path.join(out_dir, 'cnn_model.h5'))

# Evaluate model performance
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy}")

# Get predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=encoder.classes_)

# Save confusion matrix and classification report
np.savetxt(os.path.join(out_dir, 'confusion_matrix.txt'), conf_matrix, fmt='%d')
with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
    f.write(class_report)

# Plot ROC curve for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"ROC curve for {encoder.classes_[i]} (area = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Classes')
plt.legend(loc="lower right")
plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
plt.close()

# Plot training and validation metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

plt.savefig(os.path.join(out_dir, 'training_validation_metrics.png'))
plt.close()

# Save final metrics
with open(os.path.join(out_dir, 'final_metrics.txt'), 'w') as f:
    f.write(f"Final Test Accuracy: {test_accuracy}\n")
    f.write(f"Final Test Loss: {test_loss}\n")
    f.write("\nClassification Report:\n")
    f.write(class_report)

# Save paths for test files
with open(os.path.join(out_dir, 'test_file_paths.txt'), 'w') as f:
    for path in test_paths:
        f.write(f"{path}\n")

# Save paths for train files
with open(os.path.join(out_dir, 'train_file_paths.txt'), 'w') as f:
    for path in train_paths:
        f.write(f"{path}\n")

# Print class names
print("Classes:", encoder.classes_)
