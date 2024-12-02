import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Correct dataset path based on directory structure
dataset_path = "Dataset"  # Correct relative path
categories = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Suprise"]  # Match folder names exactly

# Load and preprocess data
def load_data(subset):
    data = []
    labels = []
    subset_path = os.path.join(dataset_path, subset)
    print(subset_path)
    print("Absolute path being accessed:", os.path.abspath(subset_path))  # Debugging step
    print(f"Contents of {subset} folder:", os.listdir(subset_path))  # Debugging step
    for label, category in enumerate(categories):
        category_path = os.path.join(subset_path, category)
        print(f"Looking for path: {category_path}")  # Debugging step
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Path not found: {category_path}")
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))  # Resize to 48x48 pixels
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load training and testing data
X_train, y_train = load_data("Training")
X_test, y_test = load_data("Testing")

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# One-hot encode labels
y_train = to_categorical(y_train, len(categories))
y_test = to_categorical(y_test, len(categories))

print("Data loaded and preprocessed.")
