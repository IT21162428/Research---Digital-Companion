import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path and categories
dataset_path = "Dataset"  # Adjust the relative path if needed
categories = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Suprise"]

# Function to load and preprocess data
def load_data(subset):
    data = []
    labels = []
    subset_path = os.path.join(dataset_path, subset)
    print(f"Resolved path for {subset}: {os.path.abspath(subset_path)}")
    print(f"Contents of {subset} folder:", os.listdir(subset_path))
    for label, category in enumerate(categories):
        category_path = os.path.join(subset_path, category)
        print(f"Looking for path: {category_path}")
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
X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]

# One-hot encode labels
y_train = to_categorical(y_train, len(categories))
y_test = to_categorical(y_test, len(categories))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

print("Data loaded and preprocessed.")
