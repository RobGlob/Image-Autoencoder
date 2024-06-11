import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            with Image.open(img_path) as img:
                img_array = np.array(img)
                if img_array.max() > 255 or img_array.min() < 0:
                    raise ValueError(f"Image {filename} has pixels out of bounds.")
                images.append(img_array)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return np.array(images)


def create_dataset(folder_path, test_size=0.2, random_state=42):
    images = load_images_from_folder(folder_path)
    X_train, X_test = train_test_split(images, test_size=test_size, random_state=random_state)
    return X_train, X_test


# Path to the folder containing images
folder_path = 'convert_faces'

# Create dataset
X_train, X_test = create_dataset(folder_path)

# print(X_train.shape, X_test.shape) # (112, 128, 128) (28, 128, 128)
# print(X_train[0])

X_train = X_train / 255  # Normalize training data
X_test = X_test / 255
# print(X_train[0])
