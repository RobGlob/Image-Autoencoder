import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import layers, models, losses, optimizers, metrics
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from keras import regularizers


def load_images_from_folder(folder_path, target_size=(128, 128)):
    """
    Load images from a specified folder, resize them, and convert to numpy arrays.

    Parameters:
    folder_path (str): Path to the folder containing images.
    target_size (tuple): Desired image size.

    Returns:
    np.ndarray: Array of images.
    """
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            with Image.open(img_path) as img:
                img = img.resize(target_size)
                img_array = np.array(img)
                if len(img_array.shape) == 2:  # Grayscale image
                    img_array = np.expand_dims(img_array, axis=-1)
                if img_array.max() > 255 or img_array.min() < 0:
                    raise ValueError(f"Image {filename} has pixels out of bounds.")
                images.append(img_array)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return np.array(images)


def create_dataset(folder_path, test_size=0.2, random_state=42):
    """
    Create a dataset by loading images from a folder and splitting into train/test sets.

    Parameters:
    folder_path (str): Path to the folder containing images.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
    tuple: Arrays containing training and testing images.
    """
    images = load_images_from_folder(folder_path)
    X_train, X_test = train_test_split(images, test_size=test_size, random_state=random_state)
    return X_train, X_test


def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


def visualize(img, encoder, decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()


folder_path = 'convert_faces'

X_train, X_test = create_dataset(folder_path)

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape, X_test.shape)

# Define the autoencoder model
input_shape = X_train.shape[1:]

encoder_input = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(
    encoder_input)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
encoder_output = layers.Dense(128, activation='relu')(x)

encoder = models.Model(encoder_input, encoder_output)

decoder_input = layers.Input(shape=(128,))
x = layers.Dense(16 * 16 * 128, activation='relu')(decoder_input)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2DTranspose(1 if input_shape[-1] == 1 else 3, (3, 3), activation='sigmoid', padding='same')(
    x)

decoder = models.Model(decoder_input, decoder_output)

autoencoder_input = encoder_input
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded)

autoencoder.compile(optimizer='adamw', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Get embeddings for test set
embeddings = encoder.predict(X_test)

for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)

# Example to calculate cosine similarity
index1 = 0
index2 = 1
cosine_sim = cosine_similarity([embeddings[index1]], [embeddings[index2]])
print(f"Cosine similarity between image {index1} and {index2}: {cosine_sim[0][0]}")

