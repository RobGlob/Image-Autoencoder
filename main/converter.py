import cv2
import os
import imghdr


def convert_to_grayscale(input_path, output_folder):
    """
    Convert color images to grayscale and save them in the specified output folder.
    If you have any questions please contact diasizbasarov123@gmail.com

    Args:
        input_path (str): Path to the directory containing input images.
        output_folder (str): Path to the directory where grayscale images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(input_path):
        print(f"Input directory {input_path} does not exist.")
        return

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)

        if not os.path.isfile(file_path):
            continue

        # Check if the file is an image
        if imghdr.what(file_path) is None:
            print(f"{file_path} is not a valid image file.")
            continue

        image = cv2.imread(file_path)

        if image is None:
            print(f"Failed to open the image {file_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Resize the image to 128x128 pixels
        resized_image = cv2.resize(gray_image, (128, 128))

        # Full path to the output file
        output_path = os.path.join(output_folder, filename)

        cv2.imwrite(output_path, resized_image)

        print(f"Image {file_path} successfully transformed, resized to 128x128, and saved in {output_path}")


input_path = 'faces'
output_folder = 'convert_faces'

convert_to_grayscale(input_path, output_folder)
