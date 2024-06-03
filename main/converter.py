import cv2
import os

"""Данный код нужен для преобразования цветной фотки в черно-белое (в 1 цветовой канал)
Если нужны подробности свяжитесь с diasizbasarov123@gmail.com"""


def convert_to_grayscale(input_path, output_folder):
    # провекри и обработки исключении
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)

        if not os.path.isfile(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image = cv2.imread(file_path)

        if image is None:
            print(f"Ошибка: не удалось открыть изображение {file_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # преобразование в черно-белое

        # Полный путь к выходному файлу
        output_path = os.path.join(output_folder, filename)

        cv2.imwrite(output_path, gray_image)

        print(f"Изображение {file_path} успешно преобразовано и сохранено в {output_path}")


input_path = 'faces'
output_folder = 'convert_faces'

convert_to_grayscale(input_path, output_folder)
