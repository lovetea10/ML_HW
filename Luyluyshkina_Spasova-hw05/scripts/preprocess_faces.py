import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import torch


class FacePreprocessor:
    def init(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            keep_all=True,
            post_process=False,
            device=self.device
        )

    def detect_faces(self, image_path):
        """Обнаружение и выравнивание лиц на изображении"""
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        faces = self.mtcnn(img)
        return faces

    def preprocess_image(self, image_path, output_size=160):
        """Полная предобработка одного изображения"""
        try:
            faces = self.detect_faces(image_path)
            if faces is None or len(faces) == 0:
                print(f"No faces detected in {image_path}")
                return None

            face_tensor = faces[0]
            face_tensor = (face_tensor - 127.5) / 128.0  # Нормализация
            return face_tensor
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def preprocess_directory(self, input_dir, output_dir):
        """Пакетная обработка всех изображений в директории"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Создана директория: {output_dir}")

            valid_extensions = ('.jpg', '.jpeg', '.png')
            image_files = [f for f in os.listdir(input_dir)
                           if f.lower().endswith(valid_extensions)]

            if not image_files:
                print(f"В директории {input_dir} нет изображений!")
                return

            print(f"Найдено {len(image_files)} изображений для обработки")

            for img_name in tqdm(image_files, desc="Обработка изображений"):
                img_path = os.path.join(input_dir, img_name)
                output_path = os.path.join(output_dir, f"processed_{img_name.split('.')[0]}.npy")

                processed = self.preprocess_image(img_path)
                if processed is not None:
                    np.save(output_path, processed.cpu().numpy())

            print(f"Успешно обработано и сохранено в {output_dir}")
        except Exception as e:
            print(f"Ошибка при обработке директории: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Directory with raw images')
    parser.add_argument('--output_dir', help='Directory to save processed faces')
    parser.add_argument('--input_image', help='Single image to process')
    parser.add_argument('--output_file', help='Output file for single image')
    args = parser.parse_args()

    preprocessor = FacePreprocessor()

    if args.input_image and args.output_file:
        processed = preprocessor.preprocess_image(args.input_image)
        if processed is not None:
            np.save(args.output_file, processed.cpu().numpy())
            print(f"Saved processed face to {args.output_file}")
    elif args.input_dir and args.output_dir:
        preprocessor.preprocess_directory(args.input_dir, args.output_dir)
        print(f"Processed all images from {args.input_dir} to {args.output_dir}")
    else:
        print("Please specify either --input_image/--output_file or --input_dir/--output_dir")


if __name__ == "main":
    main()