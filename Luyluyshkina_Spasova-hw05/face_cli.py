import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import argparse


class FaceRecognizer:
    def __init__(self, model_path='models/facenet_model.pt', threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.threshold = threshold
        self.mtcnn = MTCNN(device=self.device)

    def get_embedding(self, image_path):
        """Получение эмбеддинга лица"""
        try:
            img = Image.open(image_path).convert('RGB')
            face = self.mtcnn(img)
            if face is None:
                print(f"Лицо не обнаружено: {image_path}")
                return None
            with torch.no_grad():
                embedding = self.model(face.unsqueeze(0)).cpu().numpy()
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Ошибка обработки {image_path}: {str(e)}")
            return None

    def compare_with_target(self, target_path, test_dir):
        """Сравнение целевого изображения со всеми изображениями в директории"""
        target_emb = self.get_embedding(target_path)
        if target_emb is None:
            print(f"Не удалось обработать целевое изображение: {target_path}")
            return []

        results = []
        test_images = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f != os.path.basename(target_path)
        ])

        for img_name in test_images:
            test_path = os.path.join(test_dir, img_name)
            test_emb = self.get_embedding(test_path)
            if test_emb is None:
                continue

            distance = np.linalg.norm(target_emb - test_emb)
            is_match = distance < self.threshold
            results.append({
                'test_image': img_name,
                'distance': float(distance),
                'match': is_match
            })

        return sorted(results, key=lambda x: x['distance'])


def main():
    parser = argparse.ArgumentParser(description='Сравнение целевого изображения с тестовыми')
    parser.add_argument('--target', default='data/raw/target.jpg',
                        help='Путь к целевому изображению (по умолчанию: data/raw/target.jpg)')
    parser.add_argument('--test_dir', default='data/raw/test_images',
                        help='Директория с тестовыми изображениями (по умолчанию: data/raw/test_images)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Порог схожести (по умолчанию: 0.9)')

    args = parser.parse_args()

    recognizer = FaceRecognizer(threshold=args.threshold)
    results = recognizer.compare_with_target(args.target, args.test_dir)

    print(f"\nРезультаты сравнения с {os.path.basename(args.target)} (порог: {args.threshold}):")
    print("-" * 60)
    for res in results:
        status = "СОВПАДАЕТ" if res['match'] else "НЕ СОВПАДАЕТ"
        print(f"{res['test_image']}: {status} (расстояние: {res['distance']:.4f})")


if __name__ == "__main__":
    main()