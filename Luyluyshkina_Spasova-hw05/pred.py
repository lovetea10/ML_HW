import os
import torch
from face_cli import FaceRecognizer
from face_recognition.config import CELEBA_CONFIG

model_path = os.path.join(CELEBA_CONFIG['models_dir'], 'facenet_model.pth')
target_path = os.path.join(CELEBA_CONFIG['output_dir'], 'raw', 'target.jpg')
test_dir = os.path.join(CELEBA_CONFIG['output_dir'], 'raw', 'test_images')

recognizer = FaceRecognizer(model_path=model_path, threshold=0.7)

results = recognizer.compare_with_target(target_path, test_dir)

results = results[:10]

output_file = os.path.join(CELEBA_CONFIG['output_dir'], 'predictions.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Результаты предсказания на тестовой выборке:\n")
    f.write("-" * 60 + "\n")
    for res in results:
        status = "СОВПАДАЕТ" if res['match'] else "НЕ СОВПАДАЕТ"
        f.write(f"{res['test_image']}: {status} (расстояние: {res['distance']:.4f})\n")

print(f"Результаты сохранены в {output_file}")