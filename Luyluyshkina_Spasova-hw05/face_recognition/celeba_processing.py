import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import albumentations as A
from config import CELEBA_CONFIG

class CelebAPreprocessor:
    def __init__(self, config=None):
        self.config = config if config else CELEBA_CONFIG
        self.detector = MTCNN()
        self.augmentor = self._build_augmentor()

        for split in ['train', 'test', 'validation']:
            os.makedirs(os.path.join(self.config['output_dir'], split), exist_ok=True)

    def _build_augmentor(self):
        """Создает пайплайн аугментаций"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.CLAHE(p=0.1),
        ])

    def _detect_face(self, image):
        """Обнаруживает лицо на изображении"""
        results = self.detector.detect_faces(image)
        if not results:
            return None
        
        best_result = max(results, key=lambda x: x['confidence'])
        x1, y1, width, height = best_result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        margin_w = int(width * 0.2)
        margin_h = int(height * 0.2)
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(image.shape[1], x2 + margin_w)
        y2 = min(image.shape[0], y2 + margin_h)
        
        return image[y1:y2, x1:x2]

    def _preprocess_image(self, face_img):
        """Основная предобработка изображения"""
        face_img = cv2.resize(face_img, self.config['target_size'])
        face_img = face_img.astype('float32')
        face_img = (face_img - self.config['mean']) / self.config['std']
        return face_img

    def _save_processed(self, image, filename, split_type):
        """Сохраняет обработанное изображение с указанием типа данных"""
        output_path = os.path.join(self.config['output_dir'], split_type, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if image.dtype != np.uint8:
            image = ((image * self.config['std'] + self.config['mean']).clip(0, 255).astype(np.uint8))
        
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def process_image(self, img_path, split_type):
        """Обрабатывает одно изображение с указанием типа данных"""
        try:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if image is None:
                print(f"Не удалось загрузить изображение: {img_path}")
                return None

            face = self._detect_face(image)
            if face is None:
                print(f"Лицо не обнаружено: {img_path}")
                return None

            processed = self._preprocess_image(face)

            augmented = []
            for _ in range(self.config['augment_count']):
                try:
                    aug_result = self.augmentor(image=face)
                    aug_face = aug_result['image']
                    aug_processed = self._preprocess_image(aug_face)
                    augmented.append(aug_processed)
                except Exception as aug_error:
                    print(f"Ошибка аугментации {img_path}: {str(aug_error)}")
                    continue
            
            return (processed, *augmented), split_type  # Возвращаем кортеж
            
        except Exception as e:
            print(f"Критическая ошибка обработки {img_path}: {str(e)}")
            return None

    def process_dataset(self):
        """Обрабатывает все изображения во всех папках"""
        for split_type in ['train', 'test', 'validation']:
            input_dir = os.path.join(self.config['input_dir'], split_type)
            if not os.path.exists(input_dir):
                print(f"Папка {input_dir} не найдена, пропускаем")
                continue
                
            image_files = [f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Обработка {split_type} ({len(image_files)} изображений)...")
            
            for filename in tqdm(image_files[:self.config['max_samples']] 
                              if self.config['max_samples'] else image_files):
                img_path = os.path.join(input_dir, filename)
                result = self.process_image(img_path, split_type)  # Не распаковываем сразу
                
                if result is not None:
                    images, split = result
                    for i, img in enumerate(images):
                        out_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
                        self._save_processed(img, out_filename, split)

if __name__ == "__main__":
    preprocessor = CelebAPreprocessor()
    preprocessor.process_dataset()
    print("Предобработка всех данных завершена!")