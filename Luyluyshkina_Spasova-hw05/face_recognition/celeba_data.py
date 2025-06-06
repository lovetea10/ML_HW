import os
import shutil
from tqdm import tqdm

def distribute_celeba_images():
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] Корневая директория проекта: {project_root}")

    data_dir = os.path.join(project_root, 'data')
    input_dir = os.path.join(data_dir, 'input')
    img_dir = os.path.join(input_dir, 'img_align_celeba')
    partition_file = os.path.join(input_dir, 'list_eval_partition.txt')

    print(f"[INFO] Путь к изображениям: {img_dir}")
    print(f"[INFO] Путь к файлу разметки: {partition_file}")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"[ERROR] Папка с изображениями не найдена: {img_dir}")
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"[ERROR] Файл разметки не найден: {partition_file}")

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    print(f"[INFO] Найдено {len(image_files)} JPEG-изображений в папке img_align_celeba")
    if len(image_files) == 0:
        raise FileNotFoundError(f"[ERROR] В папке img_align_celeba нет JPEG-изображений!")

    output_dirs = {
        '0': 'train',
        '1': 'validation',
        '2': 'test'
    }

    for folder in output_dirs.values():
        os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

    with open(partition_file, 'r') as f:
        lines = f.readlines()

    processed = 0
    missing = 0
    errors = 0

    for line in tqdm(lines, desc="Распределение изображений"):
        try:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            filename = parts[0]
            split_id = parts[1]

            if not filename.lower().endswith('.jpg'):
                filename += '.jpg'

            src_path = os.path.join(img_dir, filename)

            if not os.path.exists(src_path):
                missing += 1
                continue

            if split_id not in output_dirs:
                continue

            dest_dir = os.path.join(data_dir, output_dirs[split_id])
            dest_path = os.path.join(dest_dir, filename)

            shutil.copy2(src_path, dest_path)
            processed += 1

        except Exception as e:
            errors += 1
            print(f"[WARNING] Ошибка при обработке строки: {line.strip()}. Ошибка: {str(e)}")

    print("\n[РЕЗУЛЬТАТ]")
    print(f"Успешно обработано: {processed} изображений")
    print(f"Пропущено (не найдено): {missing} изображений")
    print(f"Ошибок обработки: {errors}")

    if missing > 0:
        print(f"\n[WARNING] Пропущено {missing} файлов. Возможные причины:")
        print("- Имена файлов в list_eval_partition.txt не соответствуют именам в папке")
        print("- Файлы имеют другие расширения (не .jpg)")
        print("- Архив с изображениями не полностью распакован")

if __name__ == "__main__":
    distribute_celeba_images()