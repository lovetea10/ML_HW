import os
from prep import prepare_datasets
from tools import DEVICE

# Основные параметры
epochs = 15
smoothing_coefficient = 0.1
import os
import dill
import pandas as pd
from tqdm import tqdm
import json
from torchtext.data import Field, Example, Dataset

# Создаем все необходимые директории
os.makedirs("../data/train", exist_ok=True)
os.makedirs("../data/val", exist_ok=True)
os.makedirs("../data/test", exist_ok=True)

# Инициализация полей
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
fields = [('source', word_field), ('target', word_field)]

# Загрузка данных (убедитесь, что news.csv существует)
try:
    data = pd.read_csv('news.csv', delimiter=',')
except FileNotFoundError:
    raise Exception("Файл news.csv не найден в текущей директории")

# Создание примеров
examples = []
for _, row in tqdm(data.iterrows(), total=len(data)):
    try:
        source_text = word_field.preprocess(row['text'])
        target_text = word_field.preprocess(row['title'])
        examples.append(Example.fromlist([source_text, target_text], fields))
    except Exception as e:
        print(f"Ошибка в строке {_}: {e}")

# Создание и разделение датасета
dataset = Dataset(examples, fields)
train_val, test_dataset = dataset.split(split_ratio=0.85)  # 85% train+val, 15% test
train_dataset, val_dataset = train_val.split(split_ratio=0.89)  # 89% от 85% = ~75% train

# Построение словаря
word_field.build_vocab(train_dataset, min_freq=7)

# Функция для сохранения
def save_dataset(dataset, path):
    """Сохраняет датасет в указанную директорию"""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "dataset.Field"), "wb") as f:
        dill.dump(dataset.fields, f)
    with open(os.path.join(path, "dataset.Example"), "wb") as f:
        dill.dump(dataset.examples, f)

# Сохраняем все датасеты
save_dataset(train_dataset, "../data/train")
save_dataset(val_dataset, "../data/val")
save_dataset(test_dataset, "../data/test")

# Сохраняем word_field
with open("../data/word_field.Field", "wb") as f:
    dill.dump(word_field, f)

# Сохраняем метаданные
metadata = {
    "Train size": len(train_dataset),
    "Validation size": len(val_dataset),
    "Test size": len(test_dataset),
    "Vocab size": len(word_field.vocab)
}

with open("../data/metadatasets_sizes.json", "w") as f:
    json.dump(metadata, f)

print("Все файлы успешно сгенерированы!")
print(f"Обучающая выборка: {len(train_dataset)} примеров")
print(f"Валидационная выборка: {len(val_dataset)} примеров")
print(f"Тестовая выборка: {len(test_dataset)} примеров")
print(f"Размер словаря: {len(word_field.vocab)}")