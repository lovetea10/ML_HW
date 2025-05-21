import json
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchtext.data import Field, Example, Dataset, BucketIterator
from navec import Navec

from tools import DEVICE, BOS_TOKEN, EOS_TOKEN

path = '../navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)


def store_data(dataset, storage_path):

    with open(storage_path + "/dataset.Field", "wb") as f:
        pickle.dump(dataset.fields, f)

    with open(storage_path + "/dataset.Example", "wb") as f:
        pickle.dump(dataset.examples, f)


def load_stored_data(storage_path):

    with open(storage_path + "/dataset.Field", "rb") as f:
        fields = pickle.load(f)

    with open(storage_path + "/dataset.Example", "rb") as f:
        examples = pickle.load(f)

    return Dataset(examples, fields)


def save_word_field(field, path):
    with open(path + "/word_field.Field", "wb") as f:
        pickle.dump(field, f)


def load_word_field(path):
    with open(path + "/word_field.Field", "rb") as f:
        word_field = pickle.load(f)

    return word_field


def prepare_datasets():
    text_field = Field(
        tokenize='moses',
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        lower=True
    )

    data_structure = [('content', text_field), ('headline', text_field)]

    input_data = pd.read_csv(
        r'..\news.csv',
        delimiter=','
    )

    samples = []
    for _, record in tqdm(input_data.iterrows(), total=len(input_data)):
        content = text_field.preprocess(record.text)
        headline = text_field.preprocess(record.title)
        samples.append(Example.fromlist([content, headline], data_structure))

    full_dataset = Dataset(samples, data_structure)

    # Разделение на обучающую, валидационную и тестовую выборки
    train_data, valid_data = full_dataset.split(split_ratio=0.9)
    train_data, test_data = train_data.split(split_ratio=0.89)

    print(f'Обучающая выборка: {len(train_data)} примеров')
    print(f'Валидационная выборка: {len(valid_data)} примеров')
    print(f'Тестовая выборка: {len(test_data)} примеров')

    text_field.build_vocab(train_data, min_freq=7)
    print(f'Размер словаря: {len(text_field.vocab)}')

    save_word_field(text_field, "../data")

    store_data(train_data, "../data/train")
    store_data(valid_data, "../data/val")
    store_data(test_data, "../data/test")

    # Сохранение метаинформации
    metadata = {
        "Train size ": len(train_data),
        "Validation size ": len(valid_data),
        "Test size ": len(test_data),
        "Vocab size ": len(text_field.vocab)
    }

    with open("metadatasets_sizes.json", "w") as f:
        json.dump(metadata, f, indent=4)
        print(f"Файл metadatasets_sizes.json сохранен в {os.path.abspath('metadatasets_sizes.json')}")

    return train_data, valid_data, test_data, text_field