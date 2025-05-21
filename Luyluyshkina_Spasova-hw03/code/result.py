import evaluate as evaluate
import torch
import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from torchtext.data import BucketIterator

from prep import load_stored_data, BOS_TOKEN, load_word_field, EOS_TOKEN
from model_ml import EncoderDecoder
from tools import DEVICE, convert_batch, make_mask, tokens_to_words, draw, words_to_tokens

def summary_generator(path_to_model, source, word_field, vocab_size):
    ''' Генератор суммаризации для модели (задание №1) '''
    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size)
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    source = source.transpose(0, 1).to(DEVICE)
    target = torch.tensor([word_field.vocab.stoi[BOS_TOKEN]], device=DEVICE).reshape(1, 1)

    source_mask, target_mask = make_mask(source, target, pad_idx=1)
    with torch.no_grad():
        encoder_output = model.encoder(source, source_mask)
        while word_field.vocab.stoi[EOS_TOKEN] not in target[0]:
            source_mask, target_mask = make_mask(source, target, pad_idx=1)
            decoder_output = model.decoder(target, encoder_output, source_mask, target_mask)
            out = decoder_output.contiguous().view(-1, decoder_output.shape[-1])
            next_token = torch.argmax(out, dim=1).reshape(1, -1)
            target = torch.cat((target, next_token), dim=-1)
            if target.shape[1] >= 50:  # Ограничение длины
                break

    return target[0]

def educ(path_to_model, word_field, vocab_size, iter):
    ''' Оценка модели на тестовой выборке с помощью ROUGE метрики (задание №2)'''
    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size)
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    predictions = []
    references = []
    with torch.no_grad():
        for elem in iter:
            source_input, target_input_ref, source_mask, _ = convert_batch(elem)
            references.append(' '.join(tokens_to_words(word_field, target_input_ref[0])))

            encoder_output = model.encoder(source_input, source_mask)
            target_input = torch.tensor([word_field.vocab.stoi[BOS_TOKEN]], device=DEVICE).reshape(1, 1)
            max_steps = max(10, target_input_ref.shape[1] // 2)
            for _ in range(max_steps):
                source_mask, target_mask = make_mask(source_input, target_input, pad_idx=1)
                decoder_output = model.decoder(target_input, encoder_output, source_mask, target_mask)
                out = decoder_output.contiguous().view(-1, decoder_output.shape[-1])
                next_token = torch.argmax(out, dim=1)[-1].unsqueeze(0)  # Берем последний токен
                target_input = torch.cat((target_input, next_token.unsqueeze(0)), dim=-1)
                if next_token.item() == word_field.vocab.stoi[EOS_TOKEN]:
                    break
            predictions.append(' '.join(tokens_to_words(word_field, target_input[0])))

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE метрики:", results)

    with open("ROUGE_metrics_emb.json", "w") as f:
        json.dump({"rouges": results}, f, indent=4)
    print("ROUGE метрики сохранены в ROUGE_metrics_emb.json")

def example_summary():
    ''' Сохранение результатов генератора суммаризации для 5 примеров из тестовой выборки
     и 5 собственных примеров(задание №1)'''
    # Тестовые примеры
    data = []
    for i, elem in enumerate(test_iter):
        if i >= 5:
            break
        text = ' '.join(tokens_to_words(word_field, elem.source))
        res = summary_generator("best_model.pt", elem.source, word_field, vocab_size)
        data.append((text, ' '.join(tokens_to_words(word_field, res))))

    with open("examples_from_tests.txt", "w", encoding='utf-8') as f:
        for i, example in enumerate(data, 1):
            f.write(f"Тестовый пример {i}:\n")
            f.write(f"Исходный текст: {example[0]}\n")
            f.write(f"Сгенерированная суммаризация: {example[1]}\n\n")
    print("Тестовые суммаризации сохранены в examples_from_tests.txt")

    # Пользовательские примеры
    try:
        with open("news_examples.txt", 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()][:5]
    except FileNotFoundError:
        print("Файл news_examples.txt не найден")
        return

    data = []
    for news in lines:
        source_news = word_field.preprocess(news)
        source_tokens = words_to_tokens(word_field, source_news).to(DEVICE)
        res = summary_generator("best_model.pt", source_tokens, word_field, vocab_size)
        data.append((news, ' '.join(tokens_to_words(word_field, res))))

    with open("our_examples.txt", "w", encoding='utf-8') as f:
        for i, example in enumerate(data, 1):
            f.write(f"Пользовательский пример {i}:\n")
            f.write(f"Исходный текст: {example[0]}\n")
            f.write(f"Сгенерированная суммаризация: {example[1]}\n\n")
    print("Пользовательские суммаризации сохранены в our_examples.txt")

def visualize_attention(model, word_field, elem, num):
    ''' Визуализация механизма внимания (задание №3) '''
    source_input, _, source_mask, _ = convert_batch(elem)
    words = tokens_to_words(word_field, elem.source)
    with torch.no_grad():
        encoder_output = model.encoder(source_input, source_mask)
        for layer in range(4):
            attn_probs = model.encoder._blocks[layer]._self_attn._attn_probs
            for h in range(4):
                plt.figure(figsize=(10, 8))
                plt.title(f"Энкодер, слой {layer + 1}, голова {h + 1}, пример {num}")
                draw(attn_probs[0, h].data.cpu(), words, words)
                plt.tick_params(labelsize=6)
                output_dir = Path(f"visualize_attention/example_{num}/encoder_layer_{layer+1}")
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f"attn_head_{h+1}.jpg")
                plt.close()

if __name__ == "__main__":
    # Загрузка тестового датасета и словаря
    test_dataset = load_stored_data("../data/test")
    test_iter = BucketIterator(test_dataset, batch_size=1, device=DEVICE, shuffle=False)
    word_field = load_word_field("../data")

    # Проверка специальных токенов
    print("Проверка специальных токенов:")
    print(f"<unk>: {word_field.vocab.stoi['<unk>']} (ожидалось 0)")
    print(f"<pad>: {word_field.vocab.stoi['<pad>']} (ожидалось 1)")
    print(f"<bos>: {word_field.vocab.stoi[BOS_TOKEN]}")
    print(f"<eos>: {word_field.vocab.stoi[EOS_TOKEN]}")

    # Загрузка размера словаря из metadatasets_sizes.json
    try:
        with open("metadatasets_sizes.json") as f_in:
            datasets_sizes = json.load(f_in)
        vocab_size = datasets_sizes["Vocab size"]
    except FileNotFoundError:
        print("Файл metadatasets_sizes.json не найден")
        raise
    except KeyError:
        print("Ключ 'Vocab size ' отсутствует в metadatasets_sizes.json")
        raise
    print("Vocab size =", vocab_size)

    # Загрузка модели для визуализации
    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size)
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    # Задание 1: Генерация суммаризаций
    print(r'Генерация суммаризаций...')
    example_summary()

    # Задание 2: Оценка ROUGE
    print(r'Оценка ROUGE метрик...')
    educ("best_model.pt", word_field, vocab_size, test_iter)

    # Задание 3: Визуализация внимания
    print(r'Визуализация весов внимания...')
    for i, elem in enumerate(test_iter):
        if i >= 3:
            break
        visualize_attention(model, word_field, elem, i + 1)
    print("Визуализации сохранены в visualize_attention/")