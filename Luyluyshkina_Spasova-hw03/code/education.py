from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
from torchtext.data import BucketIterator
from evaluate import load
from functools import partial
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт пользовательских модулей
from model_ml import EncoderDecoder, NoamOpt
from tools import DEVICE, convert_batch, tokens_to_words
from prep import load_stored_data, load_word_field

epochs = 15
smoothing_coefficient = 0.1

class Trainer:
    def __init__(self):
        self._init_wandb()
        self._load_data()
        self._setup_model()

    def _init_wandb(self):
        try:
            wandb.login(key="6826cbc8aaa00f3a846f7287f32782aff3359c3d")
            self.run = wandb.init(
                project="hw3",
                name="enhanced_training",
                config={
                    "batch_size": 16,
                    "val_batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": epochs,
                    "label_smoothing": smoothing_coefficient
                }
            )
        except Exception as e:
            logger.error(f"Ошибка при инициализации W&B: {e}")
            raise

    def _load_data(self):

        try:
            self.word_field = load_word_field("../data")
            train_data = load_stored_data("../data/train")
            val_data = load_stored_data("../data/val")

            self.train_iter = BucketIterator(
                train_data,
                batch_size=wandb.config.batch_size,
                device=DEVICE,
                shuffle=True
            )
            self.val_iter = BucketIterator(
                val_data,
                batch_size=wandb.config.val_batch_size,
                device=DEVICE,
                shuffle=False
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def _setup_model(self):

        try:
            self.model = EncoderDecoder(
                len(self.word_field.vocab),
                len(self.word_field.vocab)
            ).to(DEVICE)

            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.word_field.vocab.stoi['<pad>'],
                label_smoothing=smoothing_coefficient
            )
            self.optimizer = NoamOpt(self.model)

            # Перенос модели на устройство
            self.model.to(DEVICE)
        except Exception as e:
            logger.error(f"Ошибка при инициализации модели: {e}")
            raise

    def _compute_metrics(self, predictions, references):

        try:
            rouge = load('rouge')
            return rouge.compute(
                predictions=predictions,
                references=references,
                rouge_types=['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        except Exception as e:
            logger.error(f"Ошибка при вычислении метрик: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def _run_epoch(self, data_iter, is_training=True):
        self.model.train(is_training)
        total_loss = 0
        predictions = []
        references = []

        with torch.set_grad_enabled(is_training):
            for batch in tqdm(data_iter, desc="Training" if is_training else "Validation"):
                src, tgt, src_mask, tgt_mask = convert_batch(batch)

                # Forward pass
                outputs = self.model(
                    src,
                    tgt[:, :-1],
                    src_mask,
                    tgt_mask[:, :-1, :-1]
                )
                outputs = outputs.contiguous().view(-1, outputs.size(-1))

                # Подготовка метрик
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(tokens_to_words(self.word_field, preds))
                references.extend(tokens_to_words(self.word_field, tgt[:, 1:].contiguous().view(-1)))

                # Вычисление потерь
                loss = self.criterion(outputs, tgt[:, 1:].contiguous().view(-1))
                total_loss += loss.item()

                # Backward pass
                if is_training:
                    self.optimizer.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        return {
            "loss": total_loss / len(data_iter),
            "metrics": self._compute_metrics(predictions, references)
        }

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Обучение
            train_results = self._run_epoch(self.train_iter, is_training=True)

            # Валидация
            val_results = self._run_epoch(self.val_iter, is_training=False)

            # Логирование
            metrics = {
                "epoch": epoch,
                "train_loss": train_results["loss"],
                "val_loss": val_results["loss"],
                **{f"train_{k}": v for k, v in train_results["metrics"].items()},
                **{f"val_{k}": v for k, v in val_results["metrics"].items()}
            }
            wandb.log(metrics)

            # Сохранение лучшей модели
            if val_results["loss"] < best_val_loss:
                best_val_loss = val_results["loss"]
                torch.save(self.model.state_dict(), "best_model.pt")
                logger.info(f"Новая лучшая модель сохранена на эпохе {epoch}")

        torch.save(self.model.state_dict(), "final_model.pt")
        logger.info("Обучение завершено")


if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.train()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise