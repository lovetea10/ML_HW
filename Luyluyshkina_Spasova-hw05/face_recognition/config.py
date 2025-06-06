import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

CELEBA_CONFIG = {
    'input_dir': os.path.join(OUTPUT_DIR, 'input'),
    'output_dir': OUTPUT_DIR,
    'models_dir': MODELS_DIR,
    'target_size': (224, 224),
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'augment_count': 2,
    'max_samples': None,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 10
}