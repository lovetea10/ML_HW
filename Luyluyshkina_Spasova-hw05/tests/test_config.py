from face_recognition.config import Config

def test_config_defaults():
    config = Config()
    assert config.batch_size == 256
    assert config.epochs == 3
    assert config.lr == 0.001
    assert config.margin == 0.5
    assert config.embedding_size == 512
    assert config.data_root == 'data/processed'
    assert config.model_save_path == 'models/facenet.pth'