import torch
from face_recognition.model import get_model
from face_recognition.config import Config

def test_get_model():
    config = Config()
    model = get_model()
    
    assert isinstance(model, torch.nn.Module)
    
    assert next(model.parameters()).device == config.device
    
    for name, param in model.named_parameters():
        if 'block8' not in name and 'last_linear' not in name:
            assert not param.requires_grad
    
    input_tensor = torch.rand(2, 3, 160, 160).to(config.device)
    output = model(input_tensor)
    assert output.shape == (2, config.embedding_size)