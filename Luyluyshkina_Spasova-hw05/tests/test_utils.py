import pytest
import torch
import wandb
from unittest.mock import patch, MagicMock
from face_recognition.utils import (
    setup_wandb,
    get_data_loaders,
    train_epoch,
    validate_epoch,
    save_model
)
from face_recognition.config import Config

@pytest.fixture

def mock_model():
    model = MagicMock()
    model.return_value = torch.rand(10, 512)  
    return model

@patch('wandb.init')
@patch('wandb.config.update')
def test_setup_wandb(mock_update, mock_init):
    config = Config()
    setup_wandb(config)
    
    mock_init.assert_called_once()
    mock_update.assert_called_once_with(config.__dict__)

@patch('wandb.log')
@patch('wandb.init')

def test_get_data_loaders(mock_dataset):  
    config = Config()
    config.data_root = mock_dataset 
    
    train_loader, val_loader = get_data_loaders(config)
    
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert train_loader.batch_size == config.batch_size
    assert val_loader.batch_size == config.batch_size

def test_train_epoch(mock_model):
    config = Config()
    train_loader = [(torch.rand(2, 3, 160, 160), torch.tensor([0, 1]))]
    criterion = MagicMock(return_value=torch.tensor(0.5))
    optimizer = MagicMock()
    scaler = MagicMock()
    
    with patch('wandb.log') as mock_log:
        loss = train_epoch(mock_model, train_loader, criterion, optimizer, scaler, config, 0)
        assert isinstance(loss, float)
        mock_log.assert_called()

def test_validate_epoch(mock_model):
    config = Config()
    val_loader = [(torch.rand(2, 3, 160, 160), torch.tensor([0, 1]))]
    criterion = MagicMock(return_value=torch.tensor(0.5)) 
    
    loss = validate_epoch(mock_model, val_loader, criterion, config)
    assert isinstance(loss, float)

def test_save_model(tmp_path):
    config = Config()
    config.model_save_path = str(tmp_path / "model.pth")
    
    from face_recognition.model import get_model
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters())
    
    with patch('wandb.save') as mock_wandb_save:
        save_model(model, optimizer, 1, 0.5, config)
        
        assert (tmp_path / "model.pth").exists()
        mock_wandb_save.assert_called_once_with(config.model_save_path)