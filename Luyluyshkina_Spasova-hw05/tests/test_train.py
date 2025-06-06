import torch
from unittest.mock import patch, MagicMock
from face_recognition.train import train

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_train_process():
    with patch('train.Config') as mock_config, \
         patch('train.setup_wandb') as mock_setup_wandb, \
         patch('train.get_data_loaders') as mock_get_loaders, \
         patch('train.get_model') as mock_get_model, \
         patch('train.TripletLoss') as mock_loss, \
         patch('train.optim.Adam') as mock_optimizer, \
         patch('train.GradScaler') as mock_scaler, \
         patch('train.wandb') as mock_wandb:
        
        mock_config.return_value = MagicMock(
            epochs=1,
            batch_size=2,
            model_save_path='test_model.pth'
        )
        mock_get_loaders.return_value = (
            [(torch.rand(2, 3, 160, 160), torch.tensor([0, 1]))],  
            [(torch.rand(2, 3, 160, 160), torch.tensor([0, 1]))]   
        )
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_loss.return_value = MagicMock(return_value=torch.tensor(0.5))
        
        train()
        
        mock_setup_wandb.assert_called_once()
        mock_get_loaders.assert_called_once()
        mock_get_model.assert_called_once()
        mock_model.train.assert_called()
        mock_model.eval.assert_called()
        assert mock_wandb.finish.called