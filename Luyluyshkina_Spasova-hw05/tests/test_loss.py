import torch
from face_recognition.loss import TripletLoss

def test_triplet_loss():
    margin = 0.5
    loss_fn = TripletLoss(margin=margin)
    
    anchor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    positive = torch.tensor([[1.1, 2.1]], dtype=torch.float32)
    negative = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    
    loss = loss_fn(anchor, positive, negative)
    assert isinstance(loss, torch.Tensor)
    
    pos_dist = ((anchor - positive)**2).sum()
    neg_dist = ((anchor - negative)**2).sum()
    expected_loss = torch.relu(pos_dist - neg_dist + margin)
    
    assert torch.allclose(loss, expected_loss)