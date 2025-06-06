import torch
from face_recognition.dataset import FaceDataset, get_transforms
from torchvision import transforms
import pytest

def test_get_transforms():
    train_transform, val_transform = get_transforms()
    assert isinstance(train_transform, transforms.Compose)
    assert isinstance(val_transform, transforms.Compose)
    assert len(train_transform.transforms) == 4
    assert len(val_transform.transforms) == 3

@pytest.fixture

def mock_dataset(tmp_path):
    from PIL import Image
    import numpy as np
    
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    
    for class_id in ["class1", "class2"]:
        class_dir = data_dir / class_id
        class_dir.mkdir()
        for i in range(3):
            img_path = class_dir / f"image_{i}.jpg"
            img_array = np.random.rand(160, 160, 3) * 255
            img = Image.fromarray(img_array.astype('uint8')).resize((160, 160))
            img.save(img_path)
    
    return str(data_dir)