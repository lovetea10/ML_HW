import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
from config import CELEBA_CONFIG
from torch.cuda.amp import autocast, GradScaler
import wandb


class FaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class CelebADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


def train_model():

    wandb.init(project="face-recognition", config=CELEBA_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = CELEBA_CONFIG
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    train_dataset = CelebADataset(config['output_dir'], 'train', transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = FaceNet().to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, _ = batch
            images = images.to(device)

            batch_size = images.size(0)
            indices = torch.randperm(batch_size)
            positive = images[indices]
            negative = images[torch.roll(indices, shifts=1)]

            optimizer.zero_grad()

            with autocast():
                anchor_emb = model(images)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            wandb.log({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss})

    os.makedirs(config['models_dir'], exist_ok=True)
    model_path = os.path.join(config['models_dir'], 'facenet_model.pth')
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully!")

    artifact = wandb.Artifact('facenet_model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train_model()