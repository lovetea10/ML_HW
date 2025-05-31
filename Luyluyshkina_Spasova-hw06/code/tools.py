import numpy as np
import torch
import random
import cv2
from collections import deque

class ExperienceBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def preprocess_image(image):
    image = image[50:-20, :]  # Crop irrelevant areas
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold((blurred * 255).astype(np.uint8), 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    thresholded = thresholded.astype(np.float32) / 255.0
    resized = cv2.resize(thresholded, (80, 80), interpolation=cv2.INTER_AREA)
    return resized

def stack_images(images):
    return np.stack(images, axis=0)

def select_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB")
        print(f"Memory Cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB")
    return device

DEVICE = select_device()