import pytest
import numpy as np
import torch
from tools import preprocess_image
from model import FlappyDQN
from game.wr_flappy import FlappyEnvironment


def test_preprocessing():
    env = FlappyEnvironment(for_model=True)
    image, _, _, _ = env.frame_step([1, 0])
    processed = preprocess_image(np.transpose(image, (1, 0, 2)))

    assert processed.shape == (80, 80)
    assert processed.dtype == np.float32
    assert np.allclose(processed.min(), 0, atol=1e-6)
    assert np.allclose(processed.max(), 1, atol=1e-5)
    assert np.all(processed >= -1e-6)
    assert np.all(processed <= 1 + 1e-5)


def test_frame_step_output():
    env = FlappyEnvironment(for_model=True)
    image, reward, done, score = env.frame_step([1, 0])

    assert isinstance(image, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(score, int)

def test_model_architecture():
    model = FlappyDQN()
    test_input = torch.randn(1, 4, 80, 80)
    output = model(test_input)

    assert output.shape == (1, 2)
    assert not torch.isnan(output).any()


def test_experience_buffer():
    from tools import ExperienceBuffer
    buffer = ExperienceBuffer(100)

    for i in range(150):
        buffer.add(i, i % 2, float(i), i + 1, i % 10 == 0)

    assert len(buffer) == 100
    sample = buffer.sample(10)
    assert len(sample) == 10