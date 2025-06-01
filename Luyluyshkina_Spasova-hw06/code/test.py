import torch
import numpy as np
from collections import deque
from game.wr_flappy import FlappyEnvironment
from model import FlappyDQN
from tools import preprocess_image, stack_images, DEVICE

def evaluate_model():
    env = FlappyEnvironment(for_model=False)
    model = FlappyDQN().to(DEVICE)
    try:
        model.load_state_dict(torch.load("flappy_model_final.pth", map_location=DEVICE))
    except FileNotFoundError:
        print("Error: Model file 'flappy_model_final.pth' not found.")
        return
    model.eval()

    frame_buffer = deque(maxlen=4)
    max_score = 0
    episode = 0

    while max_score < 100:
        env = FlappyEnvironment(for_model=False)
        frame_buffer.clear()
        score = 0
        done = False

        image, reward, done, _ = env.frame_step([1, 0])
        processed = preprocess_image(np.transpose(image, (1, 0, 2)))
        for _ in range(4):
            frame_buffer.append(processed)
        state = stack_images(frame_buffer)

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = model(state_tensor)
                action_idx = q_values.argmax().item()

            action = [int(action_idx == 0), int(action_idx == 1)]
            image, reward, done, score = env.frame_step(action)

            processed = preprocess_image(np.transpose(image, (1, 0, 2)))
            frame_buffer.append(processed)
            state = stack_images(frame_buffer)

        max_score = max(max_score, score)
        episode += 1
        print(f"Episode {episode}: Score = {score}, Max Score = {max_score}")

    print(f"Achieved {max_score} pipes after {episode} episodes.")

if __name__ == "__main__":
    evaluate_model()