import pygame
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import time
from collections import deque
from itertools import cycle
import os
from game.wr_flappy import FlappyEnvironment
from model import FlappyDQN
from tools import ExperienceBuffer, preprocess_image, stack_images, DEVICE
import game.flappy_bird_utils as flappy_bird_utils

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird - Обучение')
IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

os.chdir(r'C:\Users\ASUS\Desktop\lovetea\6_sem\My_ML\ML_HW')

def train_model():
    session_id = int(time.time())
    wandb.init(
        project="flappy-bird-dqn",
        name=f"training-session-{session_id}",
        config={
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "batch_size": 64,
            "lr": 2e-4,
        }
    )
    env = FlappyEnvironment(for_model=False)
    model = FlappyDQN().to(DEVICE)
    target_model = FlappyDQN().to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = nn.SmoothL1Loss()
    memory = ExperienceBuffer(model.memory_size)
    steps_done = 0
    max_score = 0
    session_id = int(time.time())

    start_time = time.time()
    total_pipes = 0

    wandb.watch(model, log="all")
    for episode in range(5000):

        env = FlappyEnvironment(for_model=False)
        frame_buffer = deque(maxlen=4)
        total_reward = 0
        done = False
        episode_score = 0

        image, reward, done, score = env.frame_step([1, 0])
        processed = preprocess_image(np.transpose(image, (1, 0, 2)))
        for _ in range(4):
            frame_buffer.append(processed)
        state = stack_images(frame_buffer)

        while not done:
            steps_done += 1
            epsilon = model.epsilon_end + (model.epsilon_start - model.epsilon_end) * np.exp(
                -steps_done / 10000)

            if random.random() < epsilon:
                action_idx = random.randint(0, 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action_idx = model(state_tensor).argmax().item()

            action = [int(action_idx == 0), int(action_idx == 1)]

            next_image, reward, done, score = env.frame_step(action)
            total_reward += reward
            episode_score = max(episode_score, score)
            total_pipes = score

            next_processed = preprocess_image(np.transpose(next_image, (1, 0, 2)))
            frame_buffer.append(next_processed)
            next_state = stack_images(frame_buffer)

            memory.add(state, action_idx, reward, next_state, done)

            if len(memory) > model.batch_size:
                batch = memory.sample(model.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states)).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                with torch.no_grad():
                    next_q = target_model(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * model.gamma * next_q

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if steps_done % 500 == 0:
                    target_model.load_state_dict(model.state_dict())

            state = next_state

            elapsed_time = time.time() - start_time

            wandb.log({
                "Episode": episode + 1,
                "Pipes Passed": total_pipes,
                "Max Score": max_score,
                "Reward": total_reward,
                "Epsilon": epsilon,
                "Loss": loss.item() if 'loss' in locals() else 0,
                "Time (sec)": elapsed_time
            })
            print(f"Сессия: {session_id} | Эпизод: {episode + 1:4d}/5000 | "
                  f"Пройдено труб: {total_pipes:3d} | Макс. рекорд: {max_score:3d} | "
                  f"Награда: {total_reward:.2f} | Epsilon: {epsilon:.3f} | "
                  f"Время: {elapsed_time:.2f} сек", end="\r")
            time.sleep(1/FPS)

            if done:
               break

        max_score = max(max_score, episode_score)

        elapsed_time = time.time() - start_time
        print(f"Сессия: {session_id} | Эпизод: {episode + 1:4d}/5000 | "
              f"Пройдено труб: {total_pipes:3d} | Макс. рекорд: {max_score:3d} | "
              f"Награда: {total_reward:.2f} | Epsilon: {epsilon:.3f} | "
              f"Время: {elapsed_time:.2f} сек")

        if max_score >= 100:
            model_name = f"flappy_model_{session_id}_score{max_score}.pth"
            torch.save(model.state_dict(), model_name)
            wandb.save(model_name)
            print(f"Модель сохранена с рекордом {max_score}!")
            break

        if (episode + 1) % 1000 == 0:
            model_name = f"flappy_model_{session_id}_ep{episode + 1}.pth"
            torch.save(model.state_dict(), model_name)
            wandb.save(model_name)

    torch.save(model.state_dict(), "flappy_model_final.pth")
    wandb.save("flappy_model_final.pth")
    wandb.finish()
    print("Обучение завершено. Финальная модель сохранена.")

if __name__ == "__main__":
    train_model()