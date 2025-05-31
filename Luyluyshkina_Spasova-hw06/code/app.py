import pygame
from game.wr_flappy import FlappyEnvironment

def main():
    pygame.init()
    env = FlappyEnvironment(for_model=False)
    clock = pygame.time.Clock()
    running = True

    while running:
        action = [1, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    action = [0, 1]  # Flap
                if event.key == pygame.K_ESCAPE:
                    running = False

        _, _, done, pipes_passed = env.frame_step(action)  # Получаем количество пройденных труб
        if done:
            print(f"Game Over! Pipes passed: {pipes_passed}")
            env = FlappyEnvironment(for_model=False)

        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()