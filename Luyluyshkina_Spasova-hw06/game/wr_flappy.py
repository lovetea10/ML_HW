import os
import random
from itertools import cycle
import pygame
import numpy as np
import game.flappy_bird_utils as flappy_bird_utils

os.chdir(r'C:\Users\ASUS\Desktop\lovetea\6_sem\My_ML\ML_HW\Luyluyshkina_Spasova-hw06')


class FlappyEnvironment:
    def __init__(self, for_model=False, pipe_speed=-4, flap_speed=-9, rot_speed=3):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH
        self.for_model = for_model
        self.pipes_passed = 0

        self.pipeVelX = pipe_speed
        self.playerFlapAcc = flap_speed
        self.playerVelRot = rot_speed

        self.playerVelY = -2
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapped = False
        self.playerRot = 40
        self.playerRotThr = 20
        self.flap_count = 0

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

    def frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                self.flap_count += 1
                SOUNDS['wing'].play()

        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                self.pipes_passed += 1
                reward += 5.0
                SOUNDS['point'].play()

        closest_pipe_idx = 0
        for i, pipe in enumerate(self.upperPipes):
            if pipe['x'] + PIPE_WIDTH > self.playerx:
                closest_pipe_idx = i
                break

        upper_pipe_y = self.upperPipes[closest_pipe_idx]['y']
        lower_pipe_y = self.lowerPipes[closest_pipe_idx]['y']
        gap_center = (upper_pipe_y + PIPE_HEIGHT + lower_pipe_y) / 2
        bird_center = self.playery + PLAYER_HEIGHT / 2
        distance_to_gap = abs(bird_center - gap_center)

        proximity_reward = max(0, 2.0 - distance_to_gap / 20.0)
        reward += proximity_reward

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
            self.playerRot = 45

        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        isCrash = checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)

        safe_zone_top = SCREENHEIGHT * 0.3
        safe_zone_bottom = BASEY - PLAYER_HEIGHT - 30
        optimal_zone_center = (safe_zone_top + safe_zone_bottom) / 2

        if self.playery < safe_zone_top or self.playery > safe_zone_bottom:
            height_penalty = -0.1 * abs(self.playery - optimal_zone_center) / (SCREENHEIGHT / 2)
            reward += height_penalty

        if isCrash:
            terminal = True
            SOUNDS['hit'].play()
            SOUNDS['die'].play()
            reward = -10.0
            self.__init__(for_model=self.for_model, pipe_speed=self.pipeVelX,
                          flap_speed=self.playerFlapAcc, rot_speed=self.playerVelRot)

        if not self.for_model:
            SCREEN.blit(IMAGES['background'], (0, 0))
            for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
                SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
            SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
            showScore(self.pipes_passed)

            visibleRot = self.playerRotThr
            if self.playerRot <= self.playerRotThr:
                visibleRot = self.playerRot
            playerSurface = pygame.transform.rotate(IMAGES['player'][self.playerIndex], visibleRot)
            SCREEN.blit(playerSurface, (self.playerx, self.playery))
            pygame.display.update()

        SCREEN_for_model = SCREEN.copy()
        SCREEN_for_model.blit(IMAGES['background_for_model'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN_for_model.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN_for_model.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN_for_model.blit(IMAGES['base'], (self.basex, BASEY))
        playerSurface = pygame.transform.rotate(IMAGES['player'][self.playerIndex], self.playerRot)
        SCREEN_for_model.blit(playerSurface, (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(SCREEN_for_model)
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminal, self.pipes_passed

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')
IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79
PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


def getRandomPipe():
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs) - 1)
    gapY = gapYs[index]
    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10
    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def showScore(score):
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = sum(IMAGES['numbers'][digit].get_width() for digit in scoreDigits)
    Xoffset = (SCREENWIDTH - totalWidth) / 2
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        pHitMask = HITMASKS['player'][pi]
        uHitmask = HITMASKS['pipe'][0]
        lHitmask = HITMASKS['pipe'][1]
        uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
        lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)
        if uCollide or lCollide:
            return True
    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False