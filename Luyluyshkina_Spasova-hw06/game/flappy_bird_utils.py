import random
import pygame

PLAYERS_LIST = (
    (
        'Flappy_Bird_assets/Game Objects/yellowbird-upflap.png',
        'Flappy_Bird_assets/Game Objects/yellowbird-midflap.png',
        'Flappy_Bird_assets/Game Objects/yellowbird-downflap.png',
    ),
)

BACKGROUNDS_LIST = (
    'Flappy_Bird_assets/Game Objects/background-day.png',
    'Flappy_Bird_assets/Game Objects/background-night.png',
)

BACKGROUNDS_LIST_FOR_MODEL = 'Flappy_Bird_assets/Game Objects/background-black.png'

# List of pipes
PIPES_LIST = (
    'Flappy_Bird_assets/Game Objects/pipe-green.png',
)

SOUNDS_LIST = {
    'die': 'Flappy_Bird_assets/Sound Effects/die.wav',
    'hit': 'Flappy_Bird_assets/Sound Effects/hit.wav',
    'point': 'Flappy_Bird_assets/Sound Effects/point.wav',
    'swoosh': 'Flappy_Bird_assets/Sound Effects/swoosh.wav',
    'wing': 'Flappy_Bird_assets/Sound Effects/wing.wav'
}

def load():
    PLAYER_PATH = (
        'Flappy_Bird_assets/Game Objects/yellowbird-upflap.png',
        'Flappy_Bird_assets/Game Objects/yellowbird-midflap.png',
        'Flappy_Bird_assets/Game Objects/yellowbird-downflap.png',
    )

    PIPE_PATH = 'Flappy_Bird_assets/Game Objects/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    IMAGES['numbers'] = tuple(
        pygame.image.load(f'Flappy_Bird_assets/UI/Numbers/{i}.png').convert_alpha()
        for i in range(10)
    )

    IMAGES['base'] = pygame.image.load('Flappy_Bird_assets/Game Objects/base.png').convert_alpha()

    pygame.mixer.init()
    for sound_name, sound_path in SOUNDS_LIST.items():
        SOUNDS[sound_name] = pygame.mixer.Sound(sound_path)

    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()
    IMAGES['background_for_model'] = pygame.image.load(BACKGROUNDS_LIST_FOR_MODEL).convert()

    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.flip(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):

    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask