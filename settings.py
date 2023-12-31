#Test
import logging
from pathlib import Path

from fallbacks import pygame

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 7#17
ROWS = 7#17
SCENARIOS = {
    # modes useful for agent development
	"empty": {
        "CRATE_DENSITY": 0, 
        "COIN_COUNT": 0 
    },	
    "one-coin": {
        "CRATE_DENSITY": 0, 
        "COIN_COUNT": 1 
    },
    "one-crate": {
        "CRATE_DENSITY": 1/177, 
        "COIN_COUNT": 1 
    },
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 50
    },
    "coin-heaven2": {
        "CRATE_DENSITY": 0.1,
        "COIN_COUNT": 10
    },
    "coin-heaven3": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 200
    },
    "coin5": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 5
    },
    "crates": {
        "CRATE_DENSITY": 0.1,
        "COIN_COUNT": 10
    }
    ,
    "loot-crate": { 
        "CRATE_DENSITY": 0.75, 
        "COIN_COUNT": 50 
    }, 
    # this is the tournament game mode
    "classic": {
        "CRATE_DENSITY": 0.75,
        "COIN_COUNT": 9
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4
#MAX_AGENTS = 2 #changed

# Round properties
MAX_STEPS = 200#400

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent / "assets"

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
LOG_GAME = logging.WARNING#INFO
LOG_AGENT_WRAPPER = logging.WARNING#INFO
LOG_AGENT_CODE = logging.WARNING#DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
