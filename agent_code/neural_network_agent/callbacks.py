import os
import pickle
import random
import model as m
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MAX_MEMORY = 100_000

# Network architecture
DIMENSIONS = [10, 20, 5]
ACTIVATION = nn.ReLU
OPTIMIZER_PARAMS = {
    "lr" : 1e-3,
    "eps" : 1e-9,
    "betas" : [0.9, 0.98]
}
EPS_0 = 1
EPS_DECAY = 0.9

def setup(self):
    # Set globals
    self.PATH = "./model/my-model.pt" #'/'.join((MODEL_FOLDER,MODEL_NAME))
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    self.model = m.QNetwork(DIMENSIONS, ACTIVATION)
    
    if os.path.isfile(PATH):
        # Disable dropout 
        self.model.load_state_dict(torch.load(path, map_location = torch.device(self.DEVICE)))
    else:
        self.model.initialize_weights()
    
    # TODO: load optimizer values
    self.optimizer = optim.Adam(model.parameters(), **OPTIMIZER_PARAMS)

    #Memory
    self.memory = deque(maxlen=MAX_MEMORY) # popleft()



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # TODO: get_actual_state(self.game_state) => derive better state representation?
    if self.train and random.random() < EPS_0:
        # Exploratory move
        self.logger.debug("Choosing action purely at random.")
        valid_actions_mask = get_valid_actions(game_state)
        # TODO: weigh some actions with higher probability?
        return np.random.choice(ACTIONS[valid_actions_mask])

    self.logger.debug("Querying model for action.")
    # Act greedily wrt to Q-function
    return ACTIONS[np.argmax(self.model.forward()[valid_actions_mask])]
    # Stochastic policy
    # return np.random.choice(ACTIONS, p=self.model)


def get_valid_actions(game_state) -> np.array:
    '''    round = game_state['round']
        step = game_state['step']
        field = game_state['field']
        bombs = game_state['bombs']
        explosion_map = game_state['explosion_map']
        coins = game_state['coins']
        my_agent = game_state['self']
        others = game_state['others']'''
    # TODO : extract state once for all tiles to improve performance?
    agent_x, agent_y = game_state['self'][3]
    up = tile_is_free(agent_x, agent_y - 1)
    down = tile_is_free(agent_x, agent_y + 1)
    left =  tile_is_free(agent_x - 1, agent_y)
    right =  tile_is_free(agent_x + 1, agent_y)
    bomb = game_state["self"][2]

    return [up, right, down, left, True, bomb]


def tile_is_free(x, y):
    is_free = (game_state['field'][x, y] == 0)
    if is_free:
        for obstacle in game_state['bombs']: 
            is_free = is_free and (obstacle[0][0] != x or obstacle[0][1] != y)
        for obstacle in game_state['others']: 
            is_free = is_free and (obstacle[3][0] != x or obstacle[3][1] != y)
    return is_free
    

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
