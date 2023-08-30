import os
import pickle
import random
from . import model as m
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque


ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']) # [1, 2, 3, 4, 5, 6]

MAX_MEMORY = 100_000

# Network architecture
DIMENSIONS = [867, 20, 6]
ACTIVATION = nn.ReLU

# Parameters
OPTIMIZER_PARAMS = {
    "lr" : 1e-3,
    "eps" : 1e-9,
    "betas" : [0.9, 0.98]
}

BATCH_SIZE = 32

# TODO: Update EPS
EPS = 1
EPS_DECAY = 0.999
GAMMA = 1
RANDOM_SEED = None
CRITERION = torch.nn.MSELoss()

def setup(self):
    # TODO : maybe set seed to make results reproducible
    np.random.seed(RANDOM_SEED)
    self.PATH = "./model/my-model.pt" #'/'.join((MODEL_FOLDER,MODEL_NAME))
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    self.model = m.QNetwork(DIMENSIONS, ACTIVATION)
    
    if os.path.isfile(self.PATH):
        # TODO: Disable dropout and batch norm
        self.model.load_state_dict(torch.load(self.PATH, map_location = torch.device(self.DEVICE)))
    else:
        #self.model.initialize_weights()
        pass

    #Memory
    self.memory = deque(maxlen=MAX_MEMORY) # popleft()

    self.batch_size = BATCH_SIZE
    self.gamma = GAMMA
    self.criterion = CRITERION
    self.optimizer_params = OPTIMIZER_PARAMS
    self.eps = EPS
    self.eps_decay = EPS_DECAY



def act(self, game_state: dict) -> str:
    valid_actions_mask = get_valid_actions(game_state)
    self.logger.debug(f'Valid actions: {ACTIONS[valid_actions_mask]}')

    if self.train and random.random() < self.eps:
        # Exploratory move
        self.logger.debug("Choosing action purely at random.")
        # TODO: weigh some actions with higher probability?        
        selected_action = np.random.choice(ACTIONS[valid_actions_mask])
    else:
        self.logger.debug("Querying model for action.")
        # Act greedily wrt to Q-function
        Q_values = self.model(state_to_features(game_state))
        max_Q = torch.max(Q_values[valid_actions_mask])
        best_actions = ACTIONS[mask:=np.array((Q_values == max_Q)) & valid_actions_mask]
        self.logger.debug(f"Mask: {mask}")
        self.logger.debug(f"Actions: {ACTIONS}")
        self.logger.debug(f"Best actions: {best_actions}")
        selected_action = np.random.choice(best_actions)
    
    self.logger.debug(f"Selected action: {selected_action}")
    
    return selected_action

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
    up = tile_is_free(game_state, agent_x, agent_y - 1)
    down = tile_is_free(game_state, agent_x, agent_y + 1)
    left =  tile_is_free(game_state, agent_x - 1, agent_y)
    right =  tile_is_free(game_state, agent_x + 1, agent_y)
    bomb = game_state["self"][2]

    return np.array([up, right, down, left, True, bomb])

def tile_is_free(game_state, x, y):
    is_free = (game_state['field'][x, y] == 0)
    if is_free:
        for obstacle in game_state['bombs']: 
            is_free = is_free and (obstacle[0][0] != x or obstacle[0][1] != y)
        for obstacle in game_state['others']: 
            is_free = is_free and (obstacle[3][0] != x or obstacle[3][1] != y)
    return is_free
    

def state_to_features(game_state: dict) -> np.array:
    '''    round = game_state['round']
        step = game_state['step']
        field = game_state['field']
        bombs = game_state['bombs']
        explosion_map = game_state['explosion_map']
        coins = game_state['coins']
        my_agent = game_state['self']
        others = game_state['others']'''
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    # Field
    field = game_state['field']
    fshape = field.shape
    channels.append(field)
    # Coins
    coin_list = game_state['coins']
    coins = np.zeros(fshape)
    for coin in coin_list:
        coins[coin] = 1
    channels.append(coins)
    # Agents
    my_pos = game_state['self'][-1]
    agent_list = game_state['others']
    agents = np.zeros(fshape)
    agents[my_pos] = -1
    for agent in agent_list:
        agents[agent[-1]] = 1
    channels.append(agents)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    output = stacked_channels.reshape(-1)
    #return output
    return torch.as_tensor(output, dtype=torch.float32)
