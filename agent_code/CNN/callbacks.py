import os
import pickle
import random
from . import model as m
import torch
import torch.nn as nn
from torch.nn.functional import softmax, normalize
import torch.optim as optim

import numpy as np
from collections import deque

#from coin_learning_agent
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import settings as s

from . import callbacks_rb as crb

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
    # Bombs possible
    bombs_possible = np.zeros(fshape)
    bombs_possible[my_pos] = int(game_state['self'][2])
    for agent in agent_list:
        bombs_possible[agent[-1]] = int(agent[2])
    channels.append(bombs_possible)
    # Bombs
    bomb_list = game_state['bombs']
    bombs = np.zeros(fshape)
    for bomb in bomb_list:
        bombs[bomb[0]] = bomb[1] + 1
    channels.append(bombs)
    # Explosions
    explosion_map = game_state['explosion_map']
    channels.append(explosion_map)
    
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    #return output
    return torch.as_tensor(stacked_channels, dtype=torch.float32)

ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']) # [1, 2, 3, 4, 5, 6]

# TODO: For submission update the global variable SCENARIO to SCENARIO = "classic"

# dummy state to determine feature dimension
DUMMY_STATE = {"round" : 0,
               "step" : 0,
               "field" : np.zeros((s.ROWS, s.COLS)),
               "bombs" : [],
               "explosion_map" : np.zeros((s.ROWS, s.COLS)),
               "coins" : [],
               "self" : ("", 0, False, (1, 1)),
               "others" : [],
               "user_input" : None
}

                                          

# Network architecture
INPUT_DIM = len(state_to_features(DUMMY_STATE)) # For a linear model
OUTPUT_DIM = len(ACTIONS)
DIMENSIONS = [INPUT_DIM, INPUT_DIM, 1, OUTPUT_DIM] #[6, 6, 6]
N_KERNELS_CONV = [3 for el in DIMENSIONS] # improve later
N_KERNELS_POOL = [2 for el in DIMENSIONS] # imporve later

ACTIVATION = nn.ReLU

# Parameters
OPTIMIZER_PARAMS = {
    "lr" : 1e-3,
    "eps" : 1e-9,
    "betas" : [0.9, 0.98]
}

BATCH_SIZE = 16

# TODO: Update EPS
EPS = 0.5
EPS_DECAY = 0.999
EPS_MIN = 0.001
# TODO: implement a minimal EPS
GAMMA = 0.99
RANDOM_SEED = None
CRITERION = torch.nn.MSELoss()

def setup(self):
    # TODO : maybe set seed to make results reproducible
    np.random.seed(RANDOM_SEED)
    self.PATH = "./model/my-model.pt" #'/'.join((MODEL_FOLDER,MODEL_NAME))
    self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self.model = m.CNN(DIMENSIONS, N_KERNELS_CONV, N_KERNELS_POOL, s.ROWS, s.COLS, ACTIVATION)
    
    if os.path.isfile(self.PATH):
        # TODO: Disable dropout and batch norm
        self.model.load_state_dict(torch.load(self.PATH, map_location = torch.device(self.DEVICE)))
    else:
        #self.model.initialize_weights()
        pass

    self.coordinate_history = deque([], 20)

    self.round_reward = 0
    self.round_score_history = []

    self.round_reward_history = []
    self.mean_round_reward_history = []

    self.step_reward_history = []

    self.eps_history = []

    self.batch_size = BATCH_SIZE
    self.gamma = GAMMA
    self.criterion = CRITERION
    self.optimizer_params = OPTIMIZER_PARAMS
    self.eps = EPS
    self.eps_decay = EPS_DECAY
    self.eps_min = EPS_MIN

    # For rule-based agent only
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0
    

#select eigentlich irgendsowas wie 
#if np.random.random() > self.eps: 
#    state = torch.tensor([observation]).to(self.Qvalues)
#    action = self.Q_evaluation.forward(state)
#    action = t.argmax(action).item
#else: 
#    action = np.random.choice(self.ACTIONS)
#    return action 

def act(self, game_state: dict) -> str:
    """
    if self.train:
        return crb.act(self, game_state)
    """

    valid_actions_mask = get_valid_actions(game_state)
    self.logger.debug(f'Valid actions: {ACTIONS[valid_actions_mask]}')
    
    
    if self.train and random.random() < self.eps:
        # Exploratory move
        self.logger.debug("Choosing action purely at random.")
        # TODO: weigh some actions with higher probability? 

        #pos_agent, pos_coin, field= game_state['self'][3], game_state['coins'], game_state['field']
        #selected_action =  find_ideal_path(pos_agent, pos_coin, field)
        selected_action = np.random.choice(ACTIONS[valid_actions_mask])
        """
        rb_action = crb.act(self, game_state)
        if rb_action is None:
            raise Exception("RB action none")
            rb_action = 'WAIT' 
        return rb_action
        """
        
    else:
        self.logger.debug("Querying model for action.")
        # Act greedily wrt to Q-function
        with torch.no_grad():
            Q_values = self.model(state_to_features(game_state))
            self.logger.debug(f"Q Values: {Q_values}")
            #Q_values = Q_values[valid_actions_mask]
        #print(Q_values,Q_values.shape)
        #Q_values = normalize(Q_values, p=1, dim=0)
        #probs = np.array(softmax(Q_values[valid_actions_mask],dim=0))
        #self.logger.debug(f"Probs: {probs}")

        #selected_action = np.random.choice(ACTIONS[valid_actions_mask], p=probs)
        
        self.logger.debug(f"Q Values: {Q_values}")
        max_Q = torch.max(Q_values[valid_actions_mask])
        mask = np.array((Q_values == max_Q)) & valid_actions_mask
        best_actions = ACTIONS[mask]
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
    bomb = False # disable bombs for now

    return np.array([up, right, down, left, True, bomb])

def tile_is_free(game_state, x, y):
    is_free = (game_state['field'][x, y] == 0)
    if is_free:
        for obstacle in game_state['bombs']: 
            is_free = is_free and (obstacle[0][0] != x or obstacle[0][1] != y)
        for obstacle in game_state['others']: 
            is_free = is_free and (obstacle[3][0] != x or obstacle[3][1] != y)
    return is_free
    




def find_ideal_path(pos_agent, pos_coin, field=None, bombs=None, explosion_map=None):
    
    field[field==1] = 2
    field[field==0] = 1
    grid = Grid(matrix=field)
    finder = AStarFinder()

    sx, sy = pos_agent
    start = grid.node(sx, sy)

    lengths = []
    for coin in pos_coin:
        cx, cy = coin
        end = grid.node(cx,cy)
        path, runs = finder.find_path(start, end, grid)
        grid.cleanup()
        lengths.append((len(path),path))

    grid.cleanup()
    lengths = sorted(lengths,key=lambda c : c[0])

    try:
        step0 = lengths[0][1][0]
        step1 = lengths[0][1][1]
    except:
        return 'WAIT'

    diff = np.array([step1.x,step1.y]) - np.array([step0.x,step0.y])

    if diff[0]==0:
        if diff[1]==1:
            move = 'DOWN'
        else:
            move = 'UP'

    elif diff[0]==1:
        move = 'RIGHT'
    else:
        move = 'LEFT'
    return move


