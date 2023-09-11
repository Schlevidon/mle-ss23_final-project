import os

import torch
import torch.nn as nn

import numpy as np
from scipy.spatial.distance import cdist

import settings as s
from .globals import GAMMA, ACTIONS_DICT, LR
from .helper import get_valid_actions, find_ideal_path, get_blast_coords, get_safety_feature
from . import callbacks_rb as crb

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

# __call__ -> Q_values form table
# train_step -> Update Q-table
# __init__ -> load Q-table or create new
# agent_x, agent_y, coin_x, coin_y
class OneCoinQTable:
    # Define architecture
    @staticmethod
    def get_architecture() -> dict:
        return {"dimensions" : (17, 17, 17, 17, 6)}

    def __init__(self, dimensions):
        self.table = torch.zeros(dimensions)
        # Init weights in a smarter way
        
    def __call__(self, feature):
        # feature = agent_x, agent_y, coin_x, coin_y
        return self.table[tuple(feature)] # output: 6 (= num_actions)
        
    def save(self, folder_path='./model', file_name='my-model.pt'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save(self.table, file_name)

    @staticmethod
    def state_to_features(game_state: dict, agent=None) -> torch.tensor:
        ''' Dict keys: round, step, field, bombs, explosion_map, my_agent, others'''
        # Features: [agent_x, agent_y, coin_x, coin_y, distance]
        my_agent = game_state['self']
        my_pos = my_agent[-1]

        # Assume that only one coin exists
        try:
            coin = game_state['coins'][0]
        except:
            coin = (8, 8)
        
        features = [my_pos[0], my_pos[1], coin[0], coin[1]]
        return torch.tensor(features)
        
    def train_step(self, agent, transition):
        # TODO: how to send tensors to GPU if available
        old_state, action, new_state, reward, new_state_dict = transition

        done = torch.any(torch.isnan(new_state)) # batch_size

        a_idx = ACTIONS_DICT[action]
        Q_old = agent.model(old_state) # 6
        Q_old = Q_old[a_idx] #  1

        target = reward - Q_old
       
        if not done:
            Q_new = agent.model(new_state)
            Q_max = torch.max(Q_new[get_valid_actions(new_state_dict)])
            target += GAMMA * Q_max
        
        self.table[old_state, a_idx] = Q_old + LR * target

class OneCoinQTableWithPath:
    # Define architecture
    @staticmethod
    def get_architecture() -> dict:
        return {"dimensions" : (6, 6)}

    def __init__(self, dimensions):
        self.table = torch.zeros(dimensions)
        # Init weights in a smarter way
        
    def __call__(self, feature):
        # feature = agent_x, agent_y, coin_x, coin_y
        return self.table[tuple(feature)] # output: 6 (= num_actions)
        
    def save(self, folder_path='./model', file_name='my-model.pt'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save(self.table, file_name)

    @staticmethod
    def state_to_features(game_state: dict, agent=None) -> torch.tensor:
        ''' Dict keys: round, step, field, bombs, explosion_map, my_agent, others'''
        # Features: [agent_x, agent_y, coin_x, coin_y, distance]
        my_agent = game_state['self']
        my_pos = my_agent[-1]

        # Assume that only one coin exists
        coin = game_state["coins"]

        action = find_ideal_path(my_pos, coin, game_state["field"])
        
        #action = crb.act(agent, game_state)
        action_num = ACTIONS_DICT[action]
        features = [action_num]

        return torch.tensor(features)
        
    def train_step(self, agent, transition):
        # TODO: how to send tensors to GPU if available
        old_state, action, new_state, reward, new_state_dict = transition

        done = torch.any(torch.isnan(new_state)) # batch_size

        a_idx = ACTIONS_DICT[action]
        Q_old = agent.model(old_state) # 6
        Q_old = Q_old[a_idx] #  1

        target = reward - Q_old
       
        if not done:
            Q_new = agent.model(new_state)
            Q_max = torch.max(Q_new[get_valid_actions(new_state_dict)])
            target += GAMMA * Q_max
        
        self.table[old_state, a_idx] = Q_old + LR * target

class QTable:
    # Define architecture
    @staticmethod
    def get_architecture() -> dict:
        return {"dimensions" : (#4, 4, 4, 4, # 4 tile information
                                2, 2, 2, 2, # nearest coin direction
                                2, 2, 2, 2, 2, # nearest crate direction and distance
                                2, 2, 2, 2, 2, # safety direction
                                6)} # actions

    def __init__(self, dimensions):
        self.table = torch.zeros(dimensions, dtype=torch.float64)
        self.table_stats = torch.zeros(dimensions)
        # Init weights in a smarter way
        
    def __call__(self, feature):
        # feature = agent_x, agent_y, coin_x, coin_y
        return self.table[tuple(feature)] # output: 6 (= num_actions)
        
    def save(self, folder_path='./model', file_name='my-model.pt'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save(self.table, file_name)
        torch.save(self.table_stats, file_name.replace('.pt','_stats.pt'))

    @staticmethod
    def state_to_features(game_state: dict, agent=None) -> torch.tensor:
        ''' Dict keys: round, step, field, bombs, explosion_map, my_agent, others'''
        # [UP, RIGHT, DOWN, LEFT]

        my_x, my_y = game_state['self'][-1]
        field = game_state["field"]
        explosion_map = game_state['explosion_map']
        bombs = game_state["bombs"]

        # Feature: Bomb safety 
        # 2^5 = 32 permutations
        safety_feature = get_safety_feature((my_x, my_y), field, explosion_map, bombs)

        # Feature: crate direction
        # 2^5 = 32 permutations
        # TODO : how detailed should the distance to the crate be
        crates = np.argwhere(field == 1)
        if len(crates) > 0:
            dist = np.squeeze(cdist([(my_x, my_y)], crates, metric="cityblock"))
            min_dist = np.min(dist)

            crate_targets = crates[dist == min_dist].reshape(-1, 2)
            
            crate_feature = torch.tensor([int(np.any(crate_targets[:, 1] < my_y)), # UP
                                    int(np.any(my_x < crate_targets[:, 0])), # RIGHT
                                    int(np.any(my_y < crate_targets[:, 1])), # DOWN
                                    int(np.any(crate_targets[:, 0] < my_x)), # LEFT
                                    int(min_dist == 1)]) # is the agent next to the crate?
        else:
            crate_feature = torch.tensor([0, 0, 0, 0, 0]) # No coins available

        
        """# Feature: immediate neighboring tile awareness
        # 3^4 = 81 permutations
        # 4^4 = 256 permutations
        # TODO: we can use get_valid_actions to reduce the parameter size to 2^4 for the exploding bombs
        
        exploding_bombs = [bomb for bomb in bombs if bomb[1] == 0]
        predicted_explosions = tuple(np.array(get_blast_coords(exploding_bombs, field)).T)

        field = field.copy()
        field[explosion_map > 1] = 2
        field[predicted_explosions] = 2
        
        field_feature = torch.tensor([field[my_x, my_y-1], #UP
                                      field[my_x+1, my_y], #RIGHT
                                      field[my_x, my_y + 1], #DOWN
                                      field[my_x - 1,my_y]]) #LEFT
        field_feature += 1 # shift from [-1, 0, 1, 2] to [0, 1, 2, 3]"""

        # Feature: coin direction
        # TODO : implement actual pathfinding for situations like below
        #what about 010
        #           0-10
        #           0a0
        # 2^4 = 16 permutations
        # TODO : if there are multiple coins with minimal distance currently
        # only the first coin is returned. We should either return all directions
        # or reduce features since UP/DOWN and LEFT/RIGHT are mutually exclusive
        if len(game_state["coins"]) > 0:
            coins = np.array(game_state["coins"])
            dist = np.squeeze(cdist([(my_x, my_y)], coins, metric="cityblock"))
            min_dist = np.min(dist)

            coin_targets = coins[dist == min_dist].reshape(-1, 2)
            coin_feature = torch.tensor([int(np.any(coin_targets[:, 1] < my_y)), # UP
                                    int(np.any(my_x < coin_targets[:, 0])), # RIGHT
                                    int(np.any(my_y < coin_targets[:, 1])), # DOWN
                                    int(np.any(coin_targets[:, 0] < my_x))]) #LEFT
        else:
            coin_feature = torch.tensor([0, 0, 0, 0]) # No coins available

        features = torch.cat([#field_feature,
                              coin_feature,
                              crate_feature,
                              safety_feature])
        return features
        
    def train_step(self, agent, transition):
        # TODO: how to send tensors to GPU if available
        old_state, action, new_state, reward, new_state_dict = transition

        done = torch.any(torch.isnan(new_state)) # batch_size

        a_idx = ACTIONS_DICT[action]
        Q_old = agent.model(old_state) # 6
        Q_old = Q_old[a_idx] #  1

        target = reward - Q_old
       
        if not done:
            Q_new = agent.model(new_state)
            Q_max = torch.max(Q_new[get_valid_actions(new_state_dict)])
            target += GAMMA * Q_max
        
        
        self.table[tuple(old_state)][a_idx] = Q_old + LR * target
        self.table_stats[tuple(old_state)][a_idx] += 1

# CNN State to feature
def state_to_features2(game_state: dict) -> np.array:
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