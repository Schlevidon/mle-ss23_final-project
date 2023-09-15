import os

import torch
import torch.nn as nn

import numpy as np
from scipy.spatial.distance import cdist

import settings as s
from .globals import GAMMA, ACTIONS_DICT, LR
from .helper import get_valid_actions
from .features import find_ideal_path, get_blast_coords, get_safety_feature, get_coin_feature, get_enemy_agent_feature, enemy_in_blast_coords
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
        return {"dimensions" : (5, # coin or crate feature
                                5, # enemy direction feature
                                2, 2, 2, 2, 2, 2, # safety feature
                                2, 2, # bomb atack feature (agent next to crate, enemy in blast coords)
                                6)} # actions

        """return {"dimensions" : (#4, 4, 4, 4, # 4 tile information
                                2, 2, 2, 2, # nearest coin direction
                                2, 2, 2, 2, 2, # nearest crate direction and distance
                                2, 2, 2, 2, 2, # safety direction
                                6)} # actions"""

    def __init__(self, dimensions):
        self.table = torch.zeros(dimensions, dtype=torch.float64)
        # TODO : Init weights in a smarter way
        
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
        # [UP, RIGHT, DOWN, LEFT]
        my_agent = game_state["self"]
        my_x, my_y = my_pos = my_agent[-1]
        bomb_avail = my_agent[2]

        other_agents = game_state["others"]
        other_pos = [agent[-1] for agent in other_agents]

        field = game_state["field"]

        explosion_map = game_state['explosion_map']
        bombs = game_state["bombs"]
        coins = game_state["coins"]

        # Feature: Object of interest direction
        # Coin if available else crate

        coin_or_crate_feature = 4 # if no crate or coin

        coin_direction = get_coin_feature(my_pos, other_pos, coins, field)
        if coin_direction is not None: # coin
            coin_or_crate_feature = ACTIONS_DICT[coin_direction]

            # Purely based on distance
            """coins = np.array(game_state["coins"])
            dist = np.squeeze(cdist([(my_x, my_y)], coins, metric="cityblock"))
            min_dist = np.min(dist)

            coin_targets = coins[dist == min_dist].reshape(-1, 2)
            coin_feature = torch.tensor([int(np.any(coin_targets[:, 1] < my_y)), # UP
                                    int(np.any(my_x < coin_targets[:, 0])), # RIGHT
                                    int(np.any(my_y < coin_targets[:, 1])), # DOWN
                                    int(np.any(coin_targets[:, 0] < my_x))]) #LEFT"""
        else: # crate
            crates = np.argwhere(field == 1)
            if len(crates) > 0:
                dist = np.squeeze(cdist([(my_x, my_y)], crates, metric="cityblock"))
                min_dist = np.min(dist)

                crate_targets = crates[dist == min_dist].reshape(-1, 2)

                # TODO : Select which crate to target if there are multiple with minimal distance
                # For now take the first crate in the list
                selected_crate = crate_targets[0]
                
                # TODO : if the crate is diagonal which direction should be chosen?
                # For now the x coordinate is preferred
                if selected_crate[1] < my_y:
                    crate_direction = "UP"
                if selected_crate[1] > my_y:
                    crate_direction = "DOWN"
                if selected_crate[0] < my_x:
                    crate_direction = "LEFT"
                if selected_crate[0] > my_x:
                    crate_direction = "RIGHT"

                coin_or_crate_feature = ACTIONS_DICT[crate_direction]
                
                """crate_feature = torch.tensor([int(np.any(crate_targets[:, 1] < my_y)), # UP
                                        int(np.any(my_x < crate_targets[:, 0])), # RIGHT
                                        int(np.any(my_y < crate_targets[:, 1])), # DOWN
                                        int(np.any(crate_targets[:, 0] < my_x)), # LEFT
                                        #int(min_dist == 1), # is the agent next to the crate?
                                        ])"""
        # Reshape scalar to match other feature
        coin_or_crate_feature = torch.tensor(coin_or_crate_feature).view(1)
        
        # Feature for direction of enemy agent
        enemy_direction = get_enemy_agent_feature(my_pos, other_pos, field)
        enemy_feature = 4 # no enemy agents

        if enemy_direction is not None:
            enemy_feature = ACTIONS_DICT[enemy_direction]

        enemy_feature = torch.tensor(enemy_feature).view(1)

        # Feature: Bomb safety 
        # 2^6 = 64 permutations
        safety_feature = get_safety_feature((my_x, my_y), field, explosion_map, bombs)

        # Bomb attack feature

        # Check if agent next to a crate
        adjacent_fields = [(my_x, my_y - 1), (my_x, my_y + 1), (my_x - 1, my_y), (my_x + 1, my_y)]

        agent_next_to_crate = 0
        for coords in adjacent_fields:
            if field[coords] == 1:
                agent_next_to_crate = 1
                break

        # Check if enemy in blast radius
        enemy_in_blast_coords_feature = int(enemy_in_blast_coords(my_pos, other_pos, field))

        # TODO : add another more sophisticated attack feature?

        bomb_attack_feature = torch.tensor([agent_next_to_crate, enemy_in_blast_coords_feature])

        features = torch.cat([
                              coin_or_crate_feature,
                              enemy_feature,
                              safety_feature,
                              bomb_attack_feature
                              ])
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