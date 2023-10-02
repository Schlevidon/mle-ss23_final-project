import os

import torch
import torch.nn as nn

import numpy as np
from scipy.spatial.distance import cdist

import settings as s
from .globals import GAMMA, CRITERION, ACTIONS, DEVICE, ACTIONS_DICT, Transition, LR
from .helper import get_valid_actions
from .features import find_ideal_path, get_blast_coords, get_safety_feature, get_coin_feature,\
                     get_enemy_agent_feature, enemy_in_blast_coords, get_first_step_from_path, find_path_to_target
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

class DQNN(nn.Module):
    # Define architecture
    @staticmethod
    def get_architecture() -> dict:
        INPUT_DIM = len(DQNN.state_to_features(DUMMY_STATE))
        OUTPUT_DIM = len(ACTIONS)
        ARCHITECTURE = {
            "n_features" : [INPUT_DIM, 32, 64, 32, OUTPUT_DIM],
            "activation" : nn.ReLU
        }
        return ARCHITECTURE

    def __init__(self, n_features, activation=nn.ReLU):
        super(DQNN, self).__init__()
        # Define Layers
        layers = []
        n_layers = len(n_features)
        for i, (f_in, f_out) in enumerate(zip(n_features[:-1], n_features[1:])):
            lin = nn.Linear(f_in, f_out)
            layers.append(lin)

            # Add activation if not the last layer
            if i != n_layers-2:
                layers.append(activation()) 

        self.layers = nn.ModuleList(layers)
        self.apply(self._init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def save(self, folder_path='./model', file_name='my-model.pt'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)

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
                '''
                if selected_crate[1] < my_y:
                    crate_direction = "UP"
                if selected_crate[1] > my_y:
                    crate_direction = "DOWN"
                if selected_crate[0] < my_x:
                    crate_direction = "LEFT"
                if selected_crate[0] > my_x:
                    crate_direction = "RIGHT"
                '''
                # Remove target crate from field so pathfinding works
                field_temp = field.copy()
                field_temp[selected_crate] = 0

                path = find_path_to_target(my_pos, selected_crate, field_temp)
                if len(path)>1:
                    crate_direction = get_first_step_from_path(my_pos, path)

                    coin_or_crate_feature = ACTIONS_DICT[crate_direction]

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
        

        # Combine channels
        stacked_channels = torch.cat([
                              coin_or_crate_feature,
                              enemy_feature,
                              safety_feature,
                              bomb_attack_feature
                              ])
        # Flatten for linear NN?
        output = stacked_channels.flatten().squeeze()
        # Return output as tensor
        return torch.as_tensor(output, dtype=torch.float32)
        
    def train_step(self, agent, batch):
        # TODO: how to send tensors to GPU if available
        transitions = Transition(*zip(*batch))
        
        actions = transitions.action #batch_size
        old_states = torch.vstack(transitions.state).to(DEVICE) # batch_size x features
        new_states = torch.vstack(transitions.next_state).to(DEVICE) # batch_size x features
        reward = torch.tensor(transitions.reward, dtype=torch.float32).to(DEVICE) # batch_size
        new_states_dict = transitions.next_state_dict # batch_size

        mask_nan = torch.any(torch.isnan(new_states), 1) # batch_size
        a_idx = torch.tensor([ACTIONS_DICT[a] for a in actions]) # batch_size

        outputs = agent.model(old_states).to(DEVICE) # batch_size x 6
        
        target = outputs.clone().to(DEVICE) # batch_size x 6

        # For terminal states
        target[:, a_idx] = reward # batch_size
    
        # For non terminal states
        if not torch.any(mask_nan):
            valid_actions_masks = torch.vstack([torch.from_numpy(get_valid_actions(dct)) for (dct, done) in zip(new_states_dict, mask_nan) if not done]) # (batch_size - terminal states) x 6
            with torch.no_grad():
                Q_pred = agent.model(new_states[~mask_nan]) # (batch_size - teminal state) x 6
                Q_pred[~valid_actions_masks] = -float('inf') # invalid actions have Q-value -infinity -> never get selected in max
                Q_max = torch.max(Q_pred, 1)[0] # batch_size - terminal_states
            
            target[~mask_nan, a_idx[~mask_nan]] += GAMMA * Q_max # batch_size - terminal_states

        #add end of round 
        agent.optimizer.zero_grad()
        
        loss = CRITERION(outputs, target)
        loss.backward()
        agent.optimizer.step()