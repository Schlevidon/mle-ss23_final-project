import os

import torch
import torch.nn as nn

import numpy as np

import settings as s
from .globals import GAMMA, CRITERION, ACTIONS, DEVICE, ACTIONS_DICT, Transition, LR
from .helper import get_valid_actions, find_ideal_path
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

class QNetwork(nn.Module):
    # Define architecture
    @staticmethod
    def get_architecture() -> dict:
        INPUT_DIM = len(QNetwork.state_to_features(DUMMY_STATE))
        OUTPUT_DIM = len(ACTIONS)
        ARCHITECTURE = {
            "n_features" : [INPUT_DIM, 300, 100, 50, 20, OUTPUT_DIM],
            "activation" : nn.ReLU
        }
        return ARCHITECTURE

    def __init__(self, n_features, activation=nn.ReLU):
        super(QNetwork, self).__init__()
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
        
        # This is the dict before the game begins and after it ends
        if game_state is None:
            raise ValueError("Cannot convert empty state to feature (game_state should not be None)")

        # Construct features using multiple layers
        channels = []

        # Field (wall and crate locations)
        field_channel = game_state['field']

        field_shape = field_channel.shape
        channels.append(field_channel)

        # Coin positions
        coins = game_state['coins']

        coin_channel = np.zeros(field_shape) # 0 == no coin
        for c in coins:
            coin_channel[c] = 1 # 1 == coin
        channels.append(coin_channel)

        # Agents
        my_agent = game_state['self']
        other_agents = game_state['others']

        agent_channel = np.zeros(field_shape) # 0 == no agent
        agent_channel[my_agent[-1]] = -1 # -1 == my agent
        for agent in other_agents:
            agent_channel[agent[-1]] = 1 # 1 == enemy agent
        channels.append(agent_channel)

        # Bomb positions
        bombs = game_state['bombs']

        bomb_channel = np.zeros(field_shape) # 0 == no bomb
        for bomb in bombs:
            bomb_channel[bomb[0]] = bomb[1] + 1 # > 0 == timer of bomb + 1
        channels.append(bomb_channel)

        # Positions where an agent could place a bomb
        bombs_possible_channel = np.zeros(field_shape) # 0 == bomb can't be placed here
        for agent in other_agents + [my_agent]:
            bombs_possible_channel[agent[-1]] = int(agent[2]) # 1 == bomb could be placed here
        channels.append(bombs_possible_channel)

        # Explosions
        explosion_map = game_state['explosion_map']
        channels.append(explosion_map)
        # Steps
        relative_step = game_state['step']/s.MAX_STEPS

        # Combine channels
        stacked_channels = np.stack(channels)

        # Flatten for linear NN?
        output = stacked_channels.reshape(-1)
        np.append(output,relative_step)

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


class OneCoinNet(nn.Module):
    # Define architecture
    @staticmethod
    def get_architecture() -> dict:
        INPUT_DIM = len(OneCoinNet.state_to_features(DUMMY_STATE))
        OUTPUT_DIM = len(ACTIONS)
        ARCHITECTURE = {
            "n_features" : [INPUT_DIM, 10, OUTPUT_DIM],
            "activation" : nn.ReLU
        }
        return ARCHITECTURE

    def __init__(self, n_features, activation=nn.ReLU):
        super(OneCoinNet, self).__init__()
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
        # Features: [agent_x, agent_y, coin_x, coin_y, distance]
        my_agent = game_state['self']
        my_pos = my_agent[-1]

        # Assume that only one coin exists
        try:
            coin = game_state['coins'][0]
        except:
            coin = (8, 8)

        # Manhattan distance
        distance = abs(my_pos[0] - coin[0]) + abs(my_pos[1] - coin[1])
        
        features = [my_pos[0], my_pos[1], coin[0], coin[1], distance]
        return torch.tensor(features, dtype=torch.float32)
        
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
