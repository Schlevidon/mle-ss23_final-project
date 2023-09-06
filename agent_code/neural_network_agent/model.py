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
        return self.table[*feature] # output: 6 (= num_actions)
        
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
        return self.table[*feature] # output: 6 (= num_actions)
        
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

"""  
environment_rows = 17
environment_columns = 17

#Create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a) 
#The array contains 11 rows and 11 columns (to match the shape of the environment), as well as a third "action" dimension.
#The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in
#each state (see next cell for a description of possible actions). 
#The value of each (state, action) pair is initialized to 0.
q_values = np.zeros((environment_rows, environment_columns, 4))
q_values = np.zeros((agent_x, agent_y, coin_x, coin_y, actions))
actions = ['up', 'right', 'down', 'left']
rewards = np.full((environment_rows, environment_columns), -100.)

#define free locations
aisles = {} #store locations in a dictionary
for i in range(1,8):
   aisles[2*i-1] = [i for i in range(1,16)] #3,5,7,9,11,13,15
   aisles[2*i] = [1,3,5,7,9,11,13,15] #2,4,6,8,10,12,14,16

aisles[15] = [i for i in range(1,16)]


#set the rewards for all aisle locations (i.e., white squares)
for row_index in range(1,16):
  for column_index in aisles[row_index]:
    rewards[row_index, column_index] = -1.

#set the reward (coin location) to 100
x, y = random.choice(list(aisles.items()))
rewards[x,random.choice(y)] = 100
print(x,random.choice(y))

for row in rewards:
  print(row)

#helper functions
def is_terminal_state(current_row_index, current_column_index):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True

def get_starting_location():
  #get a random row and column index
  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
  #continue choosing random row and column indexes until a non-terminal state is identified
  #(i.e., until the chosen state is a 'white square').
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index

def get_next_action(current_row_index, current_column_index, epsilon):
  #if a randomly chosen value between 0 and 1 is less than epsilon, 
  #then choose the most promising value from the Q-table for this state.
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index

def get_shortest_path(start_row_index, start_column_index):
  #return immediately if this is an invalid starting location
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: #if this is a 'legal' starting location
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_row_index, current_column_index):
      #get the best action to take
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      #move to the next location on the path, and add the new location to the list
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path  

epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn

#run through 1000 training episodes
for episode in range(1000):
  #get the starting location for this episode
  row_index, column_index = get_starting_location()

  #continue taking actions (i.e., moving) until we reach a terminal state
  #(i.e., until we reach the item packaging area or crash into an item storage location)
  while not is_terminal_state(row_index, column_index):
    #choose which action to take (i.e., where to move next)
    action_index = get_next_action(row_index, column_index, epsilon)

    #perform the chosen action, and transition to the next state (i.e., move to the next location)
    old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
    row_index, column_index = get_next_location(row_index, column_index, action_index)
    
    #receive the reward for moving to the new state, and calculate the temporal difference
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    #update the Q-value for the previous state and action pair
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')  

print(get_shortest_path(3, 9))
"""
