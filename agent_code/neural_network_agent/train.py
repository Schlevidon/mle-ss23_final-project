from collections import namedtuple, deque
#from torch.utils.data import DataLoader
import random
from tqdm import tqdm, trange
import torch
import torch.optim as optim

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100_000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


ACTIONS_DICT = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5
}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # TODO: load optimizer values
    self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    events.append("ACTION")

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # TODO: Perform a training step here?


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    size = state_to_features(last_game_state).shape
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, torch.ones(size)*float('nan'), reward_from_events(self, events)))

    # Train the model
    train(self)

    # Store the model
    self.model.save()

    # Update EPS
    self.eps *= self.eps_decay
    self.logger.debug(f'EPS_value: {self.eps}')

def train(self):

    model = self.model.to(self.DEVICE)
    #print(model)
    optimizer = self.optimizer

    transitions = self.transitions
    #data = DataLoader(transitions, batch_size=self.batch_size, shuffle=True)

    random.shuffle(transitions)
    
    for i in range(0, len(transitions), self.batch_size):
        batch = list(transitions)[i:i + self.batch_size]
        train_step(self, batch)

def train_step(self, batch):
    # TODO: how to send tensors to GPU if available
    transitions = list(zip(*batch))
    actions = transitions[1]
    old_states = torch.vstack(transitions[0]).to(self.DEVICE)
    new_states = torch.vstack(transitions[2]).to(self.DEVICE)
    #print(old_states.shape, new_states.shape)
    reward = torch.tensor(transitions[3]).to(self.DEVICE)

    #mask_nan = ~np.isnan(new_states)
    #mask_nan = [np.isnan(state) for state in new_states]
    mask_nan = torch.any(np.isnan(new_states), 1)
    #new_states = new_states[mask_nan].reshape(int(np.sum(mask_nan) // new_states.shape[1]),-1)

    #terminal_states = torch.isnan(new_states).to(self.DEVICE)
    
    outputs = self.model(old_states) # batch_size x 6
    # -> [0.1, 0.3, 0.6], -> one-hot [0.1, 10, 0.6] 
    
    with torch.no_grad():
        #Q_values = self.model(new_states[~mask_nan])
        target = outputs.clone().to(self.DEVICE) # batch_size -> batch_size x 6
        # Q(s,a)
        # target:  R + gamma * max_a[ Q(s',a)- Q(s,a)] q + alpha (R+gamma * max()) \approx R(s,a,s')+ \gamma ' max(Q(s'))
        # TODO: maybe need to filter out invalid actions before taking max
        #Q_max = Q_values.max(1)[0]
        for a, r, Q_new, done, s in zip(actions, reward, target, mask_nan, new_states): # Transition: s, a -> s = 6a
            a_idx = ACTIONS_DICT[a]
            Q_new[a_idx] = r
            if not done:
                Q_new[a_idx] += self.gamma * torch.max(self.model(s)) #[0] ## for multi-dim input

        #target[mask_nan] += self.gamma * Q_values.max(1)[0]
    
    self.optimizer.zero_grad()
    
    loss = self.criterion(outputs, target)
    loss.backward()

    self.optimizer.step()
    self.logger.debug(f'Q_values: {Q_new}, Loss: {loss}')



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 300,
        #e.KILLED_OPPONENT: 5,
        "ACTION" : -1,
        e.BOMB_DROPPED : -5,
        e.KILLED_SELF : -500,
        e.SURVIVED_ROUND : 100
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
