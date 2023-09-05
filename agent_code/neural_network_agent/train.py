from collections import namedtuple, deque
from typing import List
import pickle
import random

from tqdm import tqdm, trange

import torch
import torch.optim as optim

import numpy as np

import events as e
from .helper import plot
from .callbacks import MODEL_TYPE


# This is only an example!
# Save next_state_dict to determine valid actions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_state_dict'))

# Parameters
TRANSITION_HISTORY_SIZE = 100_000 

OPTIMIZER_PARAMS = {
    "lr" : 1e-3,
    "eps" : 1e-9,
    "betas" : [0.9, 0.98]
}
CRITERION = torch.nn.MSELoss()

BATCH_SIZE = 16

# epsilon for epsilon-greedy policy
EPS_START = 0.1
EPS_END = 0.001
EPS_DECAY = 0.999

# Reward discount factor
GAMMA = 0.99

def setup_training(self):

    # Setup transition history
    # TODO : Write a class for ExperienceBuffer?
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # TODO : load previous optimizer values
    # TODO : try different optimizers
    self.optimizer = optim.Adam(self.model.parameters(), **OPTIMIZER_PARAMS)

    # Initialize epsilon for training
    # TODO : load previous epsilon when resuming training?
    self.eps = EPS_START


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    # TODO : Remove for agent submission (or maybe not if train is never called)
    events.append(e.ANY_ACTION)

    # If agent has been in the same location two times recently, it's a loop
    try:
        agent_pos = new_game_state['self'][-1]
        if self.coordinate_history.count(agent_pos) > 2:
            events.append(e.LOOP)
        self.coordinate_history.append(agent_pos)
    except:
        self.logger.debug(f'Position of agent not found: no new_game_state')

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Update total reward from round
    reward = reward_from_events(self, events)
    self.round_reward += reward

    # state_to_features is defined in model.py
    self.transitions.append(Transition(MODEL_TYPE.state_to_features(old_game_state),
                                        self_action, 
                                        MODEL_TYPE.state_to_features(new_game_state), 
                                        reward,
                                        new_game_state))
    
    # Train on one random batch
    # TODO : implement importance sampling
    batch = random.sample(self.transitions)
    self.model.train_step(batch, self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    size = MODEL_TYPE.state_to_features(last_game_state).shape
    self.transitions.append(Transition(MODEL_TYPE.state_to_features(last_game_state), 
                                       last_action, 
                                       torch.ones(size)*float('nan'), # TODO : find a better solution (this is memory inefficient and complicated)
                                       reward_from_events(self, events),
                                       None))

    # TODO : Train the model for longer at the end of a round?
    #train(self)
    
    # Plot metrics
    self.eps_history.append(self.eps)

    self.round_reward_history.append(self.round_reward)
    if len(self.round_reward_history) < 50: # TODO : move this parameter
        self.mean_round_reward_history.append(np.mean(self.round_reward_history)) 
    else:
        self.mean_round_reward_history.append(np.mean(self.round_reward_history[-50:])) 
    plot(self.round_reward_history, self.mean_round_reward_history, self.eps_history)

    # Reset metrics
    self.round_reward = 0

    # Store the model
    # TODO: only update if the new model is better?
    # TODO: or create multiple checkpoints?
    self.model.save()

    # Update EPS
    # TODO : write a scheduler?
    self.eps *= self.eps_decay
    self.logger.debug(f'EPS_value: {self.eps}')
"""
def train(self):
    #Train for a whole epoch
    optimizer = self.optimizer

    transitions = self.transitions

    random.shuffle(transitions)
    
    for i in range(0, len(transitions), self.batch_size):
        batch = list(transitions)[i:i + self.batch_size]
        train_step(self, batch)
"""

def reward_from_events(self, events: List[str]) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.OPPONENT_ELIMINATED: 1000,
        e.ANY_ACTION : -1,
        e.LOOP : -50,
        e.INVALID_ACTION : -5,
        e.BOMB_DROPPED : -10,
        e.KILLED_SELF : -300
        #e.SURVIVED_ROUND : 100
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
