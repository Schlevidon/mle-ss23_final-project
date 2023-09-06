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
from .globals import Transition, TRANSITION_HISTORY_SIZE, OPTIMIZER_PARAMS, EPS_START, EPS_DECAY, BATCH_SIZE, AVERAGE_REWARD_WINDOW, ACTIONS

def setup_training(self):

    # Setup transition history
    # TODO : Write a class for ExperienceBuffer?
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # TODO : load previous optimizer values
    # TODO : try different optimizers
    #self.optimizer = optim.Adam(self.model.parameters(), **OPTIMIZER_PARAMS)

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

    
    """
    # Distance to coin
    current_distance = self.MODEL_TYPE.state_to_features(new_game_state, self)[-1]
    if self.last_distance < current_distance:
        events.append(e.DISTANCE_MAX)
    if self.last_distance > current_distance:
        events.append(e.DISTANCE_MIN)

    self.last_distance = current_distance
    """
    # Reward ideal action
    if self_action == ACTIONS[self.MODEL_TYPE.state_to_features(old_game_state, self)]:
        events.append(e.IDEAL_ACTION)


    # Update total reward from round
    reward = reward_from_events(self, events)
    self.round_reward += reward

    # state_to_features is defined in model.py
    self.transitions.append(Transition(self.MODEL_TYPE.state_to_features(old_game_state, self),
                                        self_action, 
                                        self.MODEL_TYPE.state_to_features(new_game_state, self), 
                                        reward,
                                        new_game_state))
    
    # Train on one random batch
    # TODO : implement importance sampling
    """
    if len(self.transitions) < BATCH_SIZE:
        sample = self.transitions
    else:
        sample = random.sample(self.transitions, BATCH_SIZE)
    """
    

    sample = self.transitions[-1]
    self.model.train_step(self, sample)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    size = self.MODEL_TYPE.state_to_features(last_game_state, self).shape
    self.transitions.append(Transition(self.MODEL_TYPE.state_to_features(last_game_state, self), 
                                       last_action, 
                                       torch.ones(size)*float('nan'), # TODO : find a better solution (this is memory inefficient and complicated)
                                       reward_from_events(self, events),
                                       None))

    # TODO : Train the model for longer at the end of a round?
    #train(self)
    
    # Plot metrics
    self.eps_history.append(self.eps)

    self.round_reward_history.append(self.round_reward)
    if len(self.round_reward_history) < AVERAGE_REWARD_WINDOW:
        self.mean_round_reward_history.append(np.mean(self.round_reward_history)) 
    else:
        self.mean_round_reward_history.append(np.mean(self.round_reward_history[-AVERAGE_REWARD_WINDOW:])) 
    plot(self.round_reward_history, self.mean_round_reward_history, self.eps_history)

    # Reset metrics
    self.round_reward = 0

    # Store the model
    # TODO: only update if the new model is better?
    # TODO: or create multiple checkpoints?
    self.model.save()

    # Update EPS
    # TODO : write a scheduler?
    self.eps *= EPS_DECAY
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
        #e.COIN_COLLECTED: 10,
        #e.OPPONENT_ELIMINATED: 1000,
        #e.ANY_ACTION : -1,
        #e.LOOP : -50,
        #e.INVALID_ACTION : -5,
        #e.BOMB_DROPPED : -10,
        #e.KILLED_SELF : -300,
        #e.DISTANCE_MIN : +1,
        #e.DISTANCE_MAX : -1,
        #e.SURVIVED_ROUND : 100,
        e.IDEAL_ACTION : 1,
        #e.CRATE_DESTROYED : 5,
    } 
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
