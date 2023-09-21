from collections import deque
from typing import List
import pickle
import random

from tqdm import tqdm

import torch

import numpy as np

import events as e
from .features import get_safety_feature, enemy_in_blast_coords
from .globals import Transition, TRANSITION_HISTORY_SIZE, EPS_START, EPS_DECAY, BATCH_SIZE, AVERAGE_REWARD_WINDOW, ACTIONS, ACTIONS_DICT, NSTEPS

def setup_training(self):

    # Setup transition history
    # TODO : Write a class for ExperienceBuffer?
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # TODO : load previous optimizer values
    # TODO : try different optimizers

    # Initialize epsilon for training
    # TODO : load previous epsilon when resuming training?
    self.eps = EPS_START



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    # Add custom events
    new_events = get_new_events(self, old_game_state, self_action, new_game_state)
    events.extend(new_events)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)

    # state_to_features is defined in model.py
    old_state_feature = self.MODEL_TYPE.state_to_features(old_game_state, self)
    self.transitions.append(Transition(old_state_feature,
                                        self_action, 
                                        self.MODEL_TYPE.state_to_features(new_game_state, self), 
                                        reward,
                                        new_game_state))
    
    # update stats
    self.stats.update_step(old_game_state, self_action, new_game_state, events, reward, old_state_feature, self)

    # train on the last NSTEPS transition for a SARSA step
    if old_game_state['step'] >=NSTEPS:
        sample = []
        for i in np.arange(1, NSTEPS+1, 1)[::-1]:
            sample.append(self.transitions[-i])
        self.model.train_step(self, sample)
    '''
    if old_game_state["step"] != 1:
        sample = [self.transitions[-2], self.transitions[-1]]
        self.model.train_step(self, sample)
     '''
    self.logger.debug('_'*20)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # Add custom events
    new_events = get_new_events(self, last_game_state, last_action, None)
    events.extend(new_events)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)
    
    size = self.MODEL_TYPE.state_to_features(last_game_state, self).shape

    last_game_feature = self.MODEL_TYPE.state_to_features(last_game_state, self)
    self.transitions.append(Transition(last_game_feature, 
                                       last_action, 
                                       torch.ones(size) *float('nan'), # TODO : find a better solution (this is memory inefficient and complicated)
                                       reward,
                                       None))
    # update stats
    self.stats.update_end_of_round(last_game_state, last_action, events, reward, last_game_feature, self)


    # train on the last NSTEPS transition for a SARSA step
    # like in game_event_occured
    if last_game_state['step']  < NSTEPS:
        NSTEPS_temp = last_game_state['step']
        sample = []
        for i in np.arange(1, NSTEPS_temp+1, 1)[::-1]:
            sample.append(self.transitions[-i])
            #sample = [state[-5], state[-4], state[-3], state[-2], state[-1], None]
    else:
        sample = []
        for i in np.arange(1, NSTEPS+1, 1)[::-1]:
            sample.append(self.transitions[-i])
    sample.append(None)
    self.model.train_step(self, sample)    


    # Store the model
    # TODO: only update if the new model is better?
    # TODO: or create multiple checkpoints?
    self.model.save()

    # Store global model
    self.model.save(folder_path='../../global_model') # to prevent from overwriting the qtable

    # save the stats
    self.stats.save()

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

def get_new_events(self, old_game_state: dict, self_action: str, new_game_state: dict = None):
    # Define new events
    # TODO : Remove for agent submission (or maybe not if train is never called)
    events = []
    events.append(e.ANY_ACTION)

    # If agent has been in the same location two times recently, it's a loop
    '''    
    try:
        agent_pos = new_game_state['self'][-1]
        if self.coordinate_history.count(agent_pos) > 2:
            events.append(e.LOOP)
        self.coordinate_history.append(agent_pos)
    except:
        self.logger.debug(f'Position of agent not found: no new_game_state')
        '''

    # TODO : think of a smarte way to do this
    # Agent placed a bomb next to a crate
    if self_action == "BOMB":
        my_x, my_y = old_game_state["self"][-1]
        adjacent_fields = [(my_x, my_y - 1), (my_x + 1, my_y), (my_x, my_y + 1), (my_x - 1, my_y)]
        for coord in adjacent_fields:
            if old_game_state["field"][coord] == 1:
                events.append(e.BOMB_NEXT_TO_CRATE)
                break

    # Placed a bomb near an enemy
    my_pos = old_game_state["self"][-1]
    other_agents = old_game_state["others"]
    other_pos = [agent[-1] for agent in other_agents]

    field = old_game_state["field"]
    
    if self_action == "BOMB" and enemy_in_blast_coords(my_pos, other_pos, field):
        events.append(e.BOMB_NEAR_ENEMY)


    # Took an unsafe action
    pos_agent = old_game_state["self"][-1]
    field = old_game_state["field"]
    explosion_map = old_game_state["explosion_map"]
    bombs = old_game_state["bombs"]

    """unsafe_actions = get_safety_feature(pos_agent, field, explosion_map, bombs)
    a_idx = ACTIONS_DICT[self_action]
    if unsafe_actions[a_idNSAFEx] == 0:
        e.append(e.UNSAFE_ACTION)
    """
    safe_actions = ACTIONS[get_safety_feature(pos_agent, field, explosion_map, bombs).bool()]
    if self_action not in safe_actions and len(safe_actions) != 0:
        events.append(e.UNSAFE_ACTION)

    return events


def reward_from_events(self, events: List[str]) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 100,
        #e.OPPONENT_ELIMINATED: 1000,
        e.ANY_ACTION : -1,
        #e.LOOP : -50,
        #e.INVALID_ACTION : -100,
        #e.BOMB_DROPPED : -10,
        #e.KILLED_SELF : -250,
        #e.GOT_KILLED : -500,
        #e.KILLED_OPPONENT : 500,
        #e.DISTANCE_MIN : +1,
        #e.DISTANCE_MAX : -1,
        #e.SURVIVED_ROUND : 100,
        #e.IDEAL_ACTION : 1,
        #e.CRATE_DESTROYED : 1/3 * 100,
        e.BOMB_NEXT_TO_CRATE : 1/3 * 100,
        e.BOMB_NEAR_ENEMY : 1/10 * 500,
        e.UNSAFE_ACTION : -250,
    } 
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
