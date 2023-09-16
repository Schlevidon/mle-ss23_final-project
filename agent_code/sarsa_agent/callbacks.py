import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim

import os
import random
from collections import deque

from . import model as m
from . import callbacks_rb as crb
from .helper import get_valid_actions, Stats
from .globals import RANDOM_SEED, DEVICE, ALWAYS_RB, SAMPLE_RB, ACTIONS, STOCHASTIC_POLICY, MULTIPLE_AGENTS

import settings as s

def setup(self):
    np.random.seed(RANDOM_SEED)
    
    self.PATH = "./model/my-model.pt" #'/'.join((MODEL_FOLDER,MODEL_NAME))

    self.MODEL_TYPE = m.SarsaTable
    # Select model type here
    architecture = self.MODEL_TYPE.get_architecture()
    self.model = self.MODEL_TYPE(**architecture)
    '''    
    if os.path.isfile(self.PATH):
        # TODO: Disable dropout and batch norm
        self.model.load_state_dict(torch.load(self.PATH, map_location = torch.device(DEVICE)))
    else:
        # TODO : should we move the model to device here?
        self.model.to(DEVICE)
    '''

    self.stats = Stats(architecture["dimensions"])
    # TODO : could load old stats, but seems unneccessary for now

    if os.path.isfile(self.PATH):
        self.model.table = torch.load(self.PATH)
    
    # Initialize eps
    self.eps = 0 # if not training eps should be 0

    # For rule-based agent only
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0

    self.last_distance = 20
    
def act(self, game_state: dict) -> str:
    """state_feature = self.MODEL_TYPE.state_to_features(game_state, self).flatten().bool()
    explosion_map = game_state["explosion_map"]
    self.logger.debug("Computing state features...")
    self.logger.debug(f"Coin direction: {ACTIONS[:4][state_feature[:4]]}")
    self.logger.debug(f"Crate direction: {ACTIONS[:4][state_feature[4:8]]}")
    self.logger.debug(f"Agent next to crate: {state_feature[8]}")
    self.logger.debug(f"Safety direction: {ACTIONS[:5][state_feature[9:14].bool()]}")"""

    """state_feature = self.MODEL_TYPE.state_to_features(game_state, self).flatten()
    self.logger.debug(f"Coin or crate direction: {ACTIONS[:5][state_feature[0]]}")
    self.logger.debug(f"Safety direction: {ACTIONS[:5][state_feature[1:6].bool()]}")
    self.logger.debug(f"Agent next to crate: {bool(state_feature[6])}")"""
    
    #self.model.table = torch.load(self.PATH) # for training multiple agents on the same table simultaniously

    state_feature = self.MODEL_TYPE.state_to_features(game_state, self).flatten()
    self.logger.debug(f"Coin or crate direction: {ACTIONS[:5][state_feature[0]]}")
    self.logger.debug(f"Enemy direction: {ACTIONS[:5][state_feature[1]]}")
    self.logger.debug(f"Safety direction: {ACTIONS[state_feature[2:8].bool()]}")
    self.logger.debug(f"Agent next to crate: {bool(state_feature[8])}")
    self.logger.debug(f"Enemy agent in blast radius: {bool(state_feature[9])}")

    if self.train and MULTIPLE_AGENTS:
        self.logger.debug(f"Training with multiple agents")
        self.model.table = game_state['table']

    if self.train and ALWAYS_RB:
        self.logger.debug(f"Forced to choose rule-based agent action.")
        return crb.act(self, game_state)

    valid_actions_mask = get_valid_actions(game_state)
    self.logger.debug(f'Valid actions: {ACTIONS[valid_actions_mask]}')
    
    if self.train and random.random() < self.eps:
        # Exploratory move
        self.logger.debug(f"Chose exploratory move with probability {self.eps:.2%}")
        if SAMPLE_RB:
            self.logger.debug(f"Sampling from rule-based agent")
            selected_action = crb.act(self, game_state)
        else:
            self.logger.debug("Choosing random valid action")
            selected_action = np.random.choice(ACTIONS[valid_actions_mask])
    else:
        # Exploitation: act greedily wrt to Q-function
        self.logger.debug("Querying model for action (greedy)")
        with torch.no_grad():
            Q_values = self.model(self.MODEL_TYPE.state_to_features(game_state, self))
            self.logger.debug(f"Q Values (rounded): {[(a, round(float(Q), 5)) for a, Q in zip(ACTIONS, Q_values)]}")

        if STOCHASTIC_POLICY:
            probs = np.array(softmax(Q_values[valid_actions_mask], dim=0))
            self.logger.debug(f"Action probabilities: {probs}")

            selected_action = np.random.choice(ACTIONS[valid_actions_mask], p=probs)
        else:
            max_Q = torch.max(Q_values[valid_actions_mask])
            self.logger.debug(f"Maximum Q-value {max_Q}")
            mask = np.isclose(Q_values, max_Q) & valid_actions_mask
            #self.logger.debug(f"max mask: {max_mask}")
            #self.logger.debug(f"final mask: {mask}")

            best_actions = ACTIONS[mask]
            self.logger.debug(f"Best actions: {best_actions}")

            selected_action = np.random.choice(best_actions)
    
    self.logger.debug(f"Selected action: {selected_action}")
    
    return selected_action


    


