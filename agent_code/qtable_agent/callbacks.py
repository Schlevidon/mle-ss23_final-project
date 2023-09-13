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
from .helper import get_valid_actions
from .globals import RANDOM_SEED, DEVICE, ALWAYS_RB, SAMPLE_RB, ACTIONS, STOCHASTIC_POLICY

import settings as s

def setup(self):
    np.random.seed(RANDOM_SEED)
    
    self.PATH = "./model/my-model.pt" #'/'.join((MODEL_FOLDER,MODEL_NAME))

    self.MODEL_TYPE = m.QTable
    # Select model type here
    self.model = self.MODEL_TYPE(**self.MODEL_TYPE.get_architecture())
    '''    
    if os.path.isfile(self.PATH):
        # TODO: Disable dropout and batch norm
        self.model.load_state_dict(torch.load(self.PATH, map_location = torch.device(DEVICE)))
    else:
        # TODO : should we move the model to device here?
        self.model.to(DEVICE)
    '''
    if os.path.isfile(self.PATH):
        self.model.table = torch.load(self.PATH)
    
    # Initialize eps
    self.eps = 0 # if not training eps should be 0

    # Collecting metrics
    self.round_reward = 0
    self.round_score_history = []

    self.round_reward_history = []
    self.mean_round_reward_history = []
    
    self.event_counter = {}

    # TODO : save/display immediate rewards in each step
    self.step_reward_history = []

    self.eps_history = []

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
            self.logger.debug(f"Q Values: {Q_values}")

        if STOCHASTIC_POLICY:
            probs = np.array(softmax(Q_values[valid_actions_mask], dim=0))
            self.logger.debug(f"Action probabilities: {probs}")

            selected_action = np.random.choice(ACTIONS[valid_actions_mask], p=probs)
        else:
            max_Q = torch.max(Q_values[valid_actions_mask])
            self.logger.debug(f"Maximum Q-value {max_Q}")
            mask = np.array((max_mask := np.isclose(Q_values, max_Q))) & valid_actions_mask
            self.logger.debug(f"max mask: {max_mask}")
            self.logger.debug(f"final mask: {mask}")

            best_actions = ACTIONS[mask]
            self.logger.debug(f"Best actions: {best_actions}")

            selected_action = np.random.choice(best_actions)
    
    self.logger.debug(f"Selected action: {selected_action}")
    
    return selected_action


    


