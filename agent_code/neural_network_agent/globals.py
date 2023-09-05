from collections import namedtuple

import numpy as np
import torch

from . import model as m

ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
ACTIONS_DICT = {action : idx for idx, action in enumerate(ACTIONS)}

# Set pytorch device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: Add a way to load the selected scenario to determine the number of available coins

# Use rule based agent?
ALWAYS_RB = False
SAMPLE_RB = False

# Stochastic policy?
STOCHASTIC_POLICY = True

# TODO : Set seed to make results reproducible?
RANDOM_SEED = None

# Parameters
TRANSITION_HISTORY_SIZE = 100_000 

# Save next_state_dict to determine valid actions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_state_dict'))

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
