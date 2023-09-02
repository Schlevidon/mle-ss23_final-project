import settings as s
from . import model as m

import numpy as np

from torch import nn

# Define possible actions and dict for quick index access
ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']) 
ACTIONS_DICT = {action : idx for idx, action in enumerate(ACTIONS)}

# Simplest dummy state to determine feature dimension
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

# Network architecture
# TODO: maybe put this inside the model
MODEL_CLASS = m.QNetwork

# Linear net 
if MODEL_CLASS == m.QNetwork:
    INPUT_DIM = len(m.state_to_features(DUMMY_STATE)) # For a linear model
    OUTPUT_DIM = len(ACTIONS)
    DIMENSIONS = [INPUT_DIM, 300, 100, 50, 20, OUTPUT_DIM] #[867, 20, 6]
    ACTIVATION = nn.ReLU
else:
    raise NotImplementedError(f"Model architecture not defined for model {MODEL_CLASS}")

