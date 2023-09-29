import matplotlib
import matplotlib.pyplot as plt
#from IPython import display

import numpy as np
import torch

from typing import List
from collections import deque
import pickle
from datetime import datetime
from pathlib import Path

ACTIONS = np.array(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'])
ACTIONS_DICT = {action : idx for idx, action in enumerate(ACTIONS)}

class Stats:
    def __init__(self, table_shape=None):
        self.dir = Path("stats") 
        self.file = Path(datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + ".pkl")

        # Hyperparameters
        self.EPS_START = 0
        self.EPS_END = 0
        self.EPS_DECAY = 0

        self.GAMMA = 0
        
        self.LR = 0

        # Round reward history
        self.round_reward = 0
        self.round_reward_history = []
        
        # Actual score according to the game
        self.round_score_history = []

        # Length of each round
        self.round_length_history = []

        # Number of times a table entry was visited
        if table_shape is None:
            self.table_exploration = []
        else:
            self.table_exploration = torch.zeros(table_shape)

        # Number of times events occured
        self.event_counter = {}

        # Last eps
        self.last_eps = 0

    def save(self, path=None):
        self.dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            path = self.dir / self.file
        
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load(self, path):
        with open(path, "rb") as file:
            self = pickle.load(file)

    def update_step(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str], reward, state_feature=None, agent=None):
        # Round rewards
        self.round_reward += reward

        # Update event stats
        for event in events:
            if event in self.event_counter:
                self.event_counter[event] += 1
            else:
                self.event_counter[event] = 0

        # Update exploration stats
        a_idx = ACTIONS_DICT[self_action]
        self.table_exploration[tuple(state_feature)][a_idx] += 1

        # Eps decay
        self.last_eps = agent.eps
    
    def update_end_of_round(self, last_game_state: dict, last_action: str, events: List[str], reward, state_feature=None, agent=None):
        # Round rewards
        self.round_reward += reward
        self.round_reward_history.append(self.round_reward)
        self.round_reward = 0 # reset for next round

        # Update event stats
        for event in events:
            if event in self.event_counter:
                self.event_counter[event] += 1
            else:
                self.event_counter[event] = 0

        # Update exploration stats
        a_idx = ACTIONS_DICT[last_action]
        self.table_exploration[tuple(state_feature)][a_idx] += 1

        # Length of episode
        self.round_length_history.append(last_game_state["step"])

        # Actual score of episode
        self.round_score_history.append(last_game_state["self"][1])

        # Eps decay
        self.last_eps = agent.eps

#plt.ion()
#fig, (score_ax, eps_ax) = plt.subplots(2, sharex=True)
'''def plot(scores, mean_scores, eps):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    score_ax.set_title('Training scores')
    #score_ax.set_xlabel('Number of Games')
    score_ax.set_ylabel('Score')
    score_ax.plot(scores, label='score')
    score_ax.plot(mean_scores, label='average score')
    score_ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    score_ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    eps_ax.set_title('Training eps')
    eps_ax.set_xlabel('Number of ')
    eps_ax.set_ylabel('Epsilon')
    eps_ax.plot(eps)
    eps_ax.text(len(eps)-1, eps[-1], str(eps[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)'''

def plot(scores, mean_scores, eps, events_dict):
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    
    plt.clf()
    
    # Plotting Score
    ax1 = plt.subplot(311)
    plt.title('Training...')
    #plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='average score')
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(round(scores[-1],2)))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1],2)))
    plt.legend()

    # Plotting Epsilon
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(eps)
    plt.ylabel('Epsilon')
    plt.xlabel('Number of Games')
    plt.text(len(eps)-1, eps[-1], str(eps[-1]))
    
    # Plotting event occurence
    ax3 = plt.subplot(313)
    labels, values = events_dict.keys(), events_dict.values()
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.ylabel('Events occured')
    plt.xlabel('Event')
    #plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().tight_layout()


    plt.show(block=False)
    plt.pause(.1)

'''
def plot(scores, mean_scores, eps):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='average score')
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
    '''


def tile_is_free(game_state, x, y):
    is_free = (game_state['field'][x, y] == 0)
    if is_free:
        for obstacle in game_state['bombs']: 
            is_free = is_free and (obstacle[0][0] != x or obstacle[0][1] != y)
        for obstacle in game_state['others']: 
            is_free = is_free and (obstacle[3][0] != x or obstacle[3][1] != y)
    return is_free

def get_valid_actions(game_state) -> np.array:
    ''' round = game_state['round']
        step = game_state['step']
        field = game_state['field']
        bombs = game_state['bombs']
        explosion_map = game_state['explosion_map']
        coins = game_state['coins']
        my_agent = game_state['self']
        others = game_state['others']'''
    # TODO : extract state once for all tiles to improve performance?
    agent_x, agent_y = game_state['self'][3]
    up = tile_is_free(game_state, agent_x, agent_y - 1)
    down = tile_is_free(game_state, agent_x, agent_y + 1)
    left =  tile_is_free(game_state, agent_x - 1, agent_y)
    right =  tile_is_free(game_state, agent_x + 1, agent_y)
    wait = True
    bomb = game_state["self"][2]
    #bomb = False # disable bombs for now
    #return np.array([True, True, True, True, True, True])
    return np.array([up, right, down, left, wait, bomb])


def move_repeated(self, game_state, valid_actions_mask):
    history = self.coordinate_history
    self_pos = game_state['self'][-1]
    
    if history.count(self_pos)>2:

        last_pos = history[-1]
        action_to_last_pos = get_last_step(self_pos, last_pos)

        valid_action = [action for action in ACTIONS[valid_actions_mask] if action!='BOMB'and action!=action_to_last_pos]
        self.coordinate_history.append(self_pos)
        if len(valid_action)==0:
            return None
        selected_action = np.random.choice(valid_action)
        #self.coordinate_history = deque([], 10)
        return selected_action
    else:
        self.coordinate_history.append(self_pos)
        return None
    
def get_last_step(my_pos, last_pos):
    my_x_old, my_y_old = last_pos
    my_x, my_y = my_pos
    if my_y_old < my_y:
        return "UP"
    if my_y_old > my_y:
        return "DOWN"
    if my_x_old < my_x:
        return "LEFT"
    if my_x_old > my_x:
        return "RIGHT"
    return 'WAIT'