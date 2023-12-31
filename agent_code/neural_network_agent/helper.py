import matplotlib.pyplot as plt
from IPython import display

import numpy as np

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


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

def plot(scores, mean_scores, eps):
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.clf()
    ax1 = plt.subplot(211)
    plt.title('Training...')
    #plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='average score')
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(eps)
    plt.ylabel('Epsilon')
    plt.xlabel('Number of Games')
    plt.text(len(eps)-1, eps[-1], str(eps[-1]))
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

    return np.array([up, right, down, left, wait, bomb])

def find_ideal_path(pos_agent, pos_coin, field=None, bombs=None, explosion_map=None):
    field[field==1] = 2
    field[field==0] = 1
    grid = Grid(matrix=field)
    finder = AStarFinder()

    sx, sy = pos_agent
    start = grid.node(sx, sy)

    lengths = []
    for coin in pos_coin:
        cx, cy = coin
        end = grid.node(cx,cy)
        path, runs = finder.find_path(start, end, grid)
        grid.cleanup()
        lengths.append((len(path),path))

    grid.cleanup()
    lengths = sorted(lengths,key=lambda c : c[0])

    try:
        step0 = lengths[0][1][0]
        step1 = lengths[0][1][1]
    except:
        return 'WAIT'

    diff = np.array([step1.x,step1.y]) - np.array([step0.x,step0.y])

    if diff[0]==0:
        if diff[1]==1:
            move = 'DOWN'
        else:
            move = 'UP'

    elif diff[0]==1:
        move = 'RIGHT'
    else:
        move = 'LEFT'
    return move