import matplotlib.pyplot as plt
from IPython import display

import torch
import numpy as np

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from settings import BOMB_POWER


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
    return np.array([True, True, True, True, True, True])
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

def get_blast_coords(bombs, arena):
        power = 3
        blast_coords = []
        for bomb in bombs:
            #if bomb[1] > 0: continue 
            x, y = bomb[0]
            blast_coords.append((x, y))
            for i in range(1, BOMB_POWER + 1):
                if arena[x + i, y] == -1:
                    break
                blast_coords.append((x + i, y))
            for i in range(1, BOMB_POWER + 1):
                if arena[x - i, y] == -1:
                    break
                blast_coords.append((x - i, y))
            for i in range(1, BOMB_POWER + 1):
                if arena[x, y + i] == -1:
                    break
                blast_coords.append((x, y + i))
            for i in range(1, BOMB_POWER + 1):
                if arena[x, y - i] == -1:
                    break
                blast_coords.append((x, y - i))

        return blast_coords

def get_safety_feature(pos_agent, field, explosion_map, bombs):
    #[1,1,2,3] #0 wand; #1 frei; #2 crate; #3 expl

    # Explore directions in greedy order?
    new_explosion_map, new_field, new_bombs = explosion_map_update(explosion_map, bombs, field)
    
    field_feature = get_field_feature(pos_agent, field, new_explosion_map)

    output = torch.tensor([0, 0, 0, 0, 0])
    for i, direction in enumerate(field_feature):
        if direction != 1: continue#
        # i=0 UP, i=1 Right, i=2 DOWN, i=3 LEFT, i=4 WAIT
        # Update agent position
        new_pos = pos_update(pos_agent, i)
        # Update field (crates destroyed)
        # Update explosion map (old explosions removed, new explosions added)
        output[i] = int(find_safe_tile(new_pos, new_field, new_explosion_map, new_bombs, 1))

    return output#bool[UP, RIGHT, DOWN, LEFT, WAIT]


def find_safe_tile(pos_agent, field, explosion_map, bombs, depth):
    #[1,1,2,3] #0 wand; #1 frei; #2 crate; #3 expl
    # TODO : Limit depth for now
    if depth > 5:
        return False

    danger_positions = get_blast_coords(bombs, field)
    if pos_agent not in danger_positions:
        return True

    new_explosion_map, new_field, new_bombs = explosion_map_update(explosion_map, bombs, field)
    field_feature = get_field_feature(pos_agent, field, new_explosion_map)
    
    # TODO : Explore directions in greedy order?
    for i, direction in enumerate(field_feature):
        if direction != 1: continue

        # i=0 UP, i=1 Right, i=2 DOWN, i=3 LEFT, i=4 WAIT
        # Update agent position
        new_pos = pos_update(pos_agent, i)
        # Update field (crates destroyed)
        # Update explosion map (old explosions removed, new explosions added)
        if find_safe_tile(new_pos, new_field, new_explosion_map, new_bombs, depth + 1):
            return True

    return False

def pos_update(pos, index):
    if index==0:
        return (pos[0], pos[1] - 1) # UP
    if index==1:
        return (pos[0] + 1, pos[1]) # RIGHT
    if index==2:
        return (pos[0], pos[1] + 1) # DOWN
    if index==3:
        return (pos[0] - 1, pos[1]) # LEFT
    if index==4:
        return pos # WAIT

def explosion_map_update(explosion_map, bombs, field):
    # Update existing explosions timer
    explosion_map = explosion_map - 1
    explosion_map[explosion_map<0] = 0

    remaining_bombs = [(bomb[0], bomb[1] - 1) for bomb in bombs if bomb[1] > 0]
    exploding_bombs = [bomb for bomb in bombs if bomb[1] == 0]

    # Add new explosions from bombs
    bombs_exploded = tuple(np.array(get_blast_coords(exploding_bombs, field)).T)
    #for bomb in bombs_exploded:
    #    explosion_map[bomb] = 2
    if len(bombs_exploded) > 0:
        explosion_map[bombs_exploded] = 2
    active_explosions = explosion_map > 0 

    # Destroy crates
    field = field.copy() # Maybe not needed
    field[active_explosions & (field == 1)] = 0 # TODO: should be modified when other features included

    return explosion_map, field, remaining_bombs

def get_field_feature(my_pos, field, explosion_map):
    # TODO : predicted explosions is computed twice. It would be more efficient to reuse the computation from explosion_map_update
    my_x, my_y = my_pos

    field = field.copy()
    field[explosion_map > 0] = 2
    '''
    field_feature = torch.tensor([field[my_y - 1, my_x], #UP # field[my_y -1 , my_x] 
                                  field[my_y, my_x + 1], #RIGHT
                                  field[my_y + 1, my_x], #DOWN
                                  field[my_y, my_x - 1]]) #LEFT
    '''

    field_feature = torch.tensor([field[my_x, my_y - 1], #UP # field[my_y -1 , my_x] 
                                field[my_x + 1, my_y], #RIGHT
                                field[my_x, my_y + 1], #DOWN
                                field[my_x - 1, my_y], #LEFT
                                field[my_x, my_y]]) #WAIT
    field_feature += 1 # shift from [-1, 0, 1, 2] to [0, 1, 2,3]

    return field_feature
