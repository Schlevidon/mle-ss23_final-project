import torch
import numpy as np

from scipy.spatial.distance import cdist

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from settings import BOMB_POWER

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


def get_coin_feature(my_pos, other_pos, pos_targets, field):
    # If there are no coins
    if len(pos_targets) == 0:
        return None

    pos_targets = np.array(pos_targets)
    coin_dists = np.squeeze(cdist([my_pos], pos_targets, metric="cityblock"))
    idx_sort = np.argsort(coin_dists)    

    # Calculate paths in order of shortest manhattan distance
    for t in pos_targets[idx_sort]:
        if (my_pos[0] == t[0]) and (my_pos[1] == t[1]): continue # Filter out rare case where coin in placed directly on agent

        path = find_path_to_target(my_pos, t, field)
        
        if len(path) == 0: continue # No direct path available

        enemy_agent_closer = False
        for pos in other_pos:
            other_path = find_path_to_target(pos, t, field)
            
            if len(path) > len(other_path):
                enemy_agent_closer = True
                break
        
        if enemy_agent_closer: continue
        # If there are no other agents who could reach the coin first, return
        # [(my_x, my_y), (my_x + 1, my_y), ...]
        return get_first_step_from_path(my_pos, path)
        
    return None


def get_first_step_from_path(my_pos, path):
    first_step = path[1]
    my_x, my_y = my_pos
    if first_step.y < my_y:
        return "UP"
    if first_step.y > my_y:
        return "DOWN"
    if first_step.x < my_x:
        return "LEFT"
    if first_step.x > my_x:
        return "RIGHT"

# TODO : For now bombs, explosions or other agent are not considered in path finding.
def find_path_to_target(my_pos, pos_target, field):
    field = field.copy()
    field[field == 1] = -1 # Crates are treated as walls
    field[field == 0] = 1 # Taking a step on a free tile has cost one
    
    grid = Grid(matrix=field.T) # Transpose field so that the output direction matches the GUI
    finder = AStarFinder()

    sx, sy = my_pos
    start = grid.node(sx, sy)

    tx, ty = pos_target
    end = grid.node(tx, ty)

    path, runs = finder.find_path(start, end, grid)
    grid.cleanup()

    return path # len(path) = 0 if no path can be found

def get_enemy_agent_feature(my_pos, pos_targets, field):
    # If there are no enemies return None
    if len(pos_targets) == 0:
        return None

    # Find agent with shortest manhattan distance
    # TODO : how to decide when there are multiple agents e.g. target where more agents are in proximity?
    dist = np.squeeze(cdist([my_pos], pos_targets, metric="cityblock"))
    #min_dist = np.min(dist)
    min_idx = np.argmin(dist) # For now arbitrarily target the first agent

    enemy_target = pos_targets[min_idx]

    # First check if there is a free path with crates 
    path = find_path_to_target(my_pos, enemy_target, field)

    # If no path exists, assume that there are no crates and try to get as close as possible
    if len(path) == 0:
        field = field.copy()
        field[field == 1] = 0
        path = find_path_to_target(my_pos, enemy_target, field)
    
    return get_first_step_from_path(my_pos, path)

def enemy_in_blast_coords(my_pos, other_pos, field):
    blast_coords = get_blast_coords([(my_pos, 3)], field)
    for pos in other_pos:
        if pos in blast_coords:
            return True

    return False

# TODO : Design a feature which allows the agent to make smarter attacks?
def enemy_trapped_by_bomb(my_pos, other_pos, field, explosion_map, bombs):
    bombs = bombs.copy()
    bombs.append((my_pos, 4))

    get_safety_feature(other_pos, field, explosion_map, bombs)

def get_blast_coords(bombs, arena):
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
    # TODO : Explore directions in greedy order?

    # Update field (crates destroyed) 
    # Update explosion map (old explosions removed, new explosions added)
    new_explosion_map, new_field, new_bombs = explosion_map_update(explosion_map, bombs, field)
    
    field_feature = get_allowed_actions(pos_agent, field, explosion_map, new_explosion_map, new_bombs)

    output = torch.tensor([0, 0, 0, 0, 0, 0])
    for i, direction in enumerate(field_feature):
        if not direction: continue
        # Update agent position
        new_pos = pos_update(pos_agent, i)
        output[i] = int(find_safe_tile(new_pos, new_field, new_explosion_map, new_bombs, 1))
    
    # Bomb safety
    new_pos = pos_agent
    new_bombs.append((new_pos, 3))
    # If we can't wait we also can't place bombs
    if output[-2]!=0:
        output[-1] = int(find_safe_tile(new_pos, new_field, new_explosion_map, new_bombs, 1))

    return output #bool[UP, RIGHT, DOWN, LEFT, WAIT, BOMB]


def find_safe_tile(pos_agent, field, explosion_map, bombs, depth):
    # TODO : Limit depth for now
    if depth > 5:
        return False

    danger_positions = get_blast_coords(bombs, field)
    if pos_agent not in danger_positions:
        return True

    # Update field (crates destroyed)
    # Update explosion map (old explosions removed, new explosions added)
    new_explosion_map, new_field, new_bombs = explosion_map_update(explosion_map, bombs, field)
    field_feature = get_allowed_actions(pos_agent, field, explosion_map, new_explosion_map, new_bombs)
    
    # TODO : Explore directions in greedy order?
    for i, direction in enumerate(field_feature):
        if not direction: continue
        # Update agent position
        new_pos = pos_update(pos_agent, i)

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
    explosion_map[explosion_map < 0] = 0

    remaining_bombs = [(bomb[0], bomb[1] - 1) for bomb in bombs if bomb[1] > 0]
    exploding_bombs = [bomb for bomb in bombs if bomb[1] == 0]

    # Add new explosions from bombs
    bombs_exploded = tuple(np.array(get_blast_coords(exploding_bombs, field)).T)
    #for bomb in bombs_exploded:
    #    explosion_map[bomb] = 2
    if len(bombs_exploded) > 0:
        explosion_map[bombs_exploded] = 1
    active_explosions = explosion_map > 0 

    # Destroy crates
    field = field.copy() # Maybe not needed
    field[active_explosions & (field == 1)] = 0 # TODO: should be modified when other features included

    return explosion_map, field, remaining_bombs

# TODO : how to handle other agents
def get_allowed_actions(my_pos, field, old_explosion_map, new_explosion_map, bombs):
    my_x, my_y = my_pos

    field = field.copy()
    field[(old_explosion_map + new_explosion_map) > 0] = 5

    bombs = [bomb[0] for bomb in bombs]
    # TODO : vectorize
    for b in bombs:
        field[b] = -1

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
                                field[my_x, my_y] == 5]) #WAIT (only forbidden if explosion)

    return (field_feature == 0)