import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


def setup(self):
    np.random.seed()

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
    print(move)
    return move


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random')
    #return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    #return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.25, .25, .25, .25, .00])
    pos_agent, pos_coin, field= game_state['self'][3], game_state['coins'], game_state['field']

    return find_ideal_path(pos_agent, pos_coin, field)


#python main.py play --scenario coin-heaven --my-agent coin_learning_agent



