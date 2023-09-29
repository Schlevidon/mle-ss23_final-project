import numpy as np


def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    agent.logger.info('Waits the whole time')
    return 'WAIT'
