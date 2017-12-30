from collections import namedtuple

Action = namedtuple('ctrller_action', ['basal', 'bolus'])


class Controller(object):
    def __init__(self, init_state):
        self.state = init_state

    def policy(self, observation, env):
        self.state = observation
        action = self.state[0]
        return action
