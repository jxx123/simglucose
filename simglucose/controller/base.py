from collections import namedtuple

Action = namedtuple('ctrller_action', ['basal', 'bolus'])


class Controller(object):
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state

    def policy(self, observation, reward, done, **info):
        '''
        Every controller must have this implementation!
        ----
        Inputs:
        observation - a namedtuple defined in simglucose.simulation.env. It has
                      CHO and CGM two entries.
        reward      - current reward returned by environment
        done        - True, game over. False, game continues
        info        - additional information as key word arguments,
                      simglucose.simulation.env.T1DSimEnv returns patient_name
                      and sample_time
        ----
        Output:
        action - a namedtuple defined at the beginning of this file. The
                 controller action contains two entries: basal, bolus
        '''
        raise NotImplementedError

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        raise NotImplementedError
