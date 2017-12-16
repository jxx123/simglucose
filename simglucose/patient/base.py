"""Base class for patient"""


class Patient(object):
    def step(self, action):
        """
        Run one time step of the patient dynamics
        ------
        Input
            action: a namedtuple
        ------
        Outputs
            t: current time
            state: updated state
            observation: the observable states
        """
        raise NotImplementedError

    @staticmethod
    def model(t, state, action, params):
        """
        ordinary differential equations
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset to the initial state
        Return observation
        """
        raise NotImplementedError
