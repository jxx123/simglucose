from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P=1, I=0, D=0, target=140):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time')

        # BG is the only state for this PID controller
        bg = observation.CGM
        control_input = self.P * (bg - self.target) + \
            self.I * self.integrated_state + \
            self.D * (bg - self.prev_state) / sample_time

        logger.info('Control input: {}'.format(control_input))

        # update the states
        self.prev_state = bg
        self.integrated_state += (bg - self.target) * sample_time
        logger.info('prev state: {}'.format(self.prev_state))
        logger.info('integrated state: {}'.format(self.integrated_state))

        # return the action
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0
