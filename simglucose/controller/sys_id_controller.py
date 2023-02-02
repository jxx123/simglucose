from re import L
from .base import Controller
from .base import Action
from simglucose.utils import fetch_patient_params


class SysIDController(Controller):
    """
    This controller is just used to collect data for system identification.
    It tries to stimulate the system in a proper way to get useful information of the system.
    """
    def __init__(self, insulin_time, insulin_amount):
        assert len(insulin_time) == len(insulin_amount)
        self.insulin_time = insulin_time
        self.insulin_amount = insulin_amount
        self.basal = None

    def policy(self, observation, reward, done, **info):
        time = info['time']
        if not self.basal:
            patient_params = fetch_patient_params(info['patient_name'])
            u2ss = patient_params['u2ss']
            BW = patient_params['BW']
            self.basal = u2ss * BW / 6000  # unit: U/min

        action = Action(basal=self.basal, bolus=0.0)
        if time in self.insulin_time:
            action = Action(
                basal=self.basal,
                bolus=self.insulin_amount[self.insulin_time.index(time)])
        return action
