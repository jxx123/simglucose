from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename(
    'simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class BBController(Controller):
    def __init__(self, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(
            PATIENT_PARA_FILE)
        self.target = target

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')

        action = self._bb_policy(
            pname,
            meal,
            observation.CGM,
            sample_time)
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        if any(self.quest.Name.str.match(name)):
            q = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = np.asscalar(params.u2ss.values)
            BW = np.asscalar(params.BW.values)
        else:
            q = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                             columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43
            BW = 57.0

        basal = u2ss * BW / 6000
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.debug('glucose = {}'.format(glucose))
            bolus = np.asscalar(meal / q.CR.values + (glucose > 150)
                                * (glucose - self.target) / q.CF.values)
        else:
            bolus = 0

        bolus = bolus / env_sample_time
        action = Action(basal=basal, bolus=bolus)
        return action

    def reset(self):
        pass
