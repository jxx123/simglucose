from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
from simglucose.simulation.data_warehouse import InMemoryDataWarehouse

try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple('Observation', ['CGM'])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv(object):
    HORIZON = 1

    COL_NAMES = ['BG', 'CGM', 'insulin', 'CHO', 'Risk', 'LBGI', 'HBGI']
    INDEX_NAME = 'Time'

    def __init__(self, patient, sensor, pump, scenario):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self._reset()

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    @property
    def sample_time(self):
        return self.sensor.sample_time

    def mini_step(self, action):
        # current action
        patient_action = self.scenario.get_action(self.time)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)

        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM

    def step(self, action, reward_fun=risk_diff):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        # Only compute the moving mean of CHO and insulin. BG and CGM use the
        # sampled values.
        CHO = 0.0
        insulin = 0.0
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, BG, CGM = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time

        # Compute risk index
        LBGI, HBGI, risk = risk_index([BG], self.HORIZON)

        # Record action in the past sample interval (5 minutes)
        self.db.put(self.time - timedelta(minutes=self.sample_time),
                    CHO=CHO,
                    insulin=insulin)

        # Record current observation
        self.db.put(self.time,
                    BG=BG,
                    CGM=CGM,
                    Risk=risk,
                    LBGI=LBGI,
                    HBGI=HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.db.get('CGM')[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 70 or BG > 350  # Game over rule
        obs = Observation(CGM=CGM)

        return Step(
            observation=obs,
            reward=reward,
            done=done,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=CHO,
            patient_state=self.patient.state)

    def _reset(self):
        self.viewer = None

        BG = self.patient.observation.Gsub
        LBGI, HBGI, risk = risk_index([BG], self.HORIZON)
        CGM = self.sensor.measure(self.patient)

        # TODO: provide a factory in the future to support different types of
        # data IO
        self.db = InMemoryDataWarehouse(self.COL_NAMES, self.INDEX_NAME)
        self.db.put(self.time,
                    BG=BG,
                    CGM=CGM,
                    Risk=risk,
                    LBGI=LBGI,
                    HBGI=HBGI)

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        obs = Observation(CGM=CGM)
        return Step(
            observation=obs,
            reward=0,
            done=False,
            sample_time=self.sample_time,
            patient_name=self.patient.name,
            meal=0,
            patient_state=self.patient.state)

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        return self.db.getAll()
