from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import gymnasium


PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)


class T1DSimEnv(gym.Env):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        **kwargs,
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()
        # Set custom metadata
        for k, v in kwargs.items():
            if k in ["render.modes"]:
                continue
            self.metadata[k] = v

    def _step(self, action: np.ndarray):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid.")

        act = Action(basal=action[0], bolus=action[1])
        obs_tuple, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        obs = np.array([obs_tuple.CGM, obs_tuple.CHO], dtype=np.float32)
        return obs, reward, done, info

    def _raw_reset(self):
        obs_tuple, reward, done, info = self.env.reset()
        obs = np.array([obs_tuple.CGM, obs_tuple.CHO], dtype=np.float32)
        return obs, reward, done, info

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs_tuple, _, _, _ = self.env.reset()
        obs = np.array([obs_tuple.CGM, obs_tuple.CHO], dtype=np.float32)
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        return self.env.render(mode=mode, close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        basal_ub = self.max_basal
        bolus_ub = self.max_bolus
        return spaces.Box(low=np.array([0.0, 0.0]), high=np.array([basal_ub, bolus_ub]))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=10000, shape=(2,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]

    @property
    def max_bolus(self):
        return self.env.pump._params["max_bolus"]


class T1DSimGymnaisumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            **kwargs,
        )

        # Set custom metadata
        for k, v in kwargs.items():
            if k in ["render_modes", "render_fps"]:
                continue
            self.metadata[k] = v

        self.observation_space = gymnasium.spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        # Once the max_episode_steps is set, the truncated value will be overridden.
        truncated = False
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        return obs, info

    def render(self):
        return self.env.render(mode=self.render_mode)

    def close(self):
        self.env.close()
