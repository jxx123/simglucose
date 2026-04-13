from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import importlib.resources
import numpy as np
from datetime import datetime
import gymnasium
try:
    import gym
    _GymEnvBase = gym.Env
except ImportError:
    _GymEnvBase = object


class _Box(gymnasium.spaces.Box):
    """Box space that silently accepts numpy scalars and other array-likes.

    gymnasium's PassiveEnvChecker calls action_space.contains(action) and
    warns when the value is not already an np.ndarray.  Scalars produced by
    np.float32(...) are numpy scalars, not ndarray, so they trip that check.
    This subclass converts the value before delegating to avoid the warning.
    """

    def contains(self, x) -> bool:
        return super().contains(np.asarray(x, dtype=self.dtype))


class T1DSimEnv(_GymEnvBase):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support the gym API.
    """

    metadata = {"render.modes": ["human"]}
    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None,
        start_time=None,
    ):
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        if gym is not None:
            from gym.utils import seeding
            self.np_random, _ = seeding.np_random(seed=seed)
        else:
            self.np_random = np.random.RandomState(seed)
        self.custom_scenario = custom_scenario
        self.start_time = start_time
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        if gym is not None:
            from gym.utils import seeding
            self.np_random, seed1 = seeding.np_random(seed=seed)
        else:
            self.np_random = np.random.RandomState(seed)
            seed1 = seed
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        if gym is not None:
            from gym.utils import seeding
            seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
            seed3 = seeding.hash_seed(seed2 + 1) % 2**31
            seed4 = seeding.hash_seed(seed3 + 1) % 2**31
        else:
            seed2 = int(self.np_random.randint(0, 2**31))
            seed3 = int(self.np_random.randint(0, 2**31))
            seed4 = int(self.np_random.randint(0, 2**31))

        if self.start_time is not None:
            start_time = self.start_time
        else:
            hour = int(self.np_random.randint(low=0, high=24))
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

    def render(self, mode="human", close=False):
        self.env.render(close=close)

    def close(self):
        self.env._close_viewer()

    @property
    def action_space(self):
        ub = self.env.pump._params["max_basal"]
        if gym is not None:
            from gym import spaces
            return spaces.Box(low=0, high=ub, shape=(1,))
        return gymnasium.spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        if gym is not None:
            from gym import spaces
            return spaces.Box(low=0, high=1000, shape=(1,))
        return gymnasium.spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]


class T1DSimGymnaisumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000
    MAX_CHO = 200  # max carbohydrate intake in grams

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        start_time=None,
        # --- GPU batch parameters ---
        n_envs: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # Dispatch to the GPU batch path when n_envs > 1 or device != 'cpu'.
        # PyTorch is imported lazily so non-GPU users are unaffected.
        self._use_batch = (n_envs > 1) or (device != "cpu")
        self._n_envs = n_envs
        self._device = device

        if self._use_batch:
            from simglucose.envs.batch_env import T1DSimVectorEnv
            self._batch_env = T1DSimVectorEnv(
                n_envs=n_envs,
                patient_name=patient_name,
                reward_fun=reward_fun,
                seed=seed,
                device=device,
            )
            # Expose VectorEnv-style attributes
            self.num_envs = n_envs
            self.observation_space = self._batch_env.observation_space
            self.action_space = self._batch_env.action_space
            self.single_observation_space = self._batch_env.single_observation_space
            self.single_action_space = self._batch_env.single_action_space
            self._sample_time = self._batch_env._sample_time
            return  # skip single-patient init

        # ----- existing single-patient path (unchanged) -----
        self.render_mode = render_mode
        self.env = T1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            start_time=start_time,
        )
        sample_time = self.env.env.sensor.sample_time
        max_basal_u_hr = self.env.max_basal * 60  # convert U/min to U/hr
        max_bolus_u = self.env.env.pump._params["max_bolus"] * sample_time  # convert U/min to U per step
        self.observation_space = gymnasium.spaces.Dict({
            "CGM": _Box(low=0.0, high=self.MAX_BG, shape=(), dtype=np.float32),
            "CHO": _Box(low=0.0, high=self.MAX_CHO, shape=(), dtype=np.float32),
        })
        self.action_space = gymnasium.spaces.Dict({
            "basal": _Box(low=0.0, high=max_basal_u_hr, shape=(), dtype=np.float32),
            "bolus": _Box(low=0.0, high=max_bolus_u, shape=(), dtype=np.float32),
        })
        self._sample_time = sample_time

    @property
    def sample_time(self):
        return self._sample_time

    def step(self, action):
        if self._use_batch:
            return self._batch_env.step(action)
        # Convert basal from U/hr to U/min, bolus from U to U/min
        basal_u_min = float(action["basal"]) / 60.0
        bolus_u_min = float(action["bolus"]) / self._sample_time
        act = Action(basal=basal_u_min, bolus=bolus_u_min)
        if self.env.reward_fun is None:
            obs, reward, done, info = self.env.env.step(act)
        else:
            obs, reward, done, info = self.env.env.step(act, reward_fun=self.env.reward_fun)
        truncated = False
        # info["meal"] is in g/min (averaged over mini_steps), convert to grams
        cho_grams = info["meal"] * self._sample_time
        observation = {"CGM": np.float32(obs.CGM), "CHO": np.float32(cho_grams)}
        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if self._use_batch:
            return self._batch_env.reset(seed=seed, options=options)
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        observation = {"CGM": np.float32(obs.CGM), "CHO": np.float32(0.0)}
        return observation, info

    def render(self):
        if self._use_batch:
            return
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        if self._use_batch:
            self._batch_env.close()
            return
        self.env.close()
