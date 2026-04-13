"""
T1DSimVectorEnv — GPU-batched T1D simulator for RL rollouts.

Runs N patients simultaneously as PyTorch tensors on CPU or CUDA.
Implements the gymnasium.vector.VectorEnv API for compatibility with
RL libraries (stable-baselines3, CleanRL, Tianshou, etc.).

Patient name semantics
----------------------
patient_name : str
    All N envs use this same patient physiology (different random seeds).
patient_name : list, len == n_envs
    Fixed slot assignment: env[i] is always patient_name[i].
patient_name : list, len != n_envs  (pool)
    Each env independently samples from the list at every auto-reset.
patient_name : None
    Each env samples uniformly from all 30 virtual patients at every reset.
"""
import numpy as np
import torch
import pandas as pd
import importlib.resources
import gymnasium
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium import spaces
from typing import Callable, List, Optional, Union

from simglucose.patient.t1dpatient_batch import T1DPatientBatch, _all_patient_names
from simglucose.sensor.cgm_batch import CGMSensorBatch
from simglucose.simulation.scenario_batch import BatchScenario
from simglucose.analysis.risk import risk_diff_batch

PUMP_PARA_FILE = str(importlib.resources.files("simglucose") / "params/pump_params.csv")

U2PMOL = 6000.0
SENSOR_HARDWARE = "Dexcom"
PUMP_HARDWARE = "Insulet"
MAX_BG = 1000.0
MAX_CHO = 200.0


def _pump_action_batch(
    basal_u_hr: torch.Tensor,   # (N,) U/hr  (action space units)
    bolus_u_step: torch.Tensor, # (N,) U per step
    pump_params: pd.Series,
    sample_time: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    """
    Convert action-space units to U/min and apply pump quantisation + clamping.

    Returns
    -------
    basal_u_min : (N,) U/min
    bolus_u_min : (N,) U/min
    """
    # Convert units
    basal = basal_u_hr / 60.0                  # U/hr -> U/min
    bolus = bolus_u_step / float(sample_time)  # U/step -> U/min

    inc_basal = float(pump_params["inc_basal"])
    inc_bolus = float(pump_params["inc_bolus"])
    min_basal = float(pump_params["min_basal"])
    max_basal = float(pump_params["max_basal"])
    min_bolus = float(pump_params["min_bolus"])
    max_bolus = float(pump_params["max_bolus"])

    def quantise(x, inc, lo, hi):
        pmol = x * U2PMOL
        pmol = torch.round(pmol / inc) * inc
        return (pmol / U2PMOL).clamp(min=lo, max=hi)

    return quantise(basal, inc_basal, min_basal, max_basal), \
           quantise(bolus, inc_bolus, min_bolus, max_bolus)


class T1DSimVectorEnv(VectorEnv):
    """
    Vectorised T1D simulator for GPU-accelerated RL rollouts.

    Parameters
    ----------
    n_envs        : int   number of parallel environments
    patient_name  : str, list[str], or None  — see module docstring
    reward_fun    : callable(cgm_hist: Tensor(N,W)) -> Tensor(N,) or None
                    If None, uses risk_diff_batch.
    seed          : int, optional
    device        : str or torch.device  e.g. 'cpu', 'cuda', 'cuda:0'
    dtype         : torch.dtype  default float64 for numerical fidelity
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_envs: int,
        patient_name: Union[str, List[str], None] = None,
        reward_fun: Optional[Callable] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.num_envs = n_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        self.reward_fun = reward_fun

        # --- pump params ------------------------------------------------
        pump_df = pd.read_csv(PUMP_PARA_FILE)
        self._pump_params = pump_df.loc[pump_df.Name == PUMP_HARDWARE].squeeze()

        # --- sensor sample_time ----------------------------------------
        sensor_df = pd.read_csv(
            str(importlib.resources.files("simglucose") / "params/sensor_params.csv")
        )
        sp = sensor_df.loc[sensor_df.Name == SENSOR_HARDWARE].squeeze()
        self._sample_time = int(sp["sample_time"])
        _max_basal = float(self._pump_params["max_basal"])
        _max_bolus = float(self._pump_params["max_bolus"])

        # Action-space units: basal in U/hr, bolus in U per step
        max_basal_u_hr = _max_basal * 60.0
        max_bolus_u_step = _max_bolus * self._sample_time

        # --- spaces -----------------------------------------------------
        single_obs = spaces.Dict({
            "CGM": spaces.Box(0.0, MAX_BG,  shape=(), dtype=np.float32),
            "CHO": spaces.Box(0.0, MAX_CHO, shape=(), dtype=np.float32),
        })
        single_act = spaces.Dict({
            "basal": spaces.Box(0.0, max_basal_u_hr,   shape=(), dtype=np.float32),
            "bolus": spaces.Box(0.0, max_bolus_u_step,  shape=(), dtype=np.float32),
        })
        # gymnasium.vector.VectorEnv doesn't always accept __init__ args;
        # set required attributes directly.
        self.num_envs = n_envs
        self.observation_space = batch_space(single_obs, n_envs)
        self.action_space = batch_space(single_act, n_envs)
        self.single_observation_space = single_obs
        self.single_action_space = single_act

        # --- patient name pool / assignment ----------------------------
        self._all_names = _all_patient_names()
        self._name_spec = patient_name  # original spec, kept for reset logic
        self._name_pool: Optional[List[str]] = None   # list to sample from, or None
        self._fixed_names: Optional[List[str]] = None  # fixed per-slot names, or None

        if patient_name is None:
            self._name_pool = self._all_names
        elif isinstance(patient_name, str):
            self._fixed_names = [patient_name] * n_envs
        elif isinstance(patient_name, list):
            if len(patient_name) == n_envs:
                self._fixed_names = list(patient_name)
            else:
                self._name_pool = list(patient_name)
        else:
            raise ValueError(f"Unsupported patient_name type: {type(patient_name)}")

        self._rng = np.random.RandomState(seed)

        # --- sub-components (created on first reset) -------------------
        self._patient: Optional[T1DPatientBatch] = None
        self._sensor: Optional[CGMSensorBatch] = None
        self._scenario: Optional[BatchScenario] = None

        # CGM history buffer for reward computation (60-min window)
        self._window = max(2, 60 // self._sample_time)
        self._cgm_hist: Optional[torch.Tensor] = None  # (N, window)

        self._t = 0  # global step counter

    # ------------------------------------------------------------------
    def _sample_names(self, indices: List[int]) -> List[str]:
        """Sample patient names for the given slot indices."""
        if self._fixed_names is not None:
            return [self._fixed_names[i] for i in indices]
        else:
            return [
                self._name_pool[int(self._rng.randint(0, len(self._name_pool)))]
                for _ in indices
            ]

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        all_idx = list(range(self.num_envs))
        names = self._sample_names(all_idx)

        if self._patient is None:
            # First ever reset — create components
            self._patient = T1DPatientBatch(
                names, device=self.device, random_init_bg=True,
                seed=seed, dtype=self.dtype,
            )
            self._sensor = CGMSensorBatch(
                SENSOR_HARDWARE, self.num_envs,
                device=self.device, seed=seed, dtype=self.dtype
            )
            self._scenario = BatchScenario(
                self.num_envs, device=self.device, seed=seed, dtype=self.dtype
            )
        else:
            # Reassign patient slots if using a pool
            if self._name_pool is not None:
                self._patient.update_patient_slots(all_idx, names)
            self._patient.reset()
            self._sensor.reset()
            self._scenario.reset()

        self._t = 0

        # Initial CGM measurement
        cgm = self._sensor.measure(self._patient.observation, self._t)  # (N,)
        self._cgm_hist = cgm.unsqueeze(1).expand(
            self.num_envs, self._window
        ).clone()  # (N, window)

        obs = {
            "CGM": cgm.cpu().float().numpy(),
            "CHO": np.zeros(self.num_envs, dtype=np.float32),
        }
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, actions):
        """
        Parameters
        ----------
        actions : dict {"basal": array(N,), "bolus": array(N,)}
                  basal in U/hr, bolus in U per step (matching action_space)

        Returns
        -------
        obs, rewards, terminated, truncated, info  — all shape (N,) numpy
        """
        # Convert actions to tensors
        basal_t = torch.as_tensor(
            actions["basal"], dtype=self.dtype, device=self.device
        ).reshape(self.num_envs)
        bolus_t = torch.as_tensor(
            actions["bolus"], dtype=self.dtype, device=self.device
        ).reshape(self.num_envs)

        # Apply pump quantisation + clamping
        basal_u_min, bolus_u_min = _pump_action_batch(
            basal_t, bolus_t, self._pump_params,
            self._sample_time, self.dtype, self.device
        )

        # Inner loop: sample_time patient steps (1 min each)
        cho_sum = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        bg_sum  = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        cgm_last = self._sensor.last_cgm.clone()

        for _ in range(self._sample_time):
            cho = self._scenario.get_cho_rate(self._t)  # (N,) grams
            insulin = basal_u_min + bolus_u_min          # (N,) U/min
            self._patient.step(insulin, cho)
            bg_sum  = bg_sum  + self._patient.bg
            cho_sum = cho_sum + cho
            self._t += 1

        bg_avg  = bg_sum  / self._sample_time
        cho_avg = cho_sum / self._sample_time

        # CGM measurement (after all mini-steps, once per sample_time)
        cgm = self._sensor.measure(self._patient.observation, self._t)  # (N,)

        # Roll CGM history and insert new measurement
        self._cgm_hist = torch.roll(self._cgm_hist, -1, dims=1)
        self._cgm_hist[:, -1] = cgm

        # Reward
        if self.reward_fun is not None:
            rewards_t = self.reward_fun(self._cgm_hist)
        else:
            rewards_t = risk_diff_batch(self._cgm_hist)

        # Termination
        terminated_t = (bg_avg < 10.0) | (bg_avg > 600.0)  # (N,) bool
        truncated_t  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Build obs
        cgm_np  = cgm.cpu().float().numpy()
        cho_np  = (cho_avg * self._sample_time).cpu().float().numpy()  # grams this step
        rew_np  = rewards_t.cpu().float().numpy()
        term_np = terminated_t.cpu().numpy()
        trunc_np = truncated_t.cpu().numpy()

        obs = {"CGM": cgm_np, "CHO": cho_np}
        info = {}

        # Auto-reset terminated sub-envs (gymnasium vector convention)
        done_idx = np.where(term_np)[0].tolist()
        if done_idx:
            # Save final observations before resetting
            final_cgm = cgm_np.copy()
            final_cho = cho_np.copy()
            info["final_observation"] = {"CGM": final_cgm, "CHO": final_cho}
            info["_final_observation"] = term_np.copy()

            # Resample patient names for done slots (pool semantics)
            new_names = self._sample_names(done_idx)
            if self._name_pool is not None:
                self._patient.update_patient_slots(done_idx, new_names)

            # Reset done patients/sensors/scenarios
            self._patient.reset(indices=done_idx)
            self._sensor.reset(indices=done_idx)
            self._scenario.reset(indices=done_idx)

            # Measure new initial CGM for done slots
            new_cgm = self._sensor.measure(
                self._patient.observation, self._t
            )  # (N,) — only done slots updated

            init_cgm = new_cgm[done_idx].cpu().float().numpy()
            obs["CGM"][done_idx] = init_cgm
            obs["CHO"][done_idx] = 0.0

            # Reset CGM hist for done slots
            hist_init = new_cgm[done_idx].unsqueeze(1).expand(
                len(done_idx), self._window
            )
            idx_t = torch.tensor(done_idx, dtype=torch.long, device=self.device)
            self._cgm_hist[idx_t] = hist_init

        return obs, rew_np, term_np, trunc_np, info

    # ------------------------------------------------------------------
    def render(self):
        pass  # no rendering for batch env

    def close(self):
        pass
