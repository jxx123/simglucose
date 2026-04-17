"""
Vectorised CGM noise for N sensors running in parallel.

The AR(1) + Johnson-SU model mirrors noise_gen.py but operates directly
at 1-min steps (rather than 15-min samples with cubic interpolation).
The AR(1) coefficient and Johnson-SU parameters are the same as in the
original Dexcom model.  For RL training the simplified noise model is
statistically appropriate.
"""
import torch
import pandas as pd
import numpy as np
from typing import Optional, Union
from simglucose.utils import _get_resource_path

SENSOR_PARA_FILE = _get_resource_path("simglucose", "params/sensor_params.csv")


class CGMSensorBatch:
    """
    Vectorised CGM sensor for N patients.

    Parameters
    ----------
    sensor_name : str  e.g. 'Dexcom'
    n           : int  number of parallel sensors
    device      : str or torch.device
    seed        : int, optional
    dtype       : torch.dtype
    """

    def __init__(
        self,
        sensor_name: str,
        n: int,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed

        df = pd.read_csv(SENSOR_PARA_FILE)
        p = df.loc[df.Name == sensor_name].squeeze()
        self.sample_time = int(p["sample_time"])
        self._cgm_min = float(p["min"])
        self._cgm_max = float(p["max"])
        self._PACF = float(p["PACF"])
        self._gamma = float(p["gamma"])
        self._lam = float(p["lambda"])
        self._delta = float(p["delta"])
        self._xi = float(p["xi"])

        self._rng = None
        self._gen = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, indices=None):
        if indices is None:
            self._rng = np.random.RandomState(self.seed)
            # Initialise AR(1) state
            self.e = torch.tensor(self._rng.randn(self.n), dtype=self.dtype, device=self.device)
            self.last_cgm = torch.zeros(self.n, dtype=self.dtype, device=self.device)
            
            # 15-min noise buffer for interpolation
            self.noise15_prev = self._gen_noise15()
            self.noise15_next = self._gen_noise15()
        else:
            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            self.e[idx_t] = torch.tensor(self._rng.randn(len(indices)), dtype=self.dtype, device=self.device)
            self.last_cgm[idx_t] = 0.0
            
            # Reset buffers for specific indices
            n_idx = len(indices)
            self.noise15_prev[idx_t] = self._gen_noise15_idx(indices)
            self.noise15_next[idx_t] = self._gen_noise15_idx(indices)

    def _gen_noise15(self) -> torch.Tensor:
        """Advance AR(1) and apply Johnson SU for all N."""
        z = torch.tensor(self._rng.randn(self.n), dtype=self.dtype, device=self.device)
        self.e = self._PACF * (self.e + z)
        return self._xi + self._lam * torch.sinh((self.e - self._gamma) / self._delta)

    def _gen_noise15_idx(self, indices) -> torch.Tensor:
        """Advance AR(1) and apply Johnson SU for specific indices."""
        z = torch.tensor(self._rng.randn(len(indices)), dtype=self.dtype, device=self.device)
        self.e[indices] = self._PACF * (self.e[indices] + z)
        return self._xi + self._lam * torch.sinh((self.e[indices] - self._gamma) / self._delta)

    # ------------------------------------------------------------------
    def measure(self, gsub: torch.Tensor, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Measure CGM for all N patients at patient-specific time t with 
        linear interpolation between 15-minute AR(1) samples.
        """
        MDL_SAMPLE_TIME = 15
        
        # Update 15-min targets
        if isinstance(t, torch.Tensor):
            update_mask = (t > 0) & (t % MDL_SAMPLE_TIME == 0)
            if update_mask.any():
                idx_t = torch.where(update_mask)[0]
                self.noise15_prev[idx_t] = self.noise15_next[idx_t].clone()
                self.noise15_next[idx_t] = self._gen_noise15_idx(idx_t.cpu().numpy())
            
            # Linear interpolation (vectorised)
            alpha = (t % MDL_SAMPLE_TIME).to(self.dtype) / float(MDL_SAMPLE_TIME)
            noise = (1.0 - alpha) * self.noise15_prev + alpha * self.noise15_next
            
            # Apply only on sample_time boundaries
            # Note: T1DSimVectorEnv calls measure once per sample_time outside its inner loop,
            # so we assume t is already at a sample boundary for all patients here.
            cgm = (gsub + noise).clamp(min=self._cgm_min, max=self._cgm_max)
            self.last_cgm = cgm
            return self.last_cgm

        if t > 0 and t % MDL_SAMPLE_TIME == 0:
            self.noise15_prev = self.noise15_next.clone()
            self.noise15_next = self._gen_noise15()

        if t % self.sample_time == 0:
            # Linear interpolation
            alpha = (t % MDL_SAMPLE_TIME) / float(MDL_SAMPLE_TIME)
            noise = (1.0 - alpha) * self.noise15_prev + alpha * self.noise15_next
            
            cgm = (gsub + noise).clamp(min=self._cgm_min, max=self._cgm_max)
            self.last_cgm = cgm
            
        return self.last_cgm
