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
import importlib.resources
from typing import Optional, Union

SENSOR_PARA_FILE = str(importlib.resources.files("simglucose") / "params/sensor_params.csv")


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
            # Full reset — reinitialise all AR(1) states
            self._rng = np.random.RandomState(self.seed)
            e_np = self._rng.randn(self.n)
            self.e = torch.tensor(e_np, dtype=self.dtype, device=self.device)
            self.last_cgm = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        else:
            # Partial reset for terminated sub-envs
            e_np = self._rng.randn(len(indices))
            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            self.e[idx_t] = torch.tensor(e_np, dtype=self.dtype, device=self.device)
            self.last_cgm[idx_t] = 0.0

    # ------------------------------------------------------------------
    def measure(self, gsub: torch.Tensor, t: int) -> torch.Tensor:
        """
        Measure CGM for all N patients at minute t.

        Parameters
        ----------
        gsub : (N,) subcutaneous glucose values (mg/dL)
        t    : current simulation minute (int)

        Returns
        -------
        cgm : (N,) clipped CGM readings
        """
        if t % self.sample_time == 0:
            # AR(1) step
            z_np = self._rng.randn(self.n)
            z = torch.tensor(z_np, dtype=self.dtype, device=self.device)
            self.e = self._PACF * (self.e + z)
            # Johnson SU transform
            noise = self._xi + self._lam * torch.sinh(
                (self.e - self._gamma) / self._delta
            )
            cgm = (gsub + noise).clamp(min=self._cgm_min, max=self._cgm_max)
            self.last_cgm = cgm
        return self.last_cgm
