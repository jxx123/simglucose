"""
Batched GPU-compatible T1D patient model.

Translates T1DPatient.model() to operate on (N, 13) state tensors so that
N patients can be simulated simultaneously on CPU or CUDA.

Integration is done with a fixed-step RK4 at dt=1 min (matching the original
SAMPLE_TIME).  For RL training this is numerically close to the adaptive
dopri5 used in the original single-patient path.
"""
import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Union
from simglucose.utils import _get_resource_path

PATIENT_PARA_FILE = _get_resource_path("simglucose", "params/vpatient_params.csv")

# Names of scalar parameters accessed inside the ODE model
_ODE_PARAM_NAMES = [
    "BW", "kmax", "kmin", "b", "d", "kabs", "f", "Vg",
    "kp1", "kp2", "kp3", "Fsnc", "ke1", "ke2", "k1", "k2",
    "Vm0", "Vmx", "Km0", "m2", "m4", "m1", "ka1", "ka2",
    "Vi", "p2u", "Ib", "ki", "m30", "kd", "ksc", "u2ss",
]

ALL_PATIENT_NAMES = None  # lazy cache


def _all_patient_names():
    global ALL_PATIENT_NAMES
    if ALL_PATIENT_NAMES is None:
        df = pd.read_csv(PATIENT_PARA_FILE)
        ALL_PATIENT_NAMES = list(df["Name"].values)
    return ALL_PATIENT_NAMES


def _load_params_for_names(names: List[str], device, dtype):
    """Return (params_dict, init_states) for a list of patient names."""
    df = pd.read_csv(PATIENT_PARA_FILE)
    rows = [df.loc[df.Name == n].squeeze() for n in names]

    # Initial states: columns 2:15 → x0_ 1 … x0_13
    init_np = np.array([r.iloc[2:15].values.astype(np.float64) for r in rows])
    init_states = torch.tensor(init_np, dtype=dtype, device=device)

    params = {}
    for pname in _ODE_PARAM_NAMES:
        vals = np.array([float(r[pname]) for r in rows], dtype=np.float64)
        params[pname] = torch.tensor(vals, dtype=dtype, device=device)

    return params, init_states


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def t1d_rhs_batch(
    t: float,
    x: torch.Tensor,           # (N, 13)
    insulin: torch.Tensor,     # (N,)  U/min
    cho_rate: torch.Tensor,    # (N,)  g (announced this minute)
    params: dict,
    last_Qsto: torch.Tensor,   # (N,)
    last_foodtaken: torch.Tensor,  # (N,)
) -> torch.Tensor:             # (N, 13)
    """Batched ODE RHS — exact translation of T1DPatient.model()."""
    p = params
    N = x.shape[0]
    dxdt = torch.zeros_like(x)

    d = cho_rate * 1000.0                         # g -> mg  (N,)
    ins = insulin * 6000.0 / p["BW"]             # U/min -> pmol/kg/min  (N,)

    # ---- stomach -------------------------------------------------------
    qsto = x[:, 0] + x[:, 1]                     # (N,)
    Dbar = last_Qsto + last_foodtaken * 1000.0   # mg  (N,)

    Dbar_safe = Dbar.clamp(min=1e-8)
    aa = 5.0 / (2.0 * Dbar_safe * (1.0 - p["b"]))
    cc = 5.0 / (2.0 * Dbar_safe * p["d"])
    kgut_computed = p["kmin"] + (p["kmax"] - p["kmin"]) / 2.0 * (
        torch.tanh(aa * (qsto - p["b"] * Dbar))
        - torch.tanh(cc * (qsto - p["d"] * Dbar))
        + 2.0
    )
    kgut = torch.where(Dbar > 0, kgut_computed, p["kmax"])

    dxdt[:, 0] = -p["kmax"] * x[:, 0] + d
    dxdt[:, 1] = p["kmax"] * x[:, 0] - x[:, 1] * kgut
    dxdt[:, 2] = kgut * x[:, 1] - p["kabs"] * x[:, 2]

    # ---- glucose kinetics ----------------------------------------------
    Rat = p["f"] * p["kabs"] * x[:, 2] / p["BW"]
    EGPt = p["kp1"] - p["kp2"] * x[:, 3] - p["kp3"] * x[:, 8]
    Uiit = p["Fsnc"]
    Et = p["ke1"] * torch.relu(x[:, 3] - p["ke2"])  # renal excretion

    dxdt[:, 3] = EGPt.clamp(min=0) + Rat - Uiit - Et - p["k1"] * x[:, 3] + p["k2"] * x[:, 4]
    dxdt[:, 3] = dxdt[:, 3] * (x[:, 3] >= 0).to(x.dtype)

    Vmt = p["Vm0"] + p["Vmx"] * x[:, 6]
    Kmt = p["Km0"]
    Uidt = Vmt * x[:, 4] / (Kmt + x[:, 4] + 1e-12)
    dxdt[:, 4] = -Uidt + p["k1"] * x[:, 3] - p["k2"] * x[:, 4]
    dxdt[:, 4] = dxdt[:, 4] * (x[:, 4] >= 0).to(x.dtype)

    # ---- insulin kinetics ----------------------------------------------
    dxdt[:, 5] = (
        -(p["m2"] + p["m4"]) * x[:, 5]
        + p["m1"] * x[:, 9]
        + p["ka1"] * x[:, 10]
        + p["ka2"] * x[:, 11]
    )
    dxdt[:, 5] = dxdt[:, 5] * (x[:, 5] >= 0).to(x.dtype)
    It = x[:, 5] / p["Vi"]

    dxdt[:, 6] = -p["p2u"] * x[:, 6] + p["p2u"] * (It - p["Ib"])
    dxdt[:, 7] = -p["ki"] * (x[:, 7] - It)
    dxdt[:, 8] = -p["ki"] * (x[:, 8] - x[:, 7])

    dxdt[:, 9] = -(p["m1"] + p["m30"]) * x[:, 9] + p["m2"] * x[:, 5]
    dxdt[:, 9] = dxdt[:, 9] * (x[:, 9] >= 0).to(x.dtype)

    dxdt[:, 10] = ins - (p["ka1"] + p["kd"]) * x[:, 10]
    dxdt[:, 10] = dxdt[:, 10] * (x[:, 10] >= 0).to(x.dtype)

    dxdt[:, 11] = p["kd"] * x[:, 10] - p["ka2"] * x[:, 11]
    dxdt[:, 11] = dxdt[:, 11] * (x[:, 11] >= 0).to(x.dtype)

    dxdt[:, 12] = -p["ksc"] * x[:, 12] + p["ksc"] * x[:, 3]
    dxdt[:, 12] = dxdt[:, 12] * (x[:, 12] >= 0).to(x.dtype)

    return dxdt


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------

def rk4_step(rhs, t, x, dt, **kwargs):
    """Standard 4-stage RK4 step. kwargs forwarded to rhs unchanged."""
    k1 = rhs(t,          x,                **kwargs)
    k2 = rhs(t + dt / 2, x + dt / 2 * k1, **kwargs)
    k3 = rhs(t + dt / 2, x + dt / 2 * k2, **kwargs)
    k4 = rhs(t + dt,     x + dt     * k3, **kwargs)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Batch patient class
# ---------------------------------------------------------------------------

class T1DPatientBatch:
    """
    Batch T1D patient model running N patients in parallel as tensors.

    Parameters
    ----------
    patient_names : list of str, length N
        Each entry is a virtual patient name, e.g. 'adolescent#001'.
    device : str or torch.device
    random_init_bg : bool
        If True, randomise initial blood-glucose states (same as T1DPatient).
    seed : int, optional
    dtype : torch.dtype
        Default float64 for numerical parity with the scipy path.
    """

    SAMPLE_TIME = 1  # minutes per ODE step
    EAT_RATE = 5.0   # g/min CHO

    def __init__(
        self,
        patient_names: List[str],
        device: Union[str, torch.device] = "cpu",
        random_init_bg: bool = False,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.patient_names = list(patient_names)
        self.N = len(patient_names)
        self.random_init_bg = random_init_bg
        self.seed = seed

        self._base_params, self._base_init_states = _load_params_for_names(
            patient_names, self.device, dtype
        )
        # Working copies (may differ from base if patient slots are resampled)
        self.params = {k: v.clone() for k, v in self._base_params.items()}
        self.init_states = self._base_init_states.clone()

        self._rng = np.random.RandomState(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, indices=None):
        """
        Reset all (or a subset of) patient states to initial conditions.

        Parameters
        ----------
        indices : list of int or None
            If None, reset all N patients.
        """
        if indices is None:
            indices = list(range(self.N))

        init = self.init_states[indices].clone()  # (|indices|, 13)

        if self.random_init_bg and len(indices) > 0:
            n_idx = len(indices)
            means = init[:, [3, 4, 12]]                        # (n_idx, 3)
            variances = 0.1 * means.clamp(min=0)               # (n_idx, 3)
            stds = variances.sqrt()
            noise = torch.tensor(
                self._rng.randn(n_idx, 3), dtype=self.dtype, device=self.device
            )
            init[:, 3] = (means[:, 0] + stds[:, 0] * noise[:, 0]).clamp(min=0)
            init[:, 4] = (means[:, 1] + stds[:, 1] * noise[:, 1]).clamp(min=0)
            init[:, 12] = (means[:, 2] + stds[:, 2] * noise[:, 2]).clamp(min=0)

        idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)

        if not hasattr(self, "x"):
            # First initialisation — create tensors for all N
            self.x = torch.zeros(self.N, 13, dtype=self.dtype, device=self.device)
            self._last_Qsto = torch.zeros(self.N, dtype=self.dtype, device=self.device)
            self._last_foodtaken = torch.zeros(self.N, dtype=self.dtype, device=self.device)
            self.is_eating = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            self.planned_meal = torch.zeros(self.N, dtype=self.dtype, device=self.device)
            self._last_cho = torch.zeros(self.N, dtype=self.dtype, device=self.device)
            self.t = 0

        self.x[idx_t] = init
        self._last_Qsto[idx_t] = init[:, 0] + init[:, 1]
        self._last_foodtaken[idx_t] = 0.0
        self.is_eating[idx_t] = False
        self.planned_meal[idx_t] = 0.0
        self._last_cho[idx_t] = 0.0
        if indices == list(range(self.N)):
            self.t = 0

    # ------------------------------------------------------------------
    def update_patient_slots(self, indices: List[int], new_names: List[str]):
        """
        Replace patient physiology for specific slots (used in auto-reset
        when patient_name is a pool).  Updates params and init_states in-place.
        """
        new_params, new_init = _load_params_for_names(
            new_names, self.device, self.dtype
        )
        idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
        for k in _ODE_PARAM_NAMES:
            self.params[k][idx_t] = new_params[k]
        self.init_states[idx_t] = new_init
        # Update name list
        for pos, name in zip(indices, new_names):
            self.patient_names[pos] = name

    # ------------------------------------------------------------------
    def step(self, insulin: torch.Tensor, cho_announced: torch.Tensor):
        """
        Advance all N patients by one minute (SAMPLE_TIME = 1 min).

        Parameters
        ----------
        insulin      : (N,) tensor, U/min
        cho_announced: (N,) tensor, grams announced this minute
        """
        # --- meal announcement logic (vectorised) ---
        self.planned_meal = self.planned_meal + cho_announced
        to_eat = self.planned_meal.clamp(max=self.EAT_RATE)
        to_eat = torch.where(self.planned_meal > 0, to_eat, torch.zeros_like(to_eat))
        self.planned_meal = (self.planned_meal - to_eat).clamp(min=0)

        # Meal-start event: CHO transitions 0 → >0
        meal_start = (cho_announced > 0) & (self._last_cho <= 0)
        qsto_snap = self.x[:, 0] + self.x[:, 1]
        self._last_Qsto = torch.where(meal_start, qsto_snap, self._last_Qsto)
        self._last_foodtaken = torch.where(
            meal_start, torch.zeros_like(self._last_foodtaken), self._last_foodtaken
        )
        self.is_eating = self.is_eating | meal_start

        # Accumulate food eaten
        self._last_foodtaken = self._last_foodtaken + torch.where(
            self.is_eating, to_eat, torch.zeros_like(to_eat)
        )

        # Meal-end event: CHO transitions >0 → 0
        meal_end = (cho_announced <= 0) & (self._last_cho > 0)
        self.is_eating = self.is_eating & ~meal_end
        self._last_cho = cho_announced.clone()

        # --- RK4 integration with sub-stepping ---
        # 10 sub-steps of 0.1 min per minute significantly improves parity with
        # the adaptive dopri5 solver used in the single-patient path.
        n_substeps = 10
        dt_sub = float(self.SAMPLE_TIME) / n_substeps
        
        for _ in range(n_substeps):
            self.x = rk4_step(
                t1d_rhs_batch,
                float(self.t),
                self.x,
                dt_sub,
                insulin=insulin,
                cho_rate=to_eat,
                params=self.params,
                last_Qsto=self._last_Qsto,
                last_foodtaken=self._last_foodtaken,
            )
            self.t += dt_sub

    # ------------------------------------------------------------------
    @property
    def observation(self) -> torch.Tensor:
        """Subcutaneous glucose (mg/dL), shape (N,)."""
        return self.x[:, 12] / self.params["Vg"]

    @property
    def bg(self) -> torch.Tensor:
        """Plasma glucose concentration (mg/dL), shape (N,)."""
        return self.x[:, 3] / self.params["Vg"]
