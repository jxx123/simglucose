"""
Pre-computed meal scenarios for N patients stored as a dense (N, 1440) tensor.

At reset, N independent RandomScenario schedules are generated in Python
(fast, once per episode), densified into a tensor, and indexed by
(start_minute_of_day + t) % 1440 during stepping — zero Python overhead
inside the hot loop.
"""
import torch
import numpy as np
from scipy.stats import truncnorm
from typing import List, Optional, Union


_MEAL_PROB   = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
_TIME_LB     = np.array([5, 9, 10, 14, 16, 20]) * 60
_TIME_UB     = np.array([9, 10, 14, 16, 20, 23]) * 60
_TIME_MU     = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
_TIME_SIGMA  = np.array([60, 30, 60, 30, 60, 30])
_AMOUNT_MU   = [45, 10, 70, 10, 80, 10]
_AMOUNT_SIGMA = [10, 5, 10, 5, 10, 5]


def _create_one_scenario(rng: np.random.RandomState) -> dict:
    """Generate a single day meal schedule (mirrors RandomScenario.create_scenario)."""
    scenario = {"meal": {"time": [], "amount": []}}
    for p, tlb, tub, tbar, tsd, mbar, msd in zip(
        _MEAL_PROB, _TIME_LB, _TIME_UB, _TIME_MU, _TIME_SIGMA,
        _AMOUNT_MU, _AMOUNT_SIGMA
    ):
        if rng.rand() < p:
            tmeal = int(np.round(truncnorm.rvs(
                a=(tlb - tbar) / tsd, b=(tub - tbar) / tsd,
                loc=tbar, scale=tsd, random_state=rng
            )))
            amount = max(round(rng.normal(mbar, msd)), 0)
            scenario["meal"]["time"].append(tmeal)
            scenario["meal"]["amount"].append(amount)
    return scenario


class BatchScenario:
    """
    Pre-computed meal scenarios for N patients.

    meal_schedule[i, t] = grams announced to patient i at minute-of-day t.
    Lookup at step t: gather(meal_schedule, (start_minutes + t) % 1440).

    Parameters
    ----------
    n           : int  number of patients
    device      : str or torch.device
    seed        : int, optional
    dtype       : torch.dtype
    """

    T_MAX = 1440  # minutes per day

    def __init__(
        self,
        n: int,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.meal_schedule = torch.zeros(
            n, self.T_MAX, dtype=dtype, device=self.device
        )
        self.start_minutes = torch.zeros(n, dtype=torch.long, device=self.device)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, indices=None):
        """Re-generate scenarios for all patients (or a subset)."""
        if indices is None:
            indices = list(range(self.n))
        self._reset_indices(indices)

    def _reset_indices(self, indices: List[int]):
        for i in indices:
            rng_i = np.random.RandomState(int(self._rng.randint(0, 2 ** 31)))
            scen = _create_one_scenario(rng_i)

            sched = np.zeros(self.T_MAX, dtype=np.float64)
            for t_min, amount in zip(
                scen["meal"]["time"], scen["meal"]["amount"]
            ):
                if 0 <= t_min < self.T_MAX:
                    sched[t_min] = float(amount)

            self.meal_schedule[i] = torch.tensor(
                sched, dtype=self.dtype, device=self.device
            )
            start_h = int(self._rng.randint(0, 24))
            self.start_minutes[i] = start_h * 60

    # ------------------------------------------------------------------
    def get_cho_rate(self, t: int) -> torch.Tensor:
        """
        Return announced meal (grams) for each patient at global step t.

        Uses each patient's start_minute offset so different patients are
        at different points in their daily schedule.

        Returns
        -------
        cho : (N,) tensor of grams
        """
        idx = (self.start_minutes + t) % self.T_MAX  # (N,) long
        return self.meal_schedule.gather(1, idx.unsqueeze(1)).squeeze(1)
