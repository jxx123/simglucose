import numpy as np


def risk_index(BG, horizon):
    # BG is in mg/dL
    BG_to_compute = BG[-horizon:]
    risks =[risk(r) for r in BG_to_compute]
    LBGI = np.mean([r[0] for r in risks])
    HBGI = np.mean([r[1] for r in risks])
    RI = np.mean([r[2] for r in risks])

    return (LBGI, HBGI, RI)

def risk(BG):
    """
    Risk is a percentage - ranging from 0 to 100%.
    The 20 and 600 mg/dl are just the values to which the risk formula was fit. 
    The aim is to make the risk maximum when it is either 20 or 600.
    The units in the paper below are different (mmol/l), but in our units (mg/dl) these limits are 20 and 600.

    Reference, in particular see appendix for the derivation of risk:
    https://diabetesjournals.org/care/article/20/11/1655/21162/Symmetrization-of-the-Blood-Glucose-Measurement

    """
    MIN_BG = 20.0
    MAX_BG = 600.0
    if BG <= MIN_BG: 
        return (100.0, 0.0, 100.0)
    if BG >= MAX_BG:
        return (0.0, 100.0, 100.0)
    
    U = 1.509 * (np.log(BG)**1.084 - 5.381)

    ri = 10 * U**2

    rl, rh = 0.0, 0.0
    if U <= 0:
        rl = ri
    if U >= 0:
        rh = ri
    return (rl, rh, ri)


# ---------------------------------------------------------------------------
# Vectorised batch variants (PyTorch) — used by the GPU batch environment
# ---------------------------------------------------------------------------

def risk_batch(BG):
    """
    Vectorised risk score for a (N,) BG tensor.

    Mirrors the scalar ``risk()`` function but operates on a full batch.

    Returns
    -------
    ri : Tensor, shape (N,)  — total risk index per patient
    rl : Tensor, shape (N,)  — low-BG component
    rh : Tensor, shape (N,)  — high-BG component
    """
    import torch
    MIN_BG = 20.0
    MAX_BG = 600.0

    BG_safe = BG.clamp(min=MIN_BG + 1e-8, max=MAX_BG - 1e-8)
    U = 1.509 * (torch.log(BG_safe) ** 1.084 - 5.381)
    ri = 10.0 * U ** 2

    # Boundary overrides
    ri = torch.where(BG <= MIN_BG, torch.full_like(ri, 100.0), ri)
    ri = torch.where(BG >= MAX_BG, torch.full_like(ri, 100.0), ri)

    rl = torch.where((U <= 0) & (BG > MIN_BG) & (BG < MAX_BG), ri, torch.zeros_like(ri))
    rl = torch.where(BG <= MIN_BG, torch.full_like(rl, 100.0), rl)

    rh = torch.where((U >= 0) & (BG > MIN_BG) & (BG < MAX_BG), ri, torch.zeros_like(ri))
    rh = torch.where(BG >= MAX_BG, torch.full_like(rh, 100.0), rh)

    return ri, rl, rh


def risk_diff_batch(cgm_hist):
    """
    Batch reward = risk[t-1] - risk[t]  (risk reduction is positive).

    Parameters
    ----------
    cgm_hist : Tensor shape (N, W), W >= 2 — rolling CGM window, newest last.

    Returns
    -------
    reward : Tensor shape (N,)
    """
    ri_prev, _, _ = risk_batch(cgm_hist[:, -2])
    ri_curr, _, _ = risk_batch(cgm_hist[:, -1])
    return ri_prev - ri_curr
