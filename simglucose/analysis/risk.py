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
