import numpy as np


def make_precip(n_steps=100, p_const=10.0, n_rain=10):
    """Constant input of p_const (mm/step) during the first n_rain steps."""
    p = np.zeros(n_steps)
    p[:n_rain] = p_const
    return p


def linear_reservoir(P, k, S0=0.0, Smax=None, ET0=0.0, S_wp=0.0, S_c=None):
    """Single linear reservoir, explicit Euler, dt = 1 step (day).

    Q(t) = k * S(t)                       (k in 1/dt)
    ET(t) = ET0 * clip((S - S_wp)/(S_c - S_wp), 0, 1) ** 0.5
            with S_c = Smax and S_wp = 0 this is ET0 * (S/Smax)**0.5
    S(t+1) = S(t) + P(t) - Q(t) - ET(t); water above Smax leaves as spill
    (added to Q). ET is limited to the water available.
    Returns dict of arrays: S, Q, ET, spill (all mm/step; S in mm).
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    if ET0 > 0 and Smax is None:
        raise ValueError("ET needs Smax (or S_c) to be defined")
    if S_c is None:
        S_c = Smax

    S = np.zeros(n + 1)
    Q = np.zeros(n)
    ET = np.zeros(n)
    spill = np.zeros(n)
    S[0] = S0

    for t in range(n):
        Q[t] = k * S[t]
        if ET0 > 0:
            frac = np.clip((S[t] - S_wp) / (S_c - S_wp), 0.0, 1.0)
            ET[t] = ET0 * frac ** 0.5
        available = S[t] + P[t] - Q[t]
        ET[t] = min(ET[t], max(available, 0.0))
        S_new = available - ET[t]
        if Smax is not None and S_new > Smax:
            spill[t] = S_new - Smax
            S_new = Smax
        Q[t] += spill[t]
        S[t + 1] = S_new

    return {"S": S[:-1], "Q": Q, "ET": ET, "spill": spill, "S_end": S[-1]}


def cascade(P, k1, k2, S0_1=0.0, S0_2=0.0):
    """Two linear reservoirs in series: Q1 is the inflow to reservoir 2."""
    r1 = linear_reservoir(P, k1, S0=S0_1)
    r2 = linear_reservoir(r1["Q"], k2, S0=S0_2)
    return r1, r2
