import numpy as np


def snow_fraction(T, t_snow=1.0, width=2.0):
    """Fraction of precipitation falling as snow.

    Linear transition between TLOW = t_snow - width/2 (100 % snow) and
    TUP = t_snow + width/2 (0 % snow). The 50 % point is t_snow.
    width = 0 gives a sharp threshold.
    """
    T = np.asarray(T, dtype=float)
    if width <= 0:
        return (T <= t_snow).astype(float)
    t_low = t_snow - width / 2.0
    return np.clip((t_snow + width / 2.0 - T) / width, 0.0, 1.0)


def snow_model(P, T, ddf=1.0, t_snow=1.0, width=2.0, t_melt=0.0, swe0=0.0):
    """Degree-day snow model, daily time step.

    P: precipitation (mm/d), T: mean air temperature (degC)
    ddf: degree-day factor (mm/degC/d), t_snow: 50 % snow/rain temperature,
    width: width of the rain-snow transition, t_melt: melt threshold,
    swe0: initial snow water equivalent (mm).
    Returns dict of arrays: swe (end of day), snowfall, rain, melt, outflow
    (outflow = rain + melt, the water leaving the snowpack).
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(P)
    frac = snow_fraction(T, t_snow, width)
    snowfall = P * frac
    rain = P - snowfall

    swe = np.zeros(n)
    melt = np.zeros(n)
    store = swe0
    for t in range(n):
        store += snowfall[t]
        melt[t] = min(store, ddf * max(T[t] - t_melt, 0.0))
        store -= melt[t]
        swe[t] = store

    return {"swe": swe, "snowfall": snowfall, "rain": rain,
            "melt": melt, "outflow": rain + melt}
