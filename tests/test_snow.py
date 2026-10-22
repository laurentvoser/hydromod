import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from hydromod.snow import snow_fraction, snow_model


def test_fraction_limits_and_midpoint():
    f = snow_fraction([-5, 0, 1, 2, 10], t_snow=1.0, width=2.0)
    assert f == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])


def test_no_melt_below_threshold():
    r = snow_model(np.full(10, 5.0), np.full(10, -5.0))
    assert r["melt"].sum() == 0
    assert r["swe"][-1] == pytest.approx(50.0)


def test_swe_never_negative_and_melts_out():
    P = np.r_[np.full(5, 10.0), np.zeros(20)]
    T = np.r_[np.full(5, -3.0), np.full(20, 5.0)]
    r = snow_model(P, T, ddf=4.0)
    assert r["swe"].min() >= 0
    assert r["swe"][-1] == 0


def test_mass_balance():
    rng = np.random.default_rng(0)
    P = rng.gamma(0.5, 6, 500)
    T = 8 + 10 * np.sin(np.arange(500) * 2 * np.pi / 365) + rng.normal(0, 3, 500)
    r = snow_model(P, T)
    assert P.sum() == pytest.approx(r["outflow"].sum() + r["swe"][-1])
