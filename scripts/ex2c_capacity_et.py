import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
from hydromod.models import make_precip, linear_reservoir

K = 0.4
S_MAX = 50.0   # mm, assumed
ET0 = 2.0      # mm/d
N_STEPS = 100
# Optional: wilting point / field capacity as storages in mm.
# Defaults reproduce ET = ET0 * (S/Smax)**0.5
S_WP = 0.0
S_C = S_MAX


def run():
    P = make_precip(N_STEPS, p_const=10.0, n_rain=10)
    base = linear_reservoir(P, k=K)
    cap = linear_reservoir(P, k=K, Smax=S_MAX)
    full = linear_reservoir(P, k=K, Smax=S_MAX, ET0=ET0, S_wp=S_WP, S_c=S_C)

    t = np.arange(N_STEPS)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, base["Q"], label="Q: no Smax, no ET", color="tab:gray")
    ax.plot(t, cap["Q"], label=f"Q: Smax = {S_MAX:g} mm", color="tab:blue")
    ax.plot(t, full["Q"], label="Q: Smax + ET", color="tab:red")
    ax.set_ylabel("Q (mm/step)")
    ax_p = ax.twinx()
    ax_p.bar(t, P, color="tab:cyan", alpha=0.4, width=1)
    ax_p.invert_yaxis()
    ax_p.set_ylabel("P (mm/step)")
    ax.legend(loc="center right")
    ax.set_title(f"2c Capacity and ET, k = {K}")

    ax = axes[1]
    ax.plot(t, base["S"], label="S: no Smax, no ET", color="tab:gray")
    ax.plot(t, full["S"], label="S: Smax + ET", color="tab:red")
    ax.plot(t, full["ET"], label="ET (mm/step)", color="tab:orange")
    ax.axhline(S_MAX, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("S (mm), ET (mm/step)")
    ax.legend()

    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig(Path("plots") / "ex2c_capacity_et.png", dpi=150)
    plt.close(fig)

    balance = P.sum() - full["Q"].sum() - full["ET"].sum() - full["S_end"]
    print(f"2c: max Q (Smax only) = {cap['Q'].max():.3f}, "
          f"max Q (Smax + ET) = {full['Q'].max():.3f} mm/step")
    print(f"2c: total spill = {full['spill'].sum():.3f} mm, "
          f"total ET = {full['ET'].sum():.3f} mm, mass balance error = {balance:.2e}")
    return full


if __name__ == "__main__":
    run()
