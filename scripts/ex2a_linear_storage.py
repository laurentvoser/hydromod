import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
from hydromod.models import make_precip, linear_reservoir

K = 0.1
N_STEPS = 100


def run():
    P = make_precip(N_STEPS, p_const=10.0, n_rain=10)
    res = linear_reservoir(P, k=K)

    t = np.arange(N_STEPS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, res["Q"], color="tab:blue", label="Q (mm/step)")
    ax.plot(t, res["S"], color="tab:green", label="S (mm)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Q (mm/step), S (mm)")

    ax_p = ax.twinx()
    ax_p.bar(t, P, color="tab:cyan", alpha=0.5, width=1, label="P (mm/step)")
    ax_p.invert_yaxis()
    ax_p.set_ylabel("P (mm/step)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_p.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title(f"2a Linear storage (flower pot), k = {K}")
    fig.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    fig.savefig(Path("plots") / "ex2a_linear_storage.png", dpi=150)
    plt.close(fig)

    print(f"2a: k={K}, max Q = {res['Q'].max():.3f} mm/step "
          f"at step {res['Q'].argmax()}, S_end = {res['S_end']:.3f} mm")
    return res


if __name__ == "__main__":
    run()
