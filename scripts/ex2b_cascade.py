import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hydromod.models import make_precip, cascade

K_VALUES = [0.1, 0.4, 0.7]
N_STEPS = 100


def run():
    P = make_precip(N_STEPS, p_const=10.0, n_rain=10)

    table = pd.DataFrame(index=pd.Index(K_VALUES, name="k1"),
                         columns=pd.Index(K_VALUES, name="k2"), dtype=float)
    for k1 in K_VALUES:
        for k2 in K_VALUES:
            _, r2 = cascade(P, k1, k2)
            table.loc[k1, k2] = r2["Q"].max()

    print("2b: max discharge of reservoir 2 (mm/step)")
    print(table.round(3))

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    table.to_csv(Path("data/processed") / "ex2b_max_discharge.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(table.values, origin="lower", cmap="viridis")
    ax.set_xticks(range(3), labels=K_VALUES)
    ax.set_yticks(range(3), labels=K_VALUES)
    ax.set_xlabel("k2")
    ax.set_ylabel("k1")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{table.values[i, j]:.2f}", ha="center",
                    va="center", color="white")
    fig.colorbar(im, ax=ax, label="max Q2 (mm/step)")
    ax.set_title("2b Cascade: max discharge")
    fig.tight_layout()
    fig.savefig(Path("plots") / "ex2b_cascade_heatmap.png", dpi=150)
    plt.close(fig)

    t = np.arange(N_STEPS)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for k1 in K_VALUES:
        for k2 in K_VALUES:
            _, r2 = cascade(P, k1, k2)
            ax2.plot(t, r2["Q"], label=f"k1={k1}, k2={k2}", linewidth=1)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Q2 (mm/step)")
    ax2.legend(ncol=3, fontsize=7)
    ax2.set_title("2b Cascade: outflow of reservoir 2")
    fig2.tight_layout()
    fig2.savefig(Path("plots") / "ex2b_cascade_hydrographs.png", dpi=150)
    plt.close(fig2)
    return table


if __name__ == "__main__":
    run()
