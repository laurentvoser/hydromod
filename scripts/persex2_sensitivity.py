import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.persex2_snow_model import (load, run_model, score, day_of_water_year,
                            DEFAULTS, COL_SWE, SPINUP_END)

GRID = {
    "ddf": [2, 3, 4, 5, 6],
    "t_snow": [-1, 0, 1, 2, 3],
    "width": [0, 1, 2, 4],
    "t_melt": [-1, 0, 1],
}
LABELS = {"ddf": "DDF (mm/\u00b0C/d)", "t_snow": "50 % snow temperature (\u00b0C)",
          "width": "Transition width (\u00b0C)", "t_melt": "Melt threshold (\u00b0C)"}


def metrics(df, sim):
    d = df.assign(dowy=day_of_water_year(df), swe=sim["swe"], snow=sim["snowfall"])
    d = d[d["water_year"] > SPINUP_END]
    g = d.groupby("water_year")
    max_swe = g["swe"].max()
    date_max = g["swe"].apply(lambda s: d.loc[s.idxmax(), "dowy"])
    snow_days = g["swe"].apply(lambda s: (s > 0).sum())
    rmse, corr = score(df, sim)
    return {"mean_max_swe_mm": max_swe.mean(),
            "mean_day_of_max_swe": date_max.mean(),
            "mean_snow_days": snow_days.mean(),
            "mean_annual_snowfall_mm": g["snow"].sum().mean(),
            "rmse_mm": rmse, "corr": corr}


def run():
    df = load()
    rows = []
    for name, values in GRID.items():
        for v in values:
            sim = run_model(df, **{name: v})
            rows.append({"parameter": name, "value": v, **metrics(df, sim)})
    table = pd.DataFrame(rows)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    table.to_csv(Path("data/processed") / "persex2_sensitivity_one_at_a_time.csv", index=False)
    print(table.round(2).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    d0 = day_of_water_year(df)
    month_starts = pd.to_datetime(
        ["2001-10-01", "2002-01-01", "2002-04-01", "2002-07-01"])
    ticks = (month_starts - pd.Timestamp("2001-10-01")).days + 1
    for ax, (name, values) in zip(axes.ravel(), GRID.items()):
        cmap = plt.cm.viridis(np.linspace(0, 1, len(values)))
        for c, v in zip(cmap, values):
            sim = run_model(df, **{name: v})
            d = df.assign(dowy=d0, swe=sim["swe"])
            d = d[(d["water_year"] > SPINUP_END) & (d["dowy"] <= 365)]
            clim = d.groupby("dowy")["swe"].mean()
            ax.plot(clim.index, clim.values, color=c, label=str(v))
        obs = df.assign(dowy=d0)
        obs = obs[(obs["water_year"] > SPINUP_END) & (obs["dowy"] <= 365)]
        ax.plot(obs.groupby("dowy")[COL_SWE].mean(), color="k", linestyle="--",
                linewidth=1, label="obs")
        ax.set_xticks(ticks)
        ax.set_xticklabels([m.strftime("%b") for m in month_starts])
        ax.set_title(LABELS[name], fontsize=10)
        ax.legend(fontsize=7, title="value", title_fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean SWE (mm)")
    fig.suptitle("Sensitivity: mean annual SWE cycle (others at defaults)")
    fig.tight_layout()
    fig.savefig(Path("plots") / "persex2_sensitivity_curves.png", dpi=150)
    plt.close(fig)

    ddfs, ts = GRID["ddf"], GRID["t_snow"]
    mat = np.zeros((len(ddfs), len(ts)))
    for i, a in enumerate(ddfs):
        for j, b in enumerate(ts):
            sim = run_model(df, ddf=a, t_snow=b)
            mat[i, j] = score(df, sim)[0]
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(mat, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(ts)), labels=ts)
    ax.set_yticks(range(len(ddfs)), labels=ddfs)
    ax.set_xlabel(LABELS["t_snow"])
    ax.set_ylabel(LABELS["ddf"])
    for i in range(len(ddfs)):
        for j in range(len(ts)):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
    fig2.colorbar(im, ax=ax, label="RMSE vs CAMELS-CH SWE (mm)")
    ax.set_title("RMSE: DDF vs rain/snow temperature")
    fig2.tight_layout()
    fig2.savefig(Path("plots") / "persex2_sensitivity_rmse_heatmap.png", dpi=150)
    plt.close(fig2)
    return table


if __name__ == "__main__":
    run()
