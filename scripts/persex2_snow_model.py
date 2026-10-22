import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hydromod.snow import snow_model

DATA = r"data\processed\camels_2486_combined.csv"
COL_P = "precipitation(mm/d)"
COL_T = "temperature_mean(degC)"
COL_SWE = "swe(mm)"

DEFAULTS = dict(ddf=1.0, t_snow=1.0, width=2.0, t_melt=0.0, swe0=0.0)
WATER_YEARS_PLOT = [2017, 2018, 2019]   # winters shown in the time series plot
SPINUP_END = 1981                       # water years <= this are excluded from scores


def load():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    df["water_year"] = df.index.year.where(df.index.month < 10, df.index.year + 1)
    df[COL_P] = df[COL_P].fillna(0.0)
    df[COL_T] = df[COL_T].interpolate(limit_direction="both")
    return df


def run_model(df, **params):
    p = {**DEFAULTS, **params}
    res = snow_model(df[COL_P].values, df[COL_T].values, **p)
    return pd.DataFrame(res, index=df.index)


def day_of_water_year(df):
    start = pd.to_datetime((df["water_year"] - 1).astype(str) + "-10-01")
    return ((df.index.values - start.values) / np.timedelta64(1, "D")).astype(int) + 1


def score(df, sim):
    mask = df[COL_SWE].notna() & (df["water_year"] > SPINUP_END)
    obs, s = df.loc[mask, COL_SWE], sim.loc[mask, "swe"]
    rmse = float(np.sqrt(((obs - s) ** 2).mean()))
    corr = float(np.corrcoef(obs, s)[0, 1])
    return rmse, corr


def run():
    df = load()
    sim = run_model(df)
    rmse, corr = score(df, sim)
    print(f"Defaults {DEFAULTS}")
    print(f"RMSE vs observed SWE: {rmse:.1f} mm | correlation: {corr:.2f}")
    print(f"Max simulated SWE: {sim['swe'].max():.0f} mm on {sim['swe'].idxmax().date()}")

    Path("plots").mkdir(exist_ok=True)

    wy = df["water_year"].isin(WATER_YEARS_PLOT)
    d, s = df[wy], sim[wy]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(s.index, s["swe"], color="tab:blue", label="SWE simulated")
    axes[0].plot(d.index, d[COL_SWE], color="k", linestyle="--", linewidth=1,
                 label="SWE (CAMELS-CH)")
    axes[0].set_ylabel("SWE (mm)")
    axes[0].legend()
    axes[0].set_title("Degree-day snow model, catchment 2486")

    axes[1].bar(s.index, s["snowfall"], color="tab:cyan", width=1, label="Snowfall")
    axes[1].bar(s.index, -s["melt"], color="tab:red", width=1, label="Melt")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_ylabel("mm/d")
    axes[1].legend()
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(Path("plots") / "persex2_snow_timeseries.png", dpi=150)
    plt.close(fig)

    dowy = day_of_water_year(df)
    full = df.assign(dowy=dowy, sim=sim["swe"])
    full = full[full["water_year"] > SPINUP_END]
    full = full[full["dowy"] <= 365]
    clim = full.groupby("dowy")[["sim", COL_SWE]].mean()
    fig2, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(clim.index, clim["sim"], color="tab:blue", label="Simulated")
    ax.plot(clim.index, clim[COL_SWE], color="k", linestyle="--", label="CAMELS-CH")
    month_starts = pd.to_datetime(
        ["2001-10-01", "2001-11-01", "2001-12-01", "2002-01-01", "2002-02-01",
         "2002-03-01", "2002-04-01", "2002-05-01", "2002-06-01", "2002-07-01",
         "2002-08-01", "2002-09-01"])
    ticks = (month_starts - pd.Timestamp("2001-10-01")).days + 1
    ax.set_xticks(ticks)
    ax.set_xticklabels([m.strftime("%b") for m in month_starts])
    ax.set_ylabel("Mean SWE (mm)")
    ax.set_title("Mean annual SWE cycle (water years)")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(Path("plots") / "persex2_snow_climatology.png", dpi=150)
    plt.close(fig2)
    return df, sim


if __name__ == "__main__":
    run()
