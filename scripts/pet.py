import numpy as np
import pandas as pd
import pyet

df = pd.read_csv(
    r"data\raw\timeseries\observation_based\CAMELS_CH_obs_based_2486.csv",
    sep=",",
    parse_dates=["date"],
    index_col="date",
)

lat_rad = 46.46888 * np.pi / 180

pet = pyet.hargreaves(
    tmean=df["temperature_mean(degC)"],
    tmax=df["temperature_max(degC)"],
    tmin=df["temperature_min(degC)"],
    lat=lat_rad,
    method=1,
)
df["pet(mm/d)"] = pet

print(df["pet(mm/d)"].head(15))
print(df["pet(mm/d)"].describe())

df.to_csv(r"data\processed\camels_2486_with_pet.csv")