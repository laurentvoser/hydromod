import numpy as np
import pandas as pd
import pyet

obs_path = r"data\raw\timeseries\observation_based\CAMELS_CH_obs_based_2486.csv"
sim_path = r"data\raw\timeseries\simulation_based\CAMELS_CH_sim_based_2486.csv"

for path in (obs_path, sim_path):
    with open(path) as f:
        print(path)
        print(f.readline())

df_obs = pd.read_csv(obs_path, sep=",", parse_dates=["date"], index_col="date")
df_sim = pd.read_csv(sim_path, sep=",", parse_dates=["date"], index_col="date")

print("obs columns:", df_obs.columns.tolist())
print("sim columns:", df_sim.columns.tolist())

# Merge on the date index; suffix avoids column-name clashes if both share names
df = df_obs.join(df_sim, how="inner", lsuffix="_obs", rsuffix="_sim")


print(df.head(15))
print(df.describe())

df.to_csv(r"data\processed\camels_2486_combined.csv")
print("Saved to data\\processed\\camels_2486_combined.csv")