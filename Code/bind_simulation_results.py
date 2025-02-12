#paste together all the gaussian simulation results on arwen
import pandas as pd
import os


filenames = [filename for filename in os.listdir('.') if filename.startswith("gaussian_simulation_results_parallel_")]
print(filenames)

all_results = pd.read_csv(filenames[0])
for i in range(1, len(filenames)):
    data = pd.read_csv(filenames[i])
    all_results = pd.concat([all_results, data])
all_results.to_csv("all_gaussian_simulations.csv")
