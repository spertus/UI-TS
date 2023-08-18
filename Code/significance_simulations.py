#simulate significance levels of stratified samples
#null is true because true mean of samples is below 0.5
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import pypoman
from iteround import saferound
from utils import stratified_t_test, random_truncated_gaussian
import time
import os
start_time = time.time()

# N could be anything because we are sampling from an infinite superpopulation
# but we assume a balanced population, with equal-sized strata
N = [1000, 1000]
n_grid = np.int64(np.round(np.exp(np.linspace(np.log(5),np.log(200),30))))
num_sims = 1000
alpha = 0.05
results = []

for n in n_grid:
    p_vals = np.ones(num_sims)
    for sim in np.arange(num_sims):
        p = 0.01
        mu = 0.505
        sigma = 0.001
        samples = [np.zeros(n), np.zeros(n)]
        #randomly draws a 0 with probability p, or a truncated Gaussian RV with probability 1-p
        for i in np.arange(n):
            for k in np.arange(2):
                if np.random.choice([False, True], size = 1, p = [p, 1-p]):
                    samples[k][i] = random_truncated_gaussian(mu, sigma, 1)
        p_vals[sim] = stratified_t_test(x = samples, eta_0 = 1/2, N = N)
    results_dict = {
        'n':n,
        't_test_level':np.mean(p_vals < alpha)
    }
    results.append(results_dict)
results = pd.DataFrame(results)
results.to_csv("significance_simulation_results.csv", index = False)
print("--- %s seconds ---" % (time.time() - start_time))
