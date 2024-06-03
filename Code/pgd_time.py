import pandas as pd
import itertools
import random
import numpy as np
import time
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_comparison_audit, PGD, negexp_uits,\
    banded_uitsm



#N_k_grid = [10, 50, 100, 200, 500, 1000] #number of samples from each stratum
#K_grid = [2,3,5,10,50] #number of strata
N_k_grid = [10]
K_grid = [2]
alpha = 0.05
eta_0 = 0.5
mu = 0.6

methods_list = ['uinnsm_product', 'lcb']
bet = Bets.smooth_predictable
allocation = Allocations.round_robin
results = []

for N_k, K in itertools.product(N_k_grid, K_grid):
    N = [N_k] * K
    means = [mu] * K
    samples = [np.ones(N[0]) * means[0], np.ones(N[1]) * means[1]]

    start_time = time.time()
    lower_bound = global_lower_bound(
        x = samples,
        N = N,
        lam_func = bet,
        allocation_func = allocation,
        alpha = alpha,
        breaks = 1000,
        WOR = False)
    run_time_lcb = time.time() - start_time
    stopping_time = np.where(any(lower_bound > eta_0), np.argmax(lower_bound > eta_0), np.sum(N))
    sample_size = stopping_time
    data_dict = {
        "alt":mu,
        "method":"lcb",
        "bet":"smooth_predictable",
        "allocation":"round_robin",
        "stopping_time":stopping_time,
        "sample_size":stopping_time,
        "run_time":run_time_lcb}
    results.append(data_dict)

    start_time = time.time()
    ui_mart, min_etas, T_k = negexp_uits(
                x = samples,
                N = N,
                allocation_func = allocation,
                log = True)
    run_time_uits = time.time() - start_time
    stopping_time = np.where(any(np.exp(ui_mart) > 1/alpha), np.argmax(np.exp(ui_mart) > 1/alpha), np.sum(N))
    data_dict = {
        "alt":mu,
        "method":"uits",
        "bet":"smooth_predictable",
        "allocation":"round_robin",
        "stopping_time":stopping_time,
        "sample_size":stopping_time,
        "run_time":run_time_uits}
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("pgd_run_times.csv", index = False)
