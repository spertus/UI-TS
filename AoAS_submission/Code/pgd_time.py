import pandas as pd
import itertools
import random
import numpy as np
import time
#from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_plurcomp, PGD, convex_uits,\
    banded_uits



N_k_grid = [10, 50, 100] #number of samples from each stratum
K_grid = [2,3,5,10,50] #number of strata
alpha = 0.05
eta_0 = 0.5
mu = 0.6

methods_list = ['uinnsm_product', 'lcb']
bet = Bets.inverse_eta
allocation = Allocations.round_robin
results = []

for N_k, K in itertools.product(N_k_grid, K_grid):
    print("N_k = " + str(N_k) + "; K = " + str(K))
    N = [N_k] * K
    means = [mu] * K
    samples = [np.ones(N[k]) * means[k] for k in range(K)]

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
        "N_k":N_k,
        "K":K,
        "method":"lcb",
        "bet":"inverse",
        "allocation":"round_robin",
        "stopping_time":stopping_time,
        "sample_size":stopping_time,
        "run_time":run_time_lcb}
    results.append(data_dict)

    start_time = time.time()
    ui_mart, min_etas, T_k = convex_uits(
                x = samples,
                N = N,
                allocation_func = allocation,
                log = True)
    run_time_uits = time.time() - start_time
    stopping_time = np.where(any(np.exp(ui_mart) > 1/alpha), np.argmax(np.exp(ui_mart) > 1/alpha), np.sum(N))
    data_dict = {
        "alt":mu,
        "N_k":N_k,
        "K":K,
        "method":"uits",
        "bet":"inverse",
        "allocation":"round_robin",
        "stopping_time":stopping_time,
        "sample_size":stopping_time,
        "run_time":run_time_uits}
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("pgd_run_times_short.csv", index = False)
