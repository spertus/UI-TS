# union-intersection test of predictable Kelly vs round robin; save minimizing etas
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import time
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_comparison_audit, PGD, negexp_uits,\
    banded_uits, brute_force_uits



alt_grid = np.linspace(0.51, 0.75, 20)
#alt_grid = [0.505, 0.51, 0.52, 0.53, 0.55, 0.6, 0.65, 0.7, 0.75]
delta_grid = [0, 0.5]
alpha = 0.05
eta_0 = 0.5
n_bands_grid = [1, 3, 10, 100, 500]

methods_list = ['uinnsm_product', 'lcb']
bets_dict = {
    "fixed_predictable":Bets.predictable_plugin,
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.95),
    "smooth_predictable":lambda x, eta: Bets.negative_exponential(x, eta, c = 1)}
bets_list = ["fixed_predictable", "agrapa", "smooth_predictable"]
allocations_dict = {
    "round_robin":Allocations.round_robin,
    "predictable_kelly":Allocations.predictable_kelly,
    "greedy_kelly":Allocations.greedy_kelly}
allocations_list = ["round_robin", "predictable_kelly", "greedy_kelly"]

#points = 100
K = 2
N = [200, 200]
results = []

for alt, delta, method, bet, allocation, n_bands in itertools.product(alt_grid, delta_grid, methods_list, bets_list, allocations_list, n_bands_grid):
    means = [alt - 0.5*delta, alt + 0.5*delta]
    samples = [np.ones(N[0]) * means[0], np.ones(N[1]) * means[1]]

    #calX = [np.array([0, means[0], 1]),np.array([0, means[1], 1])]
    #eta_grid, calC, ub_calC = construct_eta_grid(eta_0, calX, N)
    eta_bands = construct_eta_bands(eta_0 = eta_0, N = N, n_bands = n_bands)

    if method == 'lcb':
        min_eta = None
        if bet == 'uniform_mixture' or allocation in ['proportional_to_mart','predictable_kelly','greedy_kelly']:
            stopping_time = None
            sample_size = None
        else:
            start_time = time.time()
            lower_bound = global_lower_bound(
                x = samples,
                N = N,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                alpha = alpha,
                breaks = 1000,
                WOR = False)
            run_time = start_time - time.time()
            stopping_time = np.where(any(lower_bound > eta_0), np.argmax(lower_bound > eta_0), np.sum(N))
            sample_size = stopping_time
    else:
        start_time = time.time()
        ui_mart, min_etas, global_ss = banded_uits(
                    x = samples,
                    N = N,
                    etas = eta_bands,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    log = True,
                    WOR = False)
        run_time = start_time - time.time()
        pval = np.minimum(1, 1/np.exp(ui_mart))
        stopping_time = np.where(any(np.exp(ui_mart) > 1/alpha), np.argmax(np.exp(ui_mart) > 1/alpha), np.sum(N))
        min_eta = min_etas[stopping_time]
        sample_size = global_ss[stopping_time]
    data_dict = {
        "alt":alt,
        "n_bands":n_bands,
        "delta":delta,
        "method":str(method),
        "bet":str(bet),
        "allocation":str(allocation),
        "stopping_time":stopping_time,
        "sample_size":sample_size,
        "run_time":run_time,
        "worst_case_eta":min_eta}
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("point_mass_results.csv", index = False)
