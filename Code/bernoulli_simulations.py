#simulations of Bernoullis within strata

import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import os
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit, PGD, negexp_ui_mart



reps = 1
sim_rep = os.getenv('SLURM_ARRAY_TASK_ID')
np.random.seed(int(sim_rep)) #this sets a different seed for every rep



alt_grid = [0.51, 0.55, 0.6, 0.7]
delta_grid = [0, 0.5]
alpha = 0.05
eta_0 = 0.5

methods_list = ['uinnsm_fisher','uinnsm_product','lcb']
bets_dict = {
    "fixed":Bets.fixed,
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.95),
    "smooth_predictable":Bets.smooth_predictable}
bets_list = ["fixed", "agrapa", "smooth_predictable"]
allocations_dict = {
    "round_robin":Allocations.round_robin,
    "predictable_kelly":Allocations.predictable_kelly,
    "greedy_kelly":Allocations.greedy_kelly}
allocations_list = ["round_robin", "predictable_kelly", "greedy_kelly"]

K = 2
N = [200, 200]
results = []

for alt, delta, method, bet, allocation in itertools.product(alt_grid, delta_grid, methods_list, bets_list, allocations_list):
    means = [alt - 0.5*delta, alt + 0.5*delta]
    calX = [np.array([0, 1]), np.array([0, 1])]
    samples = [np.random.binomial(1, means[0], N[0]), np.random.binomial(1, means[1], N[1])]
    eta_grid, calC, ub_calC = construct_eta_grid(eta_0, calX, N)


    if method == 'lcb':
        min_eta = None
        if bet == 'uniform_mixture' or allocation in ['proportional_to_mart','predictable_kelly','greedy_kelly']:
            stopping_time = None
            sample_size = None
        else:
            lower_bound = global_lower_bound(
                x = samples,
                N = N,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                alpha = alpha,
                breaks = 1000,
                WOR = False)
            stopping_time = np.where(any(lower_bound > eta_0), np.argmax(lower_bound > eta_0), np.sum(N)-1)
            sample_size = stopping_time
    elif method == 'uinnsm_product':
        ui_mart, min_etas, global_ss, T_k = union_intersection_mart(
                    x = samples,
                    N = N,
                    etas = eta_grid,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    combine = "product",
                    log = False,
                    WOR = False)
        pval = np.minimum(1, 1/ui_mart)
        stopping_time = np.where(any(ui_mart > 1/alpha), np.argmax(ui_mart > 1/alpha), np.sum(N)-1)
        min_eta = min_etas[stopping_time]
        sample_size = global_ss[stopping_time]
    elif method == 'uinnsm_fisher':
        pval, min_etas, global_ss, T_k = union_intersection_mart(
                    x = samples,
                    N = N,
                    etas = eta_grid,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    combine = "fisher",
                    log = False,
                    WOR = False)
        stopping_time = np.where(any(pval < alpha), np.argmax(pval < alpha), np.sum(N)-1)
        min_eta = min_etas[stopping_time]
        sample_size = global_ss[stopping_time]
    data_dict = {
        "alt":alt,
        "rep":sim_rep,
        "delta":delta,
        "method":str(method),
        "bet":str(bet),
        "allocation":str(allocation),
        "stopping_time":stopping_time,
        "sample_size":sample_size,
        "worst_case_eta":min_eta}
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("Bernoulli_Results/bernoulli_results_parallel_" + sim_rep + ".csv", index = False)
