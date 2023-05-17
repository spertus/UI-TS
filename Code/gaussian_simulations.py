#simulations of UI-NNSM and LCB power under truncated gaussians
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit, construct_vertex_etas,\
    random_truncated_gaussian, PGD, negexp_ui_mart
import time
start_time = time.time()


alpha = 0.05
eta_0 = 1/2
reps = 50

K_grid = [2, 5]
global_mean_grid = np.linspace(0.5, 0.7, 10)
delta_grid = [0, 0.1, 0.2] #maximum spread of the stratum means
sd = 0.05
allocation_grid = ["round_robin","larger_means"]
allocation_mapping = {"round_robin":Allocations.round_robin, "larger_means":Allocations.more_to_larger_means}

results = []
for K, global_mean, delta, allocation in itertools.product(K_grid, global_mean_grid, delta_grid, allocation_grid):
    shifts = np.linspace(-0.5,0.5,K)
    deltas = shifts * delta
    N = [1000/K for _ in range(K)]
    etas = construct_vertex_etas(N = N, eta_0 = eta_0)
    allocation_rule = allocation_mapping[allocation]

    stopping_times_uinnsm_adaptive = np.zeros(reps)
    stopping_times_uinnsm_fixed = np.zeros(reps)
    stopping_times_lcb = np.zeros(reps)
    for r in range(reps):
        x = [random_truncated_gaussian(mean = global_mean + deltas[k], sd = sd, size = N[k]) for k in range(K)]
        #for wright's method
        lcb = global_lower_bound(x, N, Bets.fixed, allocation_rule, alpha = 0.05, WOR = False, breaks = 1000)
        uinnsm_fixed = union_intersection_mart(x, N, etas, Bets.fixed, allocation_rule, WOR = False, combine = "product")[0]
        uinnsm_adaptive = negexp_ui_mart(x, N, allocation_rule)
        stopping_times_lcb[r] = np.where(any(lcb > eta_0), np.argmax(lcb > eta_0), np.sum(N))
        stopping_times_uinnsm_fixed[r] = np.where(any(uinnsm_fixed > -np.log(alpha)), np.argmax(uinnsm_fixed > -np.log(alpha)), np.sum(N))
        stopping_times_uinnsm_adaptive[r] = np.where(any(uinnsm_adaptive > -np.log(alpha)), np.argmax(uinnsm_adaptive > -np.log(alpha)), np.sum(N))
    mean_stop_lcb = np.mean(stopping_times_lcb)
    mean_stop_uinnsm_fixed = np.mean(stopping_times_uinnsm_fixed)
    mean_stop_uinnsm_adaptive = np.mean(stopping_times_uinnsm_adaptive)

    results_dict = {
        "K":K,
        "global_mean":global_mean,
        "delta":delta,
        "sd":sd,
        "allocation_rule":allocation,
        "mean_stop_lcb":mean_stop_lcb,
        "mean_stop_uinnsm_fixed":mean_stop_uinnsm_fixed,
        "mean_stop_uinnsm_adaptive":mean_stop_uinnsm_adaptive
    }
    results.append(results_dict)
results = pd.DataFrame(results)
results.to_csv("full_gaussian_simulation_results.csv", index = False)
print("--- %s seconds ---" % (time.time() - start_time))
