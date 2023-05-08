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
    random_truncated_gaussian
import time
start_time = time.time()


alpha = 0.05
eta_0 = 1/2
reps = 1

K_grid = [2, 3, 4, 5]
global_mean_grid = np.linspace(0.5, 0.7, 10)
delta_grid = [0, 0.1, 0.2] #maximum spread of the stratum means
sd_grid = [0.01, 0.05, 0.1]

results = []
for K, global_mean, delta, sd in itertools.product(K_grid, global_mean_grid, delta_grid, sd_grid):
    shifts = np.linspace(-0.5,0.5,K)
    deltas = shifts * delta
    N = [1000 for _ in range(K)]
    etas = construct_vertex_etas(N = N, eta_0 = eta_0)

    stopping_times_uinnsm = np.zeros(reps)
    stopping_times_lcb = np.zeros(reps)
    for r in range(reps):
        x = [random_truncated_gaussian(mean = global_mean + deltas[k], sd = sd, size = N[k]) for k in range(K)]
        #for wright's method
        lcb = global_lower_bound(x, N, Bets.fixed, Allocations.round_robin, alpha = 0.05, WOR = False, breaks = 1000)
        uinnsm = union_intersection_mart(x, N, etas, Bets.fixed, Allocations.round_robin, WOR = False, combine = "product")[0]
        stopping_times_lcb[r] = np.where(any(lcb > eta_0), np.argmax(lcb > eta_0), np.sum(N))
        stopping_times_uinnsm[r] = np.where(any(uinnsm > -np.log(alpha)), np.argmax(uinnsm > -np.log(alpha)), np.sum(N))
    mean_stop_lcb = np.mean(stopping_times_lcb)
    mean_stop_uinnsm = np.mean(stopping_times_uinnsm)
    percentile_stop_lcb = np.quantile(stopping_times_lcb, 0.9)
    percentile_stop_uinnsm = np.quantile(stopping_times_uinnsm, 0.9)

    results_dict = {
        "K":K,
        "global_mean":global_mean,
        "delta":delta,
        "sd":sd,
        "mean_stop_lcb":mean_stop_lcb,
        "percentile90_stop_lcb":percentile_stop_lcb,
        "mean_stop_uinnsm":mean_stop_uinnsm,
        "percentile90_stop_uinnsm":percentile_stop_uinnsm
    }
    results.append(results_dict)
results = pd.DataFrame(results)
results.to_csv("gaussian_simulation_results.csv", index = False)
print("--- %s seconds ---" % (time.time() - start_time))
