#simulations of UI-NNSM and LCB power under truncated gaussians
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import pypoman
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit, construct_vertex_etas,\
    random_truncated_gaussian, PGD, negexp_ui_mart
import time
import os
start_time = time.time()

alpha = 0.05
eta_0 = 1/2
reps = 1 #sort of deprecated. Repetions now occur in parallel, through SLURM
sim_rep = os.getenv('SLURM_ARRAY_TASK_ID')
np.random.seed(int(sim_rep)) #this sets a different seed for every rep


K_grid = [2]
global_mean_grid = np.linspace(0.5, 0.7, 10)
delta_grid = [0, 0.2] #maximum spread of the stratum means
sd_grid = [0.01, 0.05]
allocation_grid = ["round_robin"]
allocation_mapping = {"round_robin":Allocations.round_robin, "larger_means":Allocations.more_to_larger_means}

results = []
for K, global_mean, delta, sd, allocation in itertools.product(K_grid, global_mean_grid, delta_grid, sd_grid, allocation_grid):
    shifts = np.linspace(-0.5,0.5,K)
    deltas = shifts * delta
    N = [int(1000/K) for _ in range(K)]
    w = N/np.sum(N)
    etas = construct_vertex_etas(N = N, eta_0 = eta_0)
    allocation_rule = allocation_mapping[allocation]

    #this is all set up so that reps can be modified to be larger than 1
    #however I am just going to run everything in parallel
    stopping_times_unstrat_fixed = np.zeros(reps)
    stopping_times_unstrat_agrapa = np.zeros(reps)
    stopping_times_uinnsm_fixed = np.zeros(reps)
    stopping_times_uinnsm_adaptive = np.zeros(reps)
    stopping_times_uinnsm_vertex = np.zeros(reps)
    stopping_times_uinnsm_uniform = np.zeros(reps)
    stopping_times_lcb_fixed = np.zeros(reps)
    stopping_times_lcb_agrapa = np.zeros(reps)
    for r in range(reps):
        x = [random_truncated_gaussian(mean = global_mean + deltas[k], sd = sd, size = N[k]) for k in range(K)]
        #unstratified sample by mixing
        x_unstrat = np.zeros(np.sum(N))
        for i in range(np.sum(N)):
            rand_k =  np.random.choice(np.arange(K), size = 1, p = w)
            x_unstrat[i] = random_truncated_gaussian(mean = global_mean + deltas[rand_k], sd = sd, size = 1)

        unstrat_fixed = mart(x_unstrat, eta = 0.5, lam_func = Bets.fixed, log = True)
        unstrat_agrapa = mart(x_unstrat, eta = 0.5, lam_func = Bets.agrapa, log = True)
        lcb_fixed = global_lower_bound(x, N, Bets.fixed, allocation_rule, alpha = 0.05, WOR = False, breaks = 1000)
        lcb_agrapa = global_lower_bound(x, N, Bets.agrapa, allocation_rule, alpha = 0.05, WOR = False, breaks = 1000)
        uinnsm_fixed = union_intersection_mart(x, N, etas, Bets.fixed, allocation_rule, WOR = False, combine = "product", log = True)[0]
        uinnsm_vertex = union_intersection_mart(x, N, etas, None, allocation_rule, mixture = "vertex", WOR = False, combine = "product", log = True)[0]
        uinnsm_uniform = union_intersection_mart(x, N, etas, None, allocation_rule, mixture = "uniform", WOR = False, combine = "product", log = True)[0]
        uinnsm_adaptive = negexp_ui_mart(x, N, allocation_rule, log = True)

        stopping_times_unstrat_fixed[r] = np.where(any(unstrat_fixed > -np.log(alpha)), np.argmax(unstrat_fixed > -np.log(alpha)), np.sum(N))
        stopping_times_unstrat_agrapa[r] = np.where(any(unstrat_agrapa > -np.log(alpha)), np.argmax(unstrat_agrapa > -np.log(alpha)), np.sum(N))
        stopping_times_lcb_agrapa[r] = np.where(any(lcb_agrapa > eta_0), np.argmax(lcb_agrapa > eta_0), np.sum(N))
        stopping_times_lcb_fixed[r] = np.where(any(lcb_fixed > eta_0), np.argmax(lcb_fixed > eta_0), np.sum(N))
        stopping_times_uinnsm_fixed[r] = np.where(any(uinnsm_fixed > -np.log(alpha)), np.argmax(uinnsm_fixed > -np.log(alpha)), np.sum(N))
        stopping_times_uinnsm_adaptive[r] = np.where(any(uinnsm_adaptive > -np.log(alpha)), np.argmax(uinnsm_adaptive > -np.log(alpha)), np.sum(N))
        stopping_times_uinnsm_vertex[r] = np.where(any(uinnsm_vertex > -np.log(alpha)), np.argmax(uinnsm_vertex > -np.log(alpha)), np.sum(N))
        stopping_times_uinnsm_uniform[r] = np.where(any(uinnsm_uniform > -np.log(alpha)), np.argmax(uinnsm_uniform > -np.log(alpha)), np.sum(N))

    mean_stop_unstrat_fixed = np.mean(stopping_times_unstrat_fixed)
    mean_stop_unstrat_agrapa = np.mean(stopping_times_unstrat_agrapa)
    mean_stop_lcb_fixed = np.mean(stopping_times_lcb_fixed)
    mean_stop_lcb_agrapa = np.mean(stopping_times_lcb_agrapa)
    mean_stop_uinnsm_fixed = np.mean(stopping_times_uinnsm_fixed)
    mean_stop_uinnsm_adaptive = np.mean(stopping_times_uinnsm_adaptive)
    mean_stop_uinnsm_vertex = np.mean(stopping_times_uinnsm_vertex)
    mean_stop_uinnsm_uniform = np.mean(stopping_times_uinnsm_uniform)

    results_dict = {
        "K":K,
        "global_mean":global_mean,
        "delta":delta,
        "sd":sd,
        "rep":sim_rep,
        "allocation_rule":allocation,
        "mean_stop_unstrat_fixed":mean_stop_unstrat_fixed,
        "mean_stop_unstrat_agrapa":mean_stop_unstrat_agrapa,
        "mean_stop_lcb_fixed":mean_stop_lcb_fixed,
        "mean_stop_lcb_agrapa":mean_stop_lcb_agrapa,
        "mean_stop_uinnsm_fixed":mean_stop_uinnsm_fixed,
        "mean_stop_uinnsm_adaptive":mean_stop_uinnsm_adaptive,
        "mean_stop_uinnsm_vertex":mean_stop_uinnsm_vertex,
        "mean_stop_uinnsm_uniform":mean_stop_uinnsm_uniform
    }
    results.append(results_dict)
results = pd.DataFrame(results)
results.to_csv("Gaussian_Results/gaussian_simulation_results_parallel_" + sim_rep + ".csv", index = False)
print("--- %s seconds ---" % (time.time() - start_time))
