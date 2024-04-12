#simulations of UI-NNSM and LCB power under truncated gaussians
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import pypoman
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit, construct_vertex_etas,\
    random_truncated_gaussian, PGD, negexp_ui_mart, construct_eta_bands, banded_uitsm
import time
import os
start_time = time.time()

alpha = 0.05
eta_0 = 1/2
rep_grid = np.arange(10) #allows reps within parallelized simulations
sim_id = os.getenv('SLURM_ARRAY_TASK_ID')
#sim_id = "45"
np.random.seed(int(sim_id)) #this sets a different seed for every rep

method_grid = ['lcb','uitsm','unstrat']
K_grid = [2]
global_mean_grid = np.linspace(0.5, 0.7, 10)
delta_grid = [0, 0.2] #maximum spread of the stratum means
sd_grid = [0.01, 0.05, 0.10]

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


results = []
for K, global_mean, delta, sd, method, allocation, bet, rep in itertools.product(K_grid, global_mean_grid, delta_grid, sd_grid, method_grid, allocations_list, bets_list, rep_grid):
    sim_rep = sim_id + "_" + str(rep)
    shifts = np.linspace(-0.5,0.5,K)
    deltas = shifts * delta
    N = [int(400/K) for _ in range(K)]
    w = N/np.sum(N)
    #etas = construct_vertex_etas(N = N, eta_0 = eta_0)
    etas = construct_eta_bands(eta_0 = eta_0, N = N, points = 100)

    x = [random_truncated_gaussian(mean = global_mean + deltas[k], sd = sd, size = N[k]) for k in range(K)]

    if method == 'lcb':
        min_eta = None
        if bet == 'uniform_mixture' or allocation in ['proportional_to_mart','predictable_kelly','greedy_kelly']:
            stopping_time = None
            sample_size = None
        else:
            lower_bound = global_lower_bound(
                x = x,
                N = N,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                alpha = alpha,
                breaks = 1000,
                WOR = False)
            stopping_time = np.where(any(lower_bound > eta_0), np.argmax(lower_bound > eta_0), np.sum(N))
            sample_size = stopping_time
    elif method == 'unstrat':
        if allocation == "round_robin":
            x_unstrat = np.zeros(np.sum(N))
            for i in range(np.sum(N)):
                rand_k =  np.random.choice(np.arange(K), size = 1, p = w)
                x_unstrat[i] = random_truncated_gaussian(mean = global_mean + deltas[rand_k], sd = sd, size = 1)

            unstrat_mart = mart(x_unstrat, eta = eta_0, lam_func = bets_dict[bet], log = True)
            stopping_time = np.where(any(np.exp(unstrat_mart) > 1/alpha), np.argmax(np.exp(unstrat_mart) > 1/alpha), np.sum(N))
            sample_size = stopping_time
        else:
            stopping_time = None
            sample_size = None
    else:
        ui_mart, min_etas, global_ss = banded_uitsm(
                    x = x,
                    N = N,
                    etas = etas,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    log = True,
                    WOR = False)
        stopping_time = np.where(any(np.exp(ui_mart) > 1/alpha), np.argmax(np.exp(ui_mart) > 1/alpha), np.sum(N))
        min_eta = min_etas[stopping_time]
        sample_size = global_ss[stopping_time]

    results_dict = {
        "K":K,
        "global_mean":global_mean,
        "delta":delta,
        "sd":sd,
        "rep":sim_rep,
        "method":str(method),
        "bet":str(bet),
        "allocation":str(allocation),
        "stopping_time":stopping_time,
        "sample_size":sample_size
    }
    results.append(results_dict)
results = pd.DataFrame(results)
results.to_csv("Gaussian_Results/gaussian_simulation_results_parallel_" + sim_rep + ".csv", index = False)
print("--- %s seconds ---" % (time.time() - start_time))
