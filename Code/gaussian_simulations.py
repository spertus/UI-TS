#simulations of UI-NNSM and LCB power under truncated gaussians
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import pypoman
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit, construct_vertex_etas,\
    random_truncated_gaussian, PGD, negexp_uits, construct_eta_bands, banded_uits, brute_force_uits
import time
import os
start_time = time.time()

alpha = 0.05
eta_0 = 1/2
rep_grid = np.arange(10) #allows reps within parallelized simulations
sim_id = os.getenv('SLURM_ARRAY_TASK_ID')
#sim_id = "45"
np.random.seed(int(sim_id)) #this sets a different seed for every rep

method_grid = ['lcb','uits','unstrat']
K_grid = [2]
global_mean_grid = np.linspace(0.5, 0.7, 10)
delta_grid = [0, 0.2] #maximum spread of the stratum means
sd_grid = [0.01, 0.10]

bets_dict = {
    "fixed_predictable":Bets.predictable_plugin,
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.95),
    "smooth_predictable":Bets.negative_exponential}
bets_list = ["fixed_predictable", "agrapa", "smooth_predictable"]
allocations_dict = {
    "round_robin":Allocations.round_robin,
    "predictable_kelly":Allocations.predictable_kelly,
    "greedy_kelly":Allocations.greedy_kelly}
allocations_list = ["round_robin", "greedy_kelly"]



K = 2
w = [1/2, 1/2]
N = np.array([100, 100]) #size of initial sample
N_next = np.array([200, 200]) #size of blocks at which to expand sample
N_max = np.array([2100, 2100]) #maximum size
results = []
eta_bands = construct_eta_bands(eta_0, N = w, n_bands = 100)

for global_mean, sd, delta, rep in itertools.product(global_mean_grid, sd_grid, delta_grid, rep_grid):
    print(str(global_mean))
    sim_rep = sim_id + "_" + str(rep)
    shifts = np.linspace(-0.5,0.5,K)
    deltas = shifts * delta
    #etas = construct_vertex_etas(N = N, eta_0 = eta_0)
    etas = construct_eta_bands(eta_0 = eta_0, N = N, n_bands = 100)
    samples = [random_truncated_gaussian(mean = global_mean + deltas[k], sd = sd, size = N[k]) for k in range(K)]
    for method, bet, allocation in itertools.product(method_grid, bets_list, allocations_list):
        if method == 'lcb':
            min_eta = None
            if allocation in ['proportional_to_mart','predictable_kelly','greedy_kelly']:
                stopping_time = None
                sample_size = None
            else:
                done = False
                while not done:
                    lower_bound = global_lower_bound(
                        x = samples,
                        N = N,
                        lam_func = bets_dict[bet],
                        allocation_func = allocations_dict[allocation],
                        alpha = alpha,
                        breaks = 1000,
                        WOR = False)
                    if any(lower_bound > eta_0):
                        sample_size = stopping_time = np.argmax(lower_bound > eta_0)
                        done = True
                    elif any(N >= N_max):
                        sample_size = stopping_time = np.sum(N)-1
                        done = True
                    else:
                        for k in range(K):
                            samples[k] = np.append(samples[k], random_truncated_gaussian(global_mean + deltas[k], sd, N_next[k]))
                        N = N + N_next
        elif method == 'unstrat':
            if allocation == "round_robin":
                x_unstrat = np.zeros(np.sum(N_max))
                for i in range(np.sum(N_max)):
                    rand_k =  np.random.choice(np.arange(K), size = 1, p = w)
                    x_unstrat[i] = random_truncated_gaussian(mean = global_mean + deltas[rand_k], sd = sd, size = 1)
                unstrat_mart = mart(x_unstrat, eta = eta_0, lam_func = bets_dict[bet], log = True)
                stopping_time = np.where(any(np.exp(unstrat_mart) > 1/alpha), np.argmax(np.exp(unstrat_mart) > 1/alpha), np.sum(N_max))
                sample_size = stopping_time
            else:
                stopping_time = None
                sample_size = None
        elif method == 'uits':
            done = False
            while not done:
                ui_mart, min_etas, global_ss = banded_uits(
                            x = samples,
                            N = N,
                            etas = eta_bands,
                            lam_func = bets_dict[bet],
                            allocation_func = allocations_dict[allocation],
                            log = True,
                            WOR = False)
                if any(ui_mart > np.log(1/alpha)):
                    stopping_time = np.argmax(ui_mart > np.log(1/alpha))
                    min_eta = min_etas[stopping_time]
                    sample_size = global_ss[stopping_time]
                    done = True
                elif any(N >= N_max):
                    stopping_time = np.sum(N)-1
                    min_eta = min_etas[stopping_time]
                    sample_size = global_ss[stopping_time]
                    done = True
                else:
                    for k in range(K):
                        samples[k] = np.append(samples[k], random_truncated_gaussian(global_mean + deltas[k], sd, N_next[k]))
                    N = N + N_next
        data_dict = {
            "K":K,
            "global_mean":global_mean,
            "delta":delta,
            "sd":sd,
            "rep":sim_rep,
            "method":str(method),
            "bet":str(bet),
            "allocation":str(allocation),
            "stopping_time":stopping_time,
            "sample_size":sample_size}
        results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("Gaussian_Results/gaussian_simulation_results_parallel_" + sim_rep + ".csv", index = False)
print("--- %s seconds ---" % (time.time() - start_time))
