#simulations of Bernoullis within strata
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import os
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_plurcomp, PGD, convex_uits,\
    banded_uits, brute_force_uits



rep_grid = np.arange(10) #allows reps within parallelized simulations
sim_id = os.getenv('SLURM_ARRAY_TASK_ID')
np.random.seed(int(sim_id)) #this sets a different seed for every rep


alt_grid = np.linspace(0.51, 0.7, 20)
delta_grid = [0.5]
alpha = 0.05
eta_0 = 0.5

methods_list = ['uinnsm_product','lcb']
bets_dict = {
    "fixed_plugin": Bets.predictable_plugin,
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.75),
    "bernoulli":lambda x, eta: Bets.predictable_bernoulli(x, eta, c = 0.75),
    "inverse": lambda x, eta: Bets.inverse_eta(x, eta, u = 0.75)}
bets_list = ["fixed_plugin", "agrapa", "bernoulli", "apriori_bernoulli", "inverse"]
allocations_dict = {
    "round_robin":Allocations.round_robin,
    "predictable_kelly":Allocations.predictable_kelly,
    "greedy_kelly":Allocations.greedy_kelly}
#allocations_list = ["round_robin", "predictable_kelly", "greedy_kelly"]
allocations_list = ["round_robin"]

K = 2
w = [1/2, 1/2]
N = np.array([100, 100]) #size of initial sample
N_next = np.array([100, 100]) #size of blocks at which sample will expand
N_max = np.array([2000, 2000]) #maximum size
results = []
eta_bands = construct_eta_bands(eta_0, N = w, n_bands = 100)

for alt, delta, rep in itertools.product(alt_grid, delta_grid, rep_grid):
    print(str(alt))
    sim_rep = sim_id + "_" + str(rep)
    means = [alt - 0.5*delta, alt + 0.5*delta]
    samples = [np.random.binomial(1, means[k], N[k]) for k in range(K)]
    #ap bernoulli is based on the true means
    bets_dict["apriori_bernoulli"] = [
        lambda x, eta: Bets.apriori_bernoulli(x, eta, mu_0 = means[0]),
        lambda x, eta: Bets.apriori_bernoulli(x, eta, mu_0 = means[1])]
    for method, bet, allocation in itertools.product(methods_list, bets_list, allocations_list):
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
                            samples[k] = np.append(samples[k], np.random.binomial(1, means[k], N_next[k]))
                        N = N + N_next
        elif method == 'uinnsm_product':
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
                        samples[k] = np.append(samples[k], np.random.binomial(1, means[k], N_next[k]))
                    N = N + N_next
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
