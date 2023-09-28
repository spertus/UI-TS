# union-intersection test of predictable Kelly vs round robin; save minimizing etas
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit



alt_grid = np.linspace(0.51, 0.75, 30)
delta_grid = [0, 0.1, 0.5]
alpha = 0.05
eta_0 = 0.5

methods_list = ['uinnsm_fisher','uinnsm_product']
bets_dict = {
    "fixed":Bets.fixed,
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.95),
    "smooth_predictable":Bets.smooth_predictable}
bets_list = ["fixed", "agrapa", "smooth_predictable"]
allocations_dict = {
    "round_robin":Allocations.round_robin,
    "larger_means":Allocations.more_to_larger_means,
    "predictable_kelly":Allocations.predictable_kelly}
allocations_list = ["round_robin", "larger_means", "predictable_kelly"]

K = 2
N = [200, 200]
results = []

for alt, delta, method, bet, allocation in itertools.product(alt_grid, delta_grid, methods_list, bets_list, allocations_list):
    means = [alt - 0.5*delta, alt + 0.5*delta]
    calX = [np.array([0, means[0]]),np.array([0, means[1]])]
    samples = [np.ones(N[0]) * means[0], np.ones(N[1]) * means[1]]
    eta_grid, calC, ub_calC = construct_eta_grid(eta_0, calX, N)

    if method == 'lcb':
        min_eta = None
        if bet == 'uniform_mixture' or allocation in ['proportional_to_mart','predictable_kelly']:
            stopping_time = None
        else:
            lower_bound = global_lower_bound(
                x = samples,
                N = N,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                alpha = alpha,
                breaks = 1000,
                WOR = True)
            stopping_time = np.where(any(lower_bound > eta_0), np.argmax(lower_bound > eta_0), np.sum(N))
    elif method == 'uinnsm_product':
        mart, min_eta = union_intersection_mart(
                    x = samples,
                    N = N,
                    etas = eta_grid,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    combine = "product",
                    log = False,
                    WOR = True)
        pval = np.minimum(1, 1/mart)
        #stopping_time = np.where(any(mart > 1/alpha), np.argmax(mart > 1/alpha), np.sum(N))
        #min_eta = np.where(any(mart > 1/alpha), min_eta[stopping_time], min_eta[np.sum(N)])
    elif method == 'uinnsm_fisher':
        pval, min_eta = union_intersection_mart(
                    x = samples,
                    N = N,
                    etas = eta_grid,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    combine = "fisher",
                    log = False,
                    WOR = True)

        #stopping_time = np.where(any(mart < alpha), np.argmax(mart > alpha), np.sum(N))
        #min_eta = np.where(any(mart < alpha), min_eta[stopping_time], min_eta[np.sum(N)])
    #instead of recording stopping times, we record the p-value at every time
    for i in range(pval.shape[0]):
        data_dict = {
            "alt":alt,
            "delta":delta,
            "method":str(method),
            "bet":str(bet),
            "allocation":str(allocation),
            "time": i + 1,
            "pval": pval[i],
            "min_eta": min_eta[i,:]
            }
        #"stopping_time":stopping_time,
        #"worst_case_eta":min_eta}
        results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("point_mass_results.csv", index = False)
