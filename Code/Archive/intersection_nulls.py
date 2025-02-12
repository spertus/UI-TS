import math
import pypoman
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
from scipy.stats import bernoulli, uniform, chi2
import numpy as np
from scipy.stats.mstats import gmean
from numpy.testing import assert_allclose
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, construct_vertex_etas, simulate_comparison_audit, random_truncated_gaussian,\
    PGD, negexp_ui_mart


etas = [[0,1], [1,0], [1/2, 1/2]]
alt_grid = [0.51, 0.53, 0.55, 0.58, 0.60, 0.65, 0.68, 0.7, 0.75]
delta_grid = [0, 0.1, 0.5]
alpha = 0.05
eta_0 = 0.5
N = [50, 50]
w = N / np.sum(N)
proj = lambda mu: mu - ((np.dot(w, mu) - eta_0) / np.sum(w**2)) * w
results = []
stopping_times = True #<- change this to False to return the complete martingales, not stopping times

methods_list = ['product']
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



for alt, delta, method, bet, allocation in itertools.product(alt_grid, delta_grid, methods_list, bets_list, allocations_list):
    means = [alt - 0.5*delta, alt + 0.5*delta]
    samples = [np.ones(N[0]) * means[0], np.ones(N[1]) * means[1]]
    #list of etas includes 3 standards and 1 that is a projection of the truth onto the null
    etas = [[0, 1], [1,0], [0.5, 0.5], list(np.round(proj(means), 2))]
    for eta in etas:
        time = np.arange(np.sum(N))
        result = intersection_mart(
                    x = samples,
                    N = N,
                    eta = eta,
                    lam_func = bets_dict[bet],
                    allocation_func = allocations_dict[allocation],
                    combine = method,
                    log = True,
                    WOR = False,
                    return_selections = True)

        if stopping_times:
            pval = result[0] if method == "fisher" else np.minimum(-result[0], 0)
            stopping_time = np.where(any(pval < np.log(alpha)), np.argmax(pval < np.log(alpha)), np.sum(N))
            data_dict = {
                    "eta":str(eta),
                    "alt":alt,
                    "delta":delta,
                    "method":str(method),
                    "bet":str(bet),
                    "allocation":str(allocation),
                    "stopping_time":stopping_time}
            results.append(data_dict)
        else:
            for i in np.arange(result[0].size):
                data_dict = {
                        "eta":str(eta),
                        "alt":alt,
                        "delta":delta,
                        "method":str(method),
                        "bet":str(bet),
                        "allocation":str(allocation),
                        "time":i,
                        "n_1":result[1][i,0],
                        "mart":result[0][i]}
                    #"stopping_time":stopping_time}
                results.append(data_dict)

results = pd.DataFrame(results)
results.to_csv("intersection_null_stopping_times.csv", index = False)
