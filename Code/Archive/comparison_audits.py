#get stopping times for error-free stratified comparison audits
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_comparison_audit, PGD, negexp_ui_mart


N = [200, 200]
K = len(N)
w = N/np.sum(N)
grand_means = = np.linspace(0.51, 0.75, 30)
stratum_gaps = [0.0, 0.5]
alpha = 0.05

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
methods_list = ['lcbs', 'uinnsm_product']


results = []
for grand_mean, gap, method, bet, allocation in itertools.product(grand_means, stratum_gaps, methods_list, bets_list, allocations_list):
    A_c = [grand_mean - 0.5*gap, grand_mean + 0.5*gap]
    #error rates are "0"
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]
    #only need 1 simulation rep unless there is auxilliary randomization
    #reps = 1 if allocation in ["round_robin","predictable_kelly","larger_means","minimax"] else 30
    if method == "lcbs":
        if allocation in ["proportional_to_mart","predictable_kelly","greedy_kelly"]:
            stopping_time, sample_size = [None, None]
        else:
            stopping_time, sample_size = simulate_comparison_audit(
                N, A_c, p_1, p_2,
                assort_method = "global",
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                method = "lcbs",
                reps = 1,
                alpha = alpha,
                WOR = False)
    elif method == "uinnsm_product":
        # if allocation == "greedy_kelly":
        #     if bet == "smooth_predictable":
        #         #setup here is slightly different since it uses negexp_ui_mart
        #         #in particular, the population needs to be defined explicitly
        #         A_c_global = np.dot(w, A_c)
        #         v = 2 * A_c_global - 1
        #         a = 1/(2-v)
        #         x = []
        #         for k in np.arange(K):
        #             num_errors = [int(n_err) for n_err in saferound([N[k]*p_2[k], N[k]*p_1[k], N[k]*(1-p_2[k]-p_1[k])], places = 0)]
        #             x.append(np.concatenate([np.zeros(num_errors[0]), np.ones(num_errors[1]) * a/2, np.ones(num_errors[2])]) * a)
        #         X = [np.random.choice(x[k],  len(x[k]), replace = True) for k in np.arange(K)]
        #         uinnsm = negexp_ui_mart(X, N, Allocations.predictable_kelly, log = True)[0]
        #         stopping_time = np.where(any(uinnsm > -np.log(alpha)), np.argmax(uinnsm > -np.log(alpha)), np.sum(N))
        #         sample_size = stopping_time
        #     else:
        #         stopping_time, sample_size = [None, None]
        # else:
        stopping_time, sample_size = simulate_comparison_audit(
            N, A_c, p_1, p_2,
            assort_method = "global",
            lam_func = bets_dict[bet],
            allocation_func = allocations_dict[allocation],
            method = "ui-nnsm",
            combine = "product",
            alpha = alpha,
            reps = 1,
            WOR = False)
    elif method == "uinnsm_fisher":
        # if allocation == "minimax":
        #     stopping_time, sample_size = [None, None]
        # else:
        stopping_time, sample_size = simulate_comparison_audit(
            N, A_c, p_1, p_2,
            assort_method = "global",
            lam_func = bets_dict[bet],
            allocation_func = allocations_dict[allocation],
            method = "ui-nnsm",
            combine = "fisher",
            alpha = alpha,
            reps = 1,
            WOR = False)
    data_dict = {
        "A_c":grand_mean,
        "stratum_gap":gap,
        "A_c_1":A_c[0],
        "A_c_2":A_c[1],
        "method":str(method),
        "bet":str(bet),
        "allocation":str(allocation),
        "stopping_time":stopping_time,
        "sample_size":sample_size
    }
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("comparison_audit_results.csv", index = False)
