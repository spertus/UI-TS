#get stopping times for error-free stratified comparison audits
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit


N = [200, 200]

grand_means = [0.5, 0.51, 0.53, 0.55, 0.57, 0.6, 0.65, 0.7, 0.75]
stratum_gaps = [0.0, 0.5]

bets_dict = {
    "fixed":Bets.fixed,
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.95),
    "smooth_predictable":Bets.smooth_predictable}
bets_list = ["fixed", "agrapa", "smooth_predictable"]
allocations_dict = {
    "round_robin":Allocations.round_robin,
    "larger_means":Allocations.more_to_larger_means,
    "predictable_kelly":Allocations.predictable_kelly,
    "minimax_predictable_kelly":None}
allocations_list = ["round_robin", "predictable_kelly", "minimax_predictable_kelly"]
methods_list = ['lcbs', 'uinnsm_product', 'uinnsm_fisher']


results = []
for grand_mean, gap, method, bet, allocation in itertools.product(grand_means, stratum_gaps, methods_list, bets_list, allocations_list):
    A_c = [grand_mean - 0.5*gap, grand_mean + 0.5*gap]
    #error rates are "0"
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]
    #only need 1 simulation rep unless there is auxilliary randomization
    #reps = 1 if allocation in ["round_robin","predictable_kelly","larger_means","minimax_predictable_kelly"] else 30
    if method == "lcbs":
        if allocation in ["proportional_to_mart","predictable_kelly","minimax_predictable_kelly"]:
            stopping_time, sample_size = [None, None]
        else:
            stopping_time, sample_size = simulate_comparison_audit(
                N, A_c, p_1, p_2,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                method = "lcbs",
                reps = 1,
                WOR = False)
    elif method == "uinnsm_product":
        if allocation == "minimax_predictable_kelly":
            if bet == "smooth_predictable":
                #NOTE: this is currently computed under sampling with replacement
                uinnsm = negexp_ui_mart(x, N, Allocations.predictable_kelly, log = True)
                stopping_time = np.where(any(uinnsm > -np.log(alpha)), np.argmax(uinnsm > -np.log(alpha)), np.sum(N))
                sample_size = stopping_time
            else:
                stopping_time, sample_size = [None, None]
        else:
            stopping_time, sample_size = simulate_comparison_audit(
                N, A_c, p_1, p_2,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                method = "ui-nnsm",
                combine = "product",
                reps = 1,
                WOR = False)
    elif method == "uinnsm_fisher":
        if allocation == "minimax_predictable_kelly":
            stopping_time, sample_size = [None, None]
        else:
            stopping_time, sample_size = simulate_comparison_audit(
                N, A_c, p_1, p_2,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                method = "ui-nnsm",
                combine = "fisher",
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
