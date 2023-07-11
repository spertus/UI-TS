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
    "proportional_to_mart":Allocations.proportional_to_mart,
    "predictable_kelly":Allocations.predictable_kelly}
allocations_list = ["round_robin", "larger_means", "proportional_to_mart", "predictable_kelly"]
methods_list = ['lcbs', 'ui-nnsm']


results = []
for grand_mean, gap, method, bet, allocation in itertools.product(grand_means, stratum_gaps, methods_list, bets_list, allocations_list):
    A_c = [grand_mean - 0.5*gap, grand_mean + 0.5*gap]
    #error rates are "0"
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]
    reps = 1 if allocation == "round_robin" else 20
    if method == "lcbs":
        if allocation in ["proportional_to_mart","predictable_kelly"]:
            stopping_time = None
        else:
            stopping_time = simulate_comparison_audit(
                N, A_c, p_1, p_2,
                lam_func = bets_dict[bet],
                allocation_func = allocations_dict[allocation],
                method = "lcbs",
                reps = reps,
                WOR = True)
    elif method == "ui-nnsm":
        stopping_time = simulate_comparison_audit(
            N, A_c, p_1, p_2,
            lam_func = bets_dict[bet],
            allocation_func = allocations_dict[allocation],
            method = "ui-nnsm",
            combine = "product",
            reps = reps,
            WOR = True)
    data_dict = {
        "A_c":grand_mean,
        "stratum_gap":gap,
        "A_c_1":A_c[0],
        "A_c_2":A_c[1],
        "method":str(method),
        "bet":str(bet),
        "allocation":str(allocation),
        "stopping_time":stopping_time
    }
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("comparison_audit_results.csv", index = False)
