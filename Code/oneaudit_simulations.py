import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import time
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_plurcomp, PGD, convex_uits,\
    banded_uits, brute_force_uits, generate_oneaudit_population

# Notes:
# for modeling San Francisco there are many batches (say 90%) of size 1
# then you would stratify on the batches that are size 1 and the batches that are bigger than 1
# smallest batch larger than 1 would be like size 100
# control the proportion of sizes that are larger than 1 (20% to 90%) where 20% is a state without much vote by mail, 90% is SF
# we could do 10, 25, 50, 75, 90
# Federal limit of 2000 registered voters per precinct;
# batches could be uniformly distributed on a grid from [100, 1000]
# if the population size is set, create batches until all the population is allocated
# or could do a stick breaking thing

# could shrink agrapa towards something like the bernoulli optimal bet
# with an a priori estimate of the mean (e.g. the reported assorter mean)

A_c_global_grid = np.linspace(0.51, 0.75, 20) # global assorter means
delta_grid = [0, 0.5] # maximum spread between batches
polarized_grid = [True, False] # whether or not there is polarization (uniform or clustered batch totals)
num_batch_grid = [1, 2, 10] # the number of batches of size > 1; note that if there is one batch it is equivalent to polling
batch_size_grid = [100, 1000] # assuming for now, equally sized batches
prop_invalid_grid = [0, 0.5] # proportion of invalid votes in each batch (uniform across batches)
num_cvr_grid = [100, 1000, 10000] # number of cvrs
alpha = 0.05 # risk limit
n_reps = 100 # the number of replicate simulations
stratified_grid = [True, False] # whether or not the population and inference will be stratified


bets_dict = {
    "agrapa": lambda x, eta: Bets.agrapa(x, eta, c = 0.99),
    "alpha": "special handling", # see below
    "kelly-optimal": lambda x, eta: Bets.kelly_optimal(x, eta, c = 0.99)}
bets_grid = list(bets_dict.keys())

results = []

for A_c_global, delta, num_batches, batch_size, prop_invalid, bet, num_cvrs, polarized, stratified in itertools.product(A_c_global_grid, delta_grid, num_batch_grid, batch_size_grid, prop_invalid_grid, bets_grid, num_cvr_grid, polarized_grid, stratified_grid):
    u = 1 # upper bound for plurality assorters
    v_global = 2 * A_c_global - 1 # global margin

    # means and sizes for batches
    if polarized:
        assert (num_batches % 2) == 0, "number of batches must be divisible by two to maintain global mean with polarization"
        A_c = np.append(
            (A_c_global - 0.5 * delta) * np.ones(int(num_batches/2)),
            (A_c_global + 0.5 * delta) * np.ones(int(num_batches/2))
        )
    else:
        A_c = np.linspace(A_c_global - 0.5 * delta, A_c_global + 0.5 * delta, num_batches)
    batch_sizes = np.ones(num_batches) * batch_size
    invalids = np.ones(num_batches) * prop_invalid

    # add CVRs
    A_c = np.append(A_c, np.ones(num_cvrs))
    batch_sizes = np.append(batch_sizes, np.ones(num_cvrs))
    invalids = np.append(invalids, np.zeros(num_cvrs))

    assorter_pop_unscaled, batch_labels = generate_oneaudit_population(
        batch_sizes = batch_sizes,
        A_c = A_c,
        invalid = invalids
    )
    eta_0_unscaled = 1/2 # global null mean

    # assorters and global null are rescaled to [0,1] for compatability with functions from utils
    # NB: if stratification is used, may need to rethink rescaling: each stratum needs to be on [0,1] and the global null should still correspond to the assertion
    assorter_pop = assorter_pop_unscaled / (2 * u / (2 * u - v_global))
    eta_0 = eta_0_unscaled / (2 * u / (2 * u - v_global))

    N = len(assorter_pop) # the size of the population/sample
    stopping_times = np.zeros(n_reps) # container for stopping times in each simulation
    run_times = np.zeros(n_reps) # container for run times in each simulation
    if not stratified:
        #derive kelly-optimal bet one time by applying numerical optimization to entire population
        if bet == "alpha":
            # alpha (predictable bernoulli) get shrunk towards the true mean of the population
            bets_dict["alpha"] = lambda x, eta: Bets.predictable_bernoulli(x, eta, c = 0.99, mu_0 = np.mean(assorter_pop))
        if bet == "kelly-optimal":
            ko_bet = Bets.kelly_optimal(assorter_pop, eta_0)
        for r in range(n_reps):
            X = np.random.permutation(assorter_pop) # the sample is a permutation of the population
            # TSMs are computed for sampling WOR
            start_time = time.time()
            if bet == "kelly-optimal":
                m = mart(X, eta = eta_0, lam = ko_bet, N = N, log = True)
            else:
                m = mart(X, eta = eta_0, lam_func = bets_dict[bet], N = N, log = True)
            run_time = start_time - time.time()
            stopping_time = np.where(any(m > -np.log(alpha)), np.argmax(m > -np.log(alpha)), N) # where the TSM crosses 1/alpha, or else the population size
            stopping_times[r] = stopping_time
            run_times[r] = run_time
    else: # we don't compute the stratified p-value if there are no cvrs
        strata = np.where(batch_sizes > 1, 0, 1) # place ballots with CVRs (batch_size == 1) into stratum 1, and larger batches into stratum 0
        K = 2 # the number of strata
        N_strat = np.unique(strata, return_counts = True)[1] # the size of the population in each stratum
        etas = construct_eta_bands(eta_0, N_strat, n_bands = 100) # the null space, partitioned into bands
        for r in range(n_reps):
            # 'draw' a stratified sample by shuffling the population within strata
            X = []
            for k in range(K):
                X.append(np.random.permutation(assorter_pop[strata == k]))
            # TSMs are computed for sampling WOR
            start_time = time.time()

            if bets == "alpha":
                bets_dict["alpha"] = [lambda x, eta: Bets.predictable_bernoulli(x, eta, c = 0.99, mu_0 = np.mean(assorter_pop[k])) for k in range(K)]
            m = banded_uits(
                X,
                N = N_strat,
                etas = etas,
                lam_func = bets_dict[bet],
                allocation_func = Allocations.proportional_round_robin,
                log = True,
                WOR = True)[0]
            run_time = start_time - time.time()
            stopping_time = np.where(any(m > -np.log(alpha)), np.argmax(m > -np.log(alpha)), N) # where the TSM crosses 1/alpha, or else the population size
            stopping_times[r] = stopping_time
            run_times[r] = run_time

    expected_sample_size = np.mean(stopping_times)
    percentile_sample_size = np.percentile(stopping_times, 90)


    data_dict = {
        "A_c_global":A_c_global,
        "delta":delta,
        "num_batches":num_batches,
        "num_cvrs":num_cvrs,
        "prop_cvrs": num_cvrs / len(assorter_pop),
        "batch_size":batch_size,
        "prop_invalid":prop_invalid,
        "bet":str(bet),
        "expected_sample_size":expected_sample_size,
        "90th_percentile_sample_size":percentile_sample_size,
        "run_time":run_time}
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("one_audit_results.csv", index = False)
