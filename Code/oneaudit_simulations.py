import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import time
import os
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_plurcomp, PGD, convex_uits,\
    banded_uits, brute_force_uits, generate_oneaudit_population, generate_hybrid_audit_population,\
    construct_eta_bands_hybrid

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


rep_grid = np.arange(50) #allows reps within parallelized simulations
n_reps = len(rep_grid)
sim_id = os.getenv('SLURM_ARRAY_TASK_ID')
np.random.seed(int(sim_id)) #this sets a different seed for every rep

#A_c_bar_grid = np.linspace(0.51, 0.75, 5) # global assorter means
A_c_bar_grid = [0.51, 0.55, 0.6] # these are the attempted global margins, the actual global margin may differ slightly because of integer rounding of votes
delta_across_grid = [0, 0.5] # controls the spread between the mean for CVRs and the mean for batches
delta_within_grid = [0, 0.5] # controls the spread between batches
polarized_grid = [True, False] # whether or not there is polarization (uniform or clustered batch totals)
num_batch_ballots = 20000
batch_size_grid = [10, 100, 1000, 20000] # assuming for now, equally sized batches
ratio_cvrs_grid = [1] # the ratio of the size of the CVR stratum to the batches
prop_invalid_grid = [0.0, 0.1] # proportion of invalid votes in each batch (uniform across batches)
alpha = 0.05 # risk limit



n_next = 500 #size of blocks at which sample will expand
n_max = 40000 # maximum size of sample, at which point the simulation will terminate


bets_dict = {
    "cobra": "special handling", # see below
    "agrapa": lambda x, eta: Bets.agrapa(x, eta, c = 0.99),
    "alpha": "special handling",
    "kelly-optimal": "special handling",
    "universal-portfolio": "special handling"
    }
bets_grid = list(bets_dict.keys())

results = []
i = 0

for A_c_bar, delta_within, delta_across, prop_invalid, bet, ratio_cvrs, batch_size, polarized in itertools.product(A_c_bar_grid, delta_within_grid, delta_across_grid, prop_invalid_grid, bets_grid, ratio_cvrs_grid, batch_size_grid, polarized_grid):
    i += 1
    print(f'A_c: {A_c_bar}, delta_w: {delta_within}, delta_a: {delta_across}, prop_invalid: {prop_invalid}, bet: {bet}, ratio_cvrs: {ratio_cvrs}, batch_size: {batch_size}, polarized: {polarized}')
    u = 1 # upper bound for plurality assorters
    assert (num_batch_ballots % batch_size) == 0, "number of batch ballots is not divisible by the number of batches"
    num_batches = int(num_batch_ballots / batch_size)
    num_cvrs = int(np.round(ratio_cvrs * num_batch_ballots))
    N = num_batch_ballots + num_cvrs # population size
    prop_cvrs = num_cvrs / N # proportion of votes that are CVRs
    prop_batches = 1 - prop_cvrs # proportion of votes that are in batches

    # means and sizes for batches
    # A_c_bar = prop_batches * A_c_bar_batches + prop_cvrs * A_c_bar_cvrs
    A_c_bar_batches = A_c_bar - 0.5 * delta_across # overall batch mean
    if num_batches == 1:
        if polarized:
            continue
        else:
            A_c_batches = A_c_bar_batches
    elif polarized:
        if delta_within == 0:
            continue
        assert (num_batches % 2) == 0, "number of batches must be divisible by two to maintain bar mean with polarization"
        A_c_batches = np.append(
            (A_c_bar_batches - 0.5 * delta_within) * np.ones(int(num_batches/2)),
            (A_c_bar_batches + 0.5 * delta_within) * np.ones(int(num_batches/2))
        )
    else:
        A_c_batches = np.linspace(
            A_c_bar_batches - 0.5 * delta_within,
            A_c_bar_batches + 0.5 * delta_within,
            num_batches)
    batch_sizes = np.ones(num_batches) * batch_size
    invalids = np.ones(num_batches) * prop_invalid


    # make CVRs
    prop_invalid_cvrs = prop_invalid
    A_c_bar_cvrs = A_c_bar + 0.5 * delta_across
    assert 0 <= A_c_bar_cvrs - prop_invalid_cvrs/2 <= (1 - prop_invalid_cvrs), "the CVR group mean is not attainable with this number of invalids"
    cvrs_i = num_cvrs * prop_invalid_cvrs # the number of CVRs showing invalid votes
    cvrs_w = num_cvrs * (A_c_bar_cvrs - prop_invalid_cvrs/2) # the number of CVRs showing votes for the winner
    cvrs_l = num_cvrs * (1 - A_c_bar_cvrs - prop_invalid_cvrs/2) # the number of CVRs showing votes for the winner
    cvrs_iwl = [int(c) for c in saferound([cvrs_i, cvrs_w, cvrs_l], places = 0)] # rounding to integers
    A_c_cvrs = np.repeat([1/2, 1, 0], cvrs_iwl)

    # add CVRs
    A_c = np.append(A_c_batches, A_c_cvrs)
    batch_sizes = np.append(batch_sizes, np.ones(num_cvrs))
    invalids = np.append(invalids, np.append(np.ones(cvrs_iwl[0]), np.zeros(cvrs_iwl[1] + cvrs_iwl[2])))

    stopping_times = np.zeros(n_reps) # container for stopping times in each simulation
    run_times = np.zeros(n_reps) # container for run times in each simulation

    # create population of ONEAudit assorters
    assorter_pop_unscaled = generate_oneaudit_population(
            batch_sizes = batch_sizes,
            A_c = A_c,
            invalid = invalids
        )
    eta_0_unscaled = 1/2 # global null mean

    v_bar = 2 * A_c_bar - 1 # global margin

    # assorters and global null are rescaled to [0,1]
    assorter_pop = assorter_pop_unscaled / (2 * u / (2 * u - v_bar))
    eta_0 = eta_0_unscaled / (2 * u / (2 * u - v_bar))

    #derive kelly-optimal bet one time by applying numerical optimization to entire population
    if bet == "alpha":
        # alpha (predictable bernoulli) get shrunk towards the true mean of the population
        bets_dict["alpha"] = lambda x, eta: Bets.predictable_bernoulli(x, eta, c = 0.99, mu_0 = np.mean(assorter_pop))
    if bet == "kelly-optimal":
        ko_bet = Bets.kelly_optimal(assorter_pop, eta_0)
    if bet == "cobra":
        bets_dict["cobra"] = lambda x, eta: Bets.cobra(x, eta, A_c = A_c_bar)
    if bet == "universal-portfolio":
        # this does not actually compute the universal portfolio bet
        # it computes a discrete mixture wealth strategy that approximates the wealth under the universal universal_portfolio
        # it seems to be both faster and more numerically stable than computing the actual universal portfolio
        # see Cover 1991, Lemma 2.5 (https://isl.stanford.edu/~cover/papers/paper93.pdf)
        bets_dict["universal-portfolio"] = [(lambda x, eta, c=b: Bets.fixed(x, eta, c=c)) for b in np.linspace(0.05, 1/eta_0-0.05, 100)]

    for r in rep_grid:
        # containers for expanding samples
        selected = np.array([], dtype = np.int32) # the index of samples that have been selected
        remaining = np.arange(N) # the index of values remaining the population
        done = False
        start_time = time.time()
        while not done:
            selected = np.append(selected, np.random.choice(remaining, size = n_next, replace = False))
            remaining = np.setdiff1d(remaining, selected)
            X = assorter_pop[selected]
            # TSMs are computed for sampling WOR
            if bet == "kelly-optimal":
                m = mart(X, eta = eta_0, lam = ko_bet[0:len(X)], N = N, log = True)
            else:
                m = mart(X, eta = eta_0, lam_func = bets_dict[bet], N = N, log = True)
            if any(m > -np.log(alpha)) or (len(X) >= n_max):
                done = True
        stopping_time = np.where(any(m > -np.log(alpha)), np.argmax(m > -np.log(alpha)), n_max) # where the TSM crosses 1/alpha, or else the population size
        run_time = time.time() - start_time
        data_dict = {
            "A_c_bar":A_c_bar,
            "delta_within":delta_within,
            "delta_across":delta_across,
            "assort_method":"ONE",
            "polarized":polarized,
            "rep":r,
            "num_batches":num_batches,
            "num_cvrs":num_cvrs,
            "prop_cvrs":prop_cvrs,
            "batch_size":batch_size,
            "prop_invalid":prop_invalid,
            "bet":str(bet),
            "sample_size": stopping_time,
            "run_time":run_time}
        results.append(data_dict)
        print(f'run_time: {run_time}, stopping_time:{stopping_time}, rep: {r}, A_c: {A_c_bar}, delta_w: {delta_within}, delta_a: {delta_across}, prop_invalid: {prop_invalid}, bet: {bet}, ratio_cvrs: {ratio_cvrs}, batch_size: {batch_size}, polarized: {polarized}')

results = pd.DataFrame(results)
results.to_csv("sims/oneaudit_betting_results_parallel_" + sim_id + ".csv", index = False)
