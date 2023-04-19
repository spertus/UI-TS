import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pypoman
import itertools
from iteround import saferound
from scipy.stats import bernoulli, multinomial, chi2
from scipy.stats.mstats import gmean


class Bets:
    '''
    Methods to set bets that are eta-adaptive, data-adaptive, both, or neither.
    Currently, bets for stratum k can only depend on eta and data within stratum k

    Parameters
        ----------
        eta: float in [0,1]
            the null mean within stratum k
        x: 1-dimensional np.array of length n_k := T_k(t) with elements in [0,1]
            the data sampled from stratum k

        Returns
        ----------
        lam: a length-1 or length-n_k corresponding to lambda_{ki} in the I-NNSM:
            prod_{i=1}^{T_k(t)} [1 + lambda_{ki (X_{ki} - eta_k)]

    '''

    def fixed(x, eta):
        '''
        lambda fixed to 0.75 (nonadaptive)
        '''
        lam = 0.75 * np.ones(x.size)
        return lam

    def agrapa(x, eta):
        '''
        AGRAPA (approximate-GRAPA) from Section B.3 of Waudby-Smith and Ramdas, 2022
        lambda is set to approximately maximize a Kelly-like objective (expectation of log martingale)
        '''
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)  # 1, 2, 3, ..., len(x)
        mu_hat = S/j
        mj = [x[0]]   # Welford's algorithm for running mean and running SD
        sdj = [0]
        for i, xj in enumerate(x[1:]):
            mj.append(mj[-1]+(xj-mj[-1])/(i+1))
            sdj.append(sdj[-1]+(xj-mj[-2])*(xj-mj[-1]))
        sdj = np.sqrt(sdj/j)
        sdj = np.insert(np.maximum(sdj,.1),0,1)[0:-1]
        #avoid divide by zero errors
        eps = 1e-5
        lam_untrunc = (mu_hat - eta) / (sdj**2 + (mu_hat - eta)**2 + eps)
        lam_trunc = np.maximum(0, np.minimum(lam_untrunc, .75/(eta + eps)))
        return lam_trunc

    def trunc(x, eta):
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)  # 1, 2, 3, ..., len(x)
        mu_hat = S/j
        eps = 1e-5
        lam_trunc = np.where(eta <= mu_hat, .75 / (eta + eps), 0)
        return lam_trunc

    def smooth(x, eta):
        lam = np.exp(-eta)
        return lam

    def smooth_predictable(x, eta):
        lag_mean = np.insert(np.cumsum(x),0,0)[0:-1] / np.arange(1,len(x)+1)
        lam = np.exp(lag_mean - eta)
        return lam

class Allocations:
    '''
    fixed, predictable, and/or eta-adaptive stratum allocation rules
    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            the data sampled from each stratum
        n: length-K list or np.array of ints
            the total sample size in each stratum
        N: length-K list or np.array of ints
            the (population) size of each stratum
        eta: length-K np.array in [0,1]^K
            the vector of null means across strata
        lam_func: callable, a function from the Bets class

    Returns
    ----------
        allocation: a length sum(n_k) sequence of interleaved stratum selections in each round
    '''

    def round_robin(x, running_T_k, n, eta, lam_func):
        exhausted = np.ones(len(n))
        exhausted[running_T_k == n] = np.inf
        next = np.argmin(exhausted * running_T_k)
        return next

    def proportional_round_robin(x, running_T_k, n, eta, lam_func):
        #this is round robin proportional to the sample size (x), not the population size
        exhausted = np.ones(len(n))
        exhausted[running_T_k == n] = np.inf
        next = np.argmin(exhausted * running_T_k / n)
        return next

class Weights:
    '''
    Predictable and eta-adaptive methods to set weights for combining across strata by summing
    Generally will be less powerful than taking products
    (see Vovk and Wang 2020 on E-value combining)

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            the data sampled from each stratum
        eta: length-K np.array in [0,1]^K
            the vector of null means across strata
        func: callable, a function from the Bets class

    Returns
    ----------
        theta: a length-K list of convex weights
            the weights for combining the martingales as a sum I-NNSM E-value at time t

    '''
    def fixed(x, eta, lam_func):
        '''
        balanced, fixed weights (usual average)
        '''
        theta = np.ones(len(eta))/len(eta)
        return theta

    def max_predictable(x, eta, lam_func):
        '''
        puts all weight on the last (lagged) largest within-stratum martingale
        '''
        lag_marts = [np.prod(1 + lam_func(eta[k], x[k][:-1]) * (x[k][:-1] - eta[k])) for k in np.arange(K)]
        theta = np.zeros(len(eta))
        theta[np.argmax(lag_marts)] = 1
        return theta

    def smooth_predictable(x, eta, lam_func):
        '''
        makes weights proportional to last (lagged) size of martingales
        '''
        lag_marts = [np.prod(1 + lam_func(eta[k], x[k][:-1]) * (x[k][:-1] - eta[k])) for k in np.arange(K)]
        theta = lag_marts / np.sum(lag_marts)
        return theta

def mart(x, eta, lam_func, N = np.inf, log = True):
    '''
    betting martingale

    Parameters
    ----------
        x: length-n_k np.array with elements in [0,1]
            data
        eta: scalar in [0,1]
            null mean
        lam_func: callable, a function from the Bets class
        log: Boolean
            indicates whether the martingale should be returned on the log scale or not
    Returns
    ----------
        mart: scalar; the value of the (log) betting martingale at time n_k

    '''
    if N < np.inf:
        S = np.insert(np.cumsum(x), 0, 0)[0:-1] #0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        eta_t = (N*eta-S)/(N-j+1)
    elif N == np.inf:
        eta_t = eta * np.ones(len(x))
    else:
        raise ValueError("Input an integer value for N, possibly np.inf")
    #note: per Waudby-Smith and Ramdas, the bets do not update when sampling WOR
    #note: eta < 0 or eta > 1 can create runtime warnings in log, but are replaced appropriately by inf
    if log:
        mart = np.insert(np.cumsum(np.log(1 + lam_func(x, eta) * (x - eta_t))), 0, 0)
        mart[np.insert(eta_t < 0, 0, False)] = np.inf
        mart[np.insert(eta_t > 1, 0, False)] = -np.inf
    else:
        mart = np.insert(np.cumprod(1 + lam_func(x, eta) * (x - eta_t)), 0, 1)
        mart[np.insert(eta_t < 0, 0, False)] = np.inf
        mart[np.insert(eta_t > 1, 0, False)] = 0
    return mart


def selector(x, N, allocation_func, eta = None, lam_func = None):
    '''
    takes data and predictable tuning parameters and returns a sequence of stratum sample sizes
    equivalent to [T_k(t) for k in 1:K]

    Parameters
    ----------
        x: length-K list of length-N_k np.arrays with elements in [0,1]
            data, may be sample from population or an entire population
            in which case N[k] = len(x[k])
        N: length-K list or np.array of ints
            the population size of each stratum
        eta: length-K np.array with elements in [0,1]
            the intersection null for H_0: mu <= eta, can be None
        lam_func: callable, a function from Bets class
        allocation_func: callable, a function from Allocations class
    Returns
    ----------
        an np.array of length np.sum(N) by
    '''
    w = N/np.sum(N)
    K = len(N)
    n = [len(x_k) for x_k in x]
    #selections from 0 in each stratum; time 1 is first sample
    T_k = np.zeros((np.sum(n) + 1, K), dtype = int)
    running_T_k = np.zeros(K, dtype = int)
    t = 0
    while np.any(running_T_k < n):
        t += 1
        next_k = allocation_func(x, running_T_k, n, eta, lam_func)
        running_T_k[next_k] += 1
        T_k[t,:] = running_T_k
    return T_k

def lower_confidence_bound(x, lam_func, alpha, N = np.inf, breaks = 1000):
    '''
    return a lower confidence bound for a betting martingale\
    given a function to sets bets (lam_func) and a level (1-alpha)

    Parameters
    ----------
        x: length-n_k np.array with elements in [0,1]
            data
        lam_func: callable, a function from the Bets class
        alpha: double in (0,1]
        N: integer
            the size of the stratum when sampling WOR or np.inf when sampling WR
        breaks: int > 0
            the number of equally-spaced breaks in [0,1] on which to compute P-value
    Returns
    ----------
        level (1-alpha) lower confidence bound on the mean
    '''
    grid = np.arange(0, 1 + 1/breaks, step = 1/breaks)
    confset = np.zeros((len(grid), len(x) + 1))
    for i in np.arange(len(grid)):
        m = grid[i]
        confset[i,:] = mart(x, eta = m, lam_func = lam_func, N = N, log = True) < np.log(1/alpha)
    lb = np.zeros(len(x)+1)
    for j in np.arange(len(x)+1):
        lb[j] = grid[np.argmax(confset[:,j])]
    return lb

def global_lower_bound(x, N, lam_func, allocation_func, alpha, WOR = False, breaks = 1000):
    '''
    return a level (1-alpha) lower confidence bound on a global mean\
    computed by using Wright's method, summing lower confidence bounds\
    across strata

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            data from each stratum
        N: length-K list or length-K np.arrays
            the size of each stratum
        lam_func: callable (TODO: allow to differ across strata)
            function from the Bets class
        allocation_func: callable
            function from Allocations class
        alpha: double in (0,1]
            1 minus the confidence level for the lower bound
        WOR: Boolean
            compute the confidence interval for sampling with or without replacement?
        breaks: int > 0
            the number of equally-spaced breaks in [0,1] on which to get each separate confidence bound
    Returns
    ----------
        level (1-alpha) lower confidence bound on the global mean of a stratified population
    '''
    w = N/np.sum(N) #stratum weights
    K = len(N)
    lcbs = []
    for k in np.arange(K):
        N_k = N[k] if WOR else np.inf
        lcbs.append(lower_confidence_bound(
            x = x[k],
            lam_func = lam_func,
            #alpha = 1 - (1 - alpha)**(1/K), #<- if we were using sidak correction (not necessar w evalues)
            alpha = alpha,
            N = N_k,
            breaks = breaks))
    T_k = selector(x, N, allocation_func, eta = None, lam_func = lam_func)
    running_lcbs = np.zeros((T_k.shape[0], K))
    for i in np.arange(T_k.shape[0]):
        running_lcbs[i,:] = np.array([lcbs[k][T_k[i, k]] for k in np.arange(K)])
    global_lcb = np.matmul(running_lcbs, w)
    return global_lcb


def intersection_mart(x, N, eta, lam_func, allocation_func, combine = "product", theta_func = None, log = True, WOR = False):
    '''
    an intersection martingale (I-NNSM) for a vector bs{eta}
    assumes sampling is with replacement: no population size is required

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            the data sampled from each stratum
        N: length-K list of ints
            population size for each stratum
        eta: length-K np.array or list in [0,1]
            the vector of null means
        lam_func: callable, a function from class Bets
        allocation_func: callable, a function from class Allocations
        combine: string, in ["product", "sum", "fisher"]
            how to combine within-stratum martingales to test the intersection null
        WOR: Boolean
            should martingales be computed under sampling with or without replacement?
        theta_func: callable, a function from class Weights
            only relevant if combine == "sum", the weights to use when combining with weighted sum
        log: Boolean
            return the log I-NNSM if true, otherwise return on original scale
    Returns
    ----------
        the value of an intersection martingale that uses all the data (not running max)
    '''
    K = len(eta)

    #compute within-stratum martingales given the sequence of x values
    ws_log = False if combine == "sum" else log
    ws_N = N if WOR else np.inf*np.ones(K)
    ws_marts = np.array([mart(x[k], eta[k], lam_func, ws_N[k], ws_log) for k in np.arange(K)])

    #construct the interleaving
    T_k = selector(x, N, allocation_func, eta = None, lam_func = None)
    marts = np.zeros((T_k.shape[0], K))
    for i in np.arange(T_k.shape[0]):
        marts_i = np.array([ws_marts[k][T_k[i, k]] for k in np.arange(K)])
        #make all marts infinite if one is, when product is taken this enforces rule:
        #we reject intersection null if certainly False in one stratum
        marts[i,:] = marts_i if not any(np.isposinf(marts_i)) else np.inf * np.ones(K)
    if combine == "product":
        int_mart = np.sum(marts, 1) if log else np.prod(marts, 1)
    elif combine == "sum":
        assert theta_func is not None, "Need to specify a theta function from Weights if using sum"
        thetas = theta_func(eta, x, lam_func)
        int_mart = np.log(np.sum(thetas * marts, 1)) if log else np.sum(thetas * marts, 1)
    elif combine == "fisher":
        pvals = np.exp(-np.maximum(0, marts)) if log else 1 / np.maximum(1, marts)
        fisher_stat = -2 * np.sum(np.log(pvals), 1)
        pval = 1 - chi2.cdf(fisher_stat, df = 2*K)
        pval = np.log(pval) if log else pval
    else:
        raise NotImplementedError("combine must be product, sum, or fisher")
    return int_mart if combine != "fisher" else pval

def plot_marts_eta(x, N, lam_func, allocation_func, combine = "product", theta_func = None, log = True, res = 1e-2):
    '''
    generate a 2-D or 3-D plot of an intersection martingale over possible values of bs{eta}
    the global null is always eta <= 1/2; future update: general global nulls

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            the data sampled from each stratum
        N: length-K np.array of positive ints,
            the vector of stratum sizes
        lam_func: callable, a function from class Bets
        combine: string, either "product" or "sum"
            how to combine within-stratum martingales to test the intersection null
        theta_func: callable, a function from class Weights
            only relevant if combine == "sum", the weights to use when combining with weighted sum
        log: Boolean
            return the log I-NNSM if true, otherwise return on original scale
        res: float > 0,
            the resolution of equally-spaced grid to compute and plot the I-NNSM over
    Returns
    ----------
        generates and shows a plot of the last value of an I-NNSM over different values of the null mean
    '''
    K = len(x)

    eta_grid = np.arange(res, 1-res, step=res)
    eta_xs, eta_ys, eta_zs, objs = [], [], [], []
    w = N / np.sum(N)
    if K == 2:
        for eta_x in eta_grid:
            eta_y = (1/2 - w[0] * eta_x) / w[1]
            if eta_y > 1 or eta_x < 0: continue
            obj = intersection_mart(x, N, np.array([eta_x,eta_y]), lam_func, allocation_func, combine, theta_func, log)[-1]
            eta_xs.append(eta_x)
            eta_ys.append(eta_y)
            objs.append(obj)
        plt.plot(eta_xs, objs, linewidth = 1)
        plt.show()
    elif K == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for eta_x in eta_grid:
            for eta_y in eta_grid:
                eta_z = (1/2 - w[0]*eta_x-w[1]*eta_y)/w[2]
                if eta_z > 1 or eta_z < 0: continue
                obj = intersection_mart(x, N, np.array([eta_x,eta_y,eta_z]), lam_func, allocation_func, combine, theta_func, log)[-1]
                eta_xs.append(eta_x)
                eta_ys.append(eta_y)
                eta_zs.append(eta_z)
                objs.append(obj)
        ax.scatter(eta_xs, eta_ys, objs, c = objs)
        ax.view_init(20, 120)
        plt.show()
    else:
        raise NotImplementedError("Can only plot I-NNSM over eta for 2 or 3 strata.")


def construct_eta_grid(eta_0, calX, N):
    '''
    construct a grid of null means for a stratified population\
    representing the null parameter space under a particular global null eta_0.
    Used to compute a UI-NNSM using a brute force strategy\
    that evaluates an I-NNSM at every feasible element of a (discrete) null parameter space.

    Parameters
    ----------
        eta_0: scalar in [0,1]
            the global null
        calX: np.array or length-K list of np.arrays
            the possible population values, overall or within each stratumwise
        N: length-K list of ints
            the size of the population within each stratum
    Returns
    ----------
        a grid of within stratum null means, to be passed into union_intersection_mart
    '''
    #From Mayuri:
        #reduce calX to set of coprime numbers e.g. from [2,3,4,5,6,7,8,9,10] -> [2,3,5,7]
    K = len(N)
    w = N / np.sum(N)
    u_k = np.array([np.max(x) for x in calX])
    l_k = np.array([np.min(x[np.nonzero(x)]) for x in calX])
    #upper bound on how many null means there are
    ub_calC = 1
    for k in np.arange(K):
        ub_calC *= sp.special.comb(N[k] + len(calX[k]) - 1, len(calX[k]) - 1)
    #build stratum-wise means
    means = [[] for _ in range(K)]
    for k in np.arange(K):
        for lst in itertools.combinations_with_replacement(calX[k], r = N[k]):
            if np.mean(lst) not in means[k]:
                means[k].append(np.mean(lst))
    etas = []

    #should there be a factor of K in here
    eps_k = w*(l_k / np.array(N))
    eps = np.max(eps_k)
    for crt_prd in itertools.product(*means):
        if eta_0 - eps <= np.dot(w, crt_prd) <= eta_0:
            etas.append(crt_prd)
    calC = len(etas)
    return etas, calC, ub_calC

def construct_eta_grid_plurcomp(N, A_c):
    '''
    construct all the intersection nulls possible in a comparison audit of a plurality contest

    Parameters
    ----------
        N: a length-K list of ints
            the size of each stratum
        A_c: a length-K np.array of floats
            the reported assorter mean bar{A}_c in each stratum

    Returns
    ----------
        every eta that is possible in a comparison risk-limiting audit\
        given the input diluted margins and stratum sizes
    '''
    w = N/np.sum(N)
    assert np.dot(w, A_c) > 0.5, "global reported margin <= 1/2"
    K = len(N)
    means = []
    for k in np.arange(K):
        means.append(np.arange(0, 1 + 0.5/N[k], step  = 0.5/N[k]))
    etas = []
    #if we can't hit w^T mu = 1/2, what is the largest gap possible?
    #should there be a factor of K
    eps_k = w*(0.5/np.array(N))
    eps = np.max(eps_k)
    for crt_prd in itertools.product(*means):
        if 1/2 - eps <= np.dot(w, crt_prd) <= 1/2:
            #null means as defined in Sweeter than SUITE https://arxiv.org/pdf/2207.03379.pdf
            #but divided by two, to map population from [0,2] to [0,1]
            etas.append(tuple((np.array(crt_prd) + 1 - A_c) / 2))
    calC = len(etas)
    return etas, calC


def union_intersection_mart(x, N, etas, lam_func, allocation_func, combine = "product", theta_func = None, log = True, WOR = False):
    '''
    compute a UI-NNSM by minimizing I-NNSMs by brute force search over feasible eta, passed as etas

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            the data sampled from each stratum
        N: length-K list of ints
            the size of each stratum
        etas: list of length-K np.arrays
            vectors of within-stratum nulls over which the minimum will be taken
        combine: string, either "product" or "sum"
            how to combine within-stratum martingales to test the intersection null
        lam_func: callable, a function from class Bets
            the function for setting the bets (lambda_{ki}) for each stratum / time
        allocation_func: callable, a function from the Allocations class
            function for allocation sample to strata for each eta
        theta_func: callable, a function from class Weights
            only relevant if combine == "sum", the weights to use when combining with weighted sum
        log: Boolean
            return the log UI-NNSM if true, otherwise return on original scale
        WOR: Boolean
            should the intersection martingales be computed under sampling without replacement
    Returns
    ----------
        the value of a union-intersection martingale using all data x
    '''
    #evaluate intersection mart on every eta
    obj = np.zeros((len(etas), np.sum(N) + 1))
    for i in np.arange(len(etas)):
        obj[i,:] = intersection_mart(x, N, etas[i], lam_func, allocation_func, combine, theta_func, log, WOR)
    opt_index = np.argmin(obj, 0) if combine != "fisher" else np.argmax(obj, 0)
    eta_opt = np.zeros((np.sum(N) + 1, len(x)))
    mart_opt = np.zeros(np.sum(N) + 1)
    for i in np.arange(np.sum(N) + 1):
        eta_opt[i,:] = etas[opt_index[i]]
        mart_opt[i] = obj[opt_index[i],i]
    return mart_opt, eta_opt


def simulate_comparison_audit(N, A_c, p_1, p_2, lam_func, allocation_func, alpha = 0.05, combine = "product", WOR = False, reps = 500):
    '''
    repeatedly simulate a comparison audit of a plurality contest
    given reported assorter means and overstatement rates in each stratum

    Parameters
    ----------
        N: a length-K list of ints
            the size of each stratum
        A_c: a length-K np.array of floats
            the reported assorter mean bar{A}_c in each stratum
        p_1: a length-K np.array of floats
            the true rate of 1 vote overstatements in each stratum
        p_2: a length-K np.array of floats
            the true rate of 2 vote overstatements in each stratum
        lam_func: callable, a function from class Bets
            the function for setting the bets (lambda_{ki}) for each stratum / time
        allocation_func: callable, a function from the Allocations class
            function for allocation sample to strata for each eta
        alpha: float in (0,1)
            the significance level of the test
        combine: string, either "product" or "sum"
            how to combine within-stratum martingales to test the intersection null
        WOR: boolean
            should the martingales be computed under sampling without replacement?
        reps: an integer
            the number of simulations of the audit to run
    Returns
    ----------
        a length reps list of stopping times
    '''
    K = len(N)
    etas = construct_eta_grid_plurcomp(N, A_c)[0]
    x = []
    for k in np.arange(K):
        num_errors = [int(n_err) for n_err in saferound([N[k]*p_2[k], N[k]*p_1[k], N[k]*(1-p_2[k]-p_1[k])], places = 0)]
        x.append(1/2 * np.concatenate([np.zeros(num_errors[0]), np.ones(num_errors[1]) * 1/2, np.ones(num_errors[2])]))
    stopping_times = np.zeros(reps)
    for r in np.arange(reps):
        X = [np.random.choice(x[k],  len(x[k]), replace = False) for k in np.arange(K)]
        uinnsm = union_intersection_mart(X, N, etas, lam_func, allocation_func, combine, WOR = WOR)[0]
        stopping_times[r] = np.where(any(uinnsm > 1/alpha), np.argmax(uinnsm > 1/alpha), np.sum(N))
    return stopping_times



############## functions for betting SMG #############
def maximize_bsmg(samples, lam, N, theta = 1/2):
    '''
    maximize a stratified betting supermartingale over possible values of eta (the within-stratum means)

    Parameters
    ----------
    samples: length-K list of np.arrays
        samples from each stratum in random order
    lam: np.array of length K
        the fixed lambda (bet) within each stratum, must be in [0,1]
    N: np.array of length K
        the number of elements in each stratum in the population
    theta: double in [0,1]
        the global null mean

    prng : np.Random.RandomState
        a PRNG (or seed, or none)
    '''
    w = N / np.sum(N)
    K = len(N)
    #define constraint set for pypoman projection
    A = np.concatenate((
        np.expand_dims(w, axis = 0),
        np.expand_dims(-w, axis = 0),
        -np.identity(K),
        np.identity(K)))
    b = np.concatenate((1/2 * np.ones(2), np.zeros(K), np.ones(K)))
    sample_means = np.array([np.mean(x) for x in samples])
    log_mart = lambda eta, k: np.sum(np.log(1 + lam[k] * (samples[k] - eta[k])))
    global_log_mart = lambda eta: np.sum([log_mart(eta, k) for k in np.arange(K)])
    partial = lambda eta, k: -np.sum(lam[k] / (1 + lam[k] * (samples[k] - eta[k])))
    grad = lambda eta: np.array([partial(eta, k) for k in np.arange(K)])
    #proj = lambda eta: np.maximum(0, np.minimum(1, eta - w * (np.dot(w, eta) - theta) / np.sum(w**2)))
    proj = lambda eta: pypoman.projection.project_point_to_polytope(point = eta, ineq = (A, b))
    delta = 1e-3
    eta_l = proj(sample_means)
    step_size = 1
    counter = 0
    while step_size > 1e-20:
        counter += 1
        grad_l = grad(eta_l)
        next_eta = proj(eta_l - delta * grad_l)
        step_size = global_log_mart(eta_l) - global_log_mart(next_eta)
        eta_l = next_eta
    eta_star = eta_l
    log_mart = global_log_mart(eta_star)
    p_value = 1/np.maximum(1, np.exp(log_mart))
    return counter, eta_star, log_mart, p_value



#TO FIX
def stratified_comparison_betting(strata: list, n: np.array, u_A: np.array, A_c: np.array):
    '''
    Stratified comparison audit with betting martingale.
    Given a sample size in each stratum, randomly sample that many ballots and compute the global P-value

    Parameters
    ----------
    samples: length-K list of np.arrays
        samples from each stratum in random order
    lambda: np.array of length K
        the fixed lambda (bet) within each stratum, must be in [0,1]
    N: np.array of length K
        the number of elements in each stratum in the population
    u: np.array of length K
        the upper bound in each stratum
    theta: double in [0,1]
        the global null mean

    prng : np.Random.RandomState
        a PRNG (or seed, or none)
    '''
    #things have to be rescaled to make a betting martingale
    shuffled_strata = [np.random.permutation(strata[k])/u_A[k] for k in np.arange(len(strata))]
    N = np.array([len(stratum) for stratum in strata])
    K = len(strata)

    samples = [shuffled_strata[i][0:(n[i]-1)] for i in np.arange(len(shuffled_strata))]
    eta_star, log_mart, p_value = maximize_bsmg(samples = samples, lam = .9*np.ones(K), N = N, u = u_A, theta = 1/2)
