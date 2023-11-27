import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pypoman
import itertools
from iteround import saferound
from scipy.stats import bernoulli, multinomial, chi2, t
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
        kwargs: additional arguments specific to each strategy
        Returns
        ----------
        lam: a length-1 or length-n_k corresponding to lambda_{ki} in the I-NNSM:
            prod_{i=1}^{T_k(t)} [1 + lambda_{ki (X_{ki} - eta_k)]

    '''
    # def __init__(self, x: np.array=None, eta: float=None, **kwargs):
    #    self.x = x
    #    self.eta = eta
    #    self.kwargs = kwargs

    #work in progress: rewrite classes to define and inherit arguments from init
    #use kwargs for additional arguments specific to each method
    def fixed(x, eta, **kwargs):
        '''
        lambda fixed to c
        eta-nonadaptive
        ---------------
        kwargs:
            c: the fixed bet size
        '''
        c = kwargs.get('c', 0.75)
        lam = c * np.ones(x.size)
        return lam

    def agrapa(x, eta, **kwargs):
        '''
        AGRAPA (approximate-GRAPA) from Section B.3 of Waudby-Smith and Ramdas, 2022
        lambda is set to approximately maximize a Kelly-like objective (expectation of log martingale)
        eta-adaptive
        -------------
        kwargs:
            sd_min: scalar, the minimum value allowed for the estimated standard deviation
            c: scalar, the maximum value allowed for bets
        '''
        sd_min = kwargs.get('sd_min', 0.01)
        c = kwargs.get('c', 0.75)
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)  # 1, 2, 3, ..., len(x)
        mu_hat = S/j
        mj = [x[0]]   # Welford's algorithm for running mean and running SD
        sdj = [0]
        for i, xj in enumerate(x[1:]):
            mj.append(mj[-1]+(xj-mj[-1])/(i+1))
            sdj.append(sdj[-1]+(xj-mj[-2])*(xj-mj[-1]))
        sdj = np.sqrt(sdj/j)
        sdj = np.insert(np.maximum(sdj,sd_min),0,1)[0:-1]
        #parameterize the truncation of sdj w kwargs and the truncation of the bet?
        #we should allow larger bets, also see truncation below. Maybe set c to be above 0.75
        lam_untrunc = (mu_hat - eta) / (sdj**2 + (mu_hat - eta)**2)
        lam_trunc = np.maximum(0, np.minimum(lam_untrunc, c/eta))
        return lam_trunc

    def trunc(x, eta, **kwargs):
        '''
        doesn't bet if running mean is less than null, otherwise bets inversely proportional to null
        eta-adaptive
        '''
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)  # 1, 2, 3, ..., len(x)
        mu_hat = S/j
        eps = 1e-5
        lam_trunc = np.where(eta <= mu_hat, .75 / (eta + eps), 0)
        return lam_trunc

    def smooth(x, eta, **kwargs):
        '''
        eta-adaptive; smooth, negative exponential in eta (doesn't use data)
        '''
        lam = np.exp(-eta)
        return lam

    def smooth_predictable(x, eta, **kwargs):
        '''
        eta-adaptive; smooth, bets more the higher the empirical mean is above the null mean
        '''
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
            the total sample size in each stratum, i.e. the length of each x
        N: length-K list or np.array of ints
            the (population) size of each stratum
        eta: length-K np.array in [0,1]^K
            the vector of null means across strata
        lam_func: callable, a function from the Bets class

    Returns
    ----------
        allocation: a length sum(n_k) sequence of interleaved stratum selections in each round
    '''

    def round_robin(x, running_T_k, n, N, eta, lam_func):
        #eta-nonadaptive round robin strategy
        exhausted = np.ones(len(n))
        exhausted[running_T_k == n] = np.inf
        next = np.argmin(exhausted * running_T_k)
        return next

    def proportional_round_robin(x, running_T_k, n, N, eta, lam_func):
        #eta-nonadaptive round robin strategy, proportional to total sample size
        exhausted = np.ones(len(n))
        exhausted[running_T_k == n] = np.inf
        next = np.argmin(exhausted * running_T_k / n)
        return next

    def more_to_larger_means(x, running_T_k, n, N, eta, lam_func, **kwargs):
        #eta-nonadaptive
        #samples more from strata with larger values of x on average
        #does round robin until every stratum has been sampled once
        if any(running_T_k == 0):
            next = Allocations.round_robin(x, running_T_k, n, N, eta, lam_func)
        else:
            K = len(x)
            eps = kwargs.get("eps", 0.01)
            sd_min = kwargs.get("sd_min", 0.05)
            #UCB-like algorithm targeting the largest stratum mean
            past_x = [x[k][0:running_T_k[k]] for k in range(K)]
            means = np.array([np.mean(px) for px in past_x])
            std_errors = np.array([np.maximum(np.std(px), sd_min) for px in past_x]) / np.sqrt(running_T_k)
            ucbs = means + 2 * std_errors
            scores = np.where(running_T_k == n, -np.inf, ucbs)
            next = np.argmax(scores)
        return next

    def neyman(x, running_T_k, n, N, eta, lam_func, **kwargs):
        #eta-adaptive
        #uses a predictable Neyman allocation to set allocation probabilities
        #see Neyman (1934)
        if any(running_T_k <= 2):
            #use round robin until we have at least 2 samples from each stratum
            next = Allocations.round_robin(x, running_T_k, n, N, eta, lam_func)
        else:
            K = len(x)
            eps = kwargs.get("eps", 0.01) #lower bound on sd
            sds = np.array([np.std(x[k][0:running_T_k[k]]) for k in range(K)]) + eps
            sds = np.where(running_T_k == n, 0, sds)
            neyman_weights = N * sds
            probs = neyman_weights / np.sum(neyman_weights)
            next = np.random.choice(np.arange(K), size = 1, p = probs)
        return next

    def proportional_to_mart(x, running_T_k, n, N, eta, lam_func, **kwargs):
        #eta-adaptive strategy, based on size of martingale for given intersection null
        #this function involves alot of overhead, may want to restructure
        if any(running_T_k <= 1):
            next = Allocations.round_robin(x, running_T_k, n, N, eta, lam_func)
        K = len(x)
        marts = np.array([mart(x[k], eta[k], lam_func, N[k], log = False)[running_T_k[k]] for k in range(K)])
        scores = np.minimum(np.maximum(marts, 1), 1e3)
        scores = np.where(running_T_k == n, 0, scores) #if the stratum is exhausted, its score is 0
        probs = scores / np.sum(scores)
        next = np.random.choice(np.arange(K), size = 1, p = probs)
        return next

    def predictable_kelly(x, running_T_k, n, N, eta, lam_func, **kwargs):
        #this estimates the expected log-growth of each martingale
        #and then draws with probability proportional to this growth
        #currently, can't use randomized betting rules (would need to pass in terms directly)
        if any(running_T_k <= 2):
            next = Allocations.round_robin(x, running_T_k, n, N, eta, lam_func)
        else:
            K = len(x)
            eps = kwargs.get("eps", 0.01)
            sd_min = kwargs.get("sd_min", 0.05)
            #return past terms for each stratum on log scale
            #compute martingale as if sampling were with replacement (N = np.inf)
            past_terms = [mart(x[k], eta[k], lam_func, np.inf, True, True)[0:running_T_k[k]] for k in range(K)]

            #use a UCB-like approach to select next stratum
            est_log_growth = np.array([np.mean(t) for t in past_terms])
            se_log_growth = np.array([np.maximum(np.std(pt), sd_min) for pt in past_terms]) / np.sqrt(running_T_k)
            ucbs_log_growth = est_log_growth + 2 * se_log_growth
            scores = np.where(running_T_k == n, -np.inf, ucbs_log_growth)
            next = np.argmax(scores)
        return next

class Weights:
    '''
    Methods to set weights for combining across strata by summing
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
    def fixed(eta):
        '''
        balanced, fixed weights (usual average)
        '''
        theta = np.ones(len(eta))/len(eta)
        return theta



def mart(x, eta, lam_func, N = np.inf, log = True, terms = False):
    '''
    betting martingale

    Parameters
    ----------
        x: length-n_k np.array with elements in [0,1]
            data
        eta: scalar in [0,1]
            null mean
        lam_func: callable, a function from the Bets class
        N: positive integer,
            the size of the population from which x is drawn (x could be the entire population)
        log: Boolean
            indicates whether the martingale should be returned on the log scale or not
        terms: Boolean
            indicates whether to return the martingale or just the sequence of terms (without multiplying them)
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
    if terms:
        if log:
            terms = np.insert(np.log(1 + lam_func(x, eta_t) * (x - eta_t)), 0, 0)
            terms[np.insert(eta_t < 0, 0, False)] = np.inf
            terms[np.insert(eta_t > 1, 0, False)] = -np.inf
        else:
            terms = np.insert(np.log(1 + lam_func(x, eta_t) * (x - eta_t)), 0, 0)
            terms[np.insert(eta_t < 0, 0, False)] = np.inf
            terms[np.insert(eta_t > 1, 0, False)] = 0
        return terms
    else:
        if log:
            mart = np.insert(np.cumsum(np.log(1 + lam_func(x, eta_t) * (x - eta_t))), 0, 0)
            mart[np.insert(eta_t < 0, 0, False)] = np.inf
            mart[np.insert(eta_t > 1, 0, False)] = -np.inf
        else:
            mart = np.insert(np.cumprod(1 + lam_func(x, eta_t) * (x - eta_t)), 0, 1)
            mart[np.insert(eta_t < 0, 0, False)] = np.inf
            mart[np.insert(eta_t > 1, 0, False)] = 0
        return mart


def selector(x, N, allocation_func, eta = None, lam_func = None, for_samples = False):
    '''
    takes data and predictable tuning parameters and returns a sequence of stratum sample sizes
    equivalent to [T_k(t) for k in 1:K]

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            data, may be sample from population or an entire population
            in which case N[k] = len(x[k])
        N: length-K list or np.array of ints
            the population size of each stratum
        eta: length-K np.array with elements in [0,1]
            the intersection null for H_0: mu <= eta, can be None
        lam_func: callable, a function from Bets class
        allocation_func: callable, a function from Allocations class
        for_samples: boolean
            this is only True when used in negexp_ui_mart,
            which needs an interleaving of _samples_ not martingales,
            and needs a slightly different indexing (one shorter)
    Returns
    ----------
        an np.array of length np.sum(N) by
    '''
    w = N/np.sum(N)
    K = len(N)
    if for_samples:
        n = [len(x_k)-1 for x_k in x]
    else:
        n = [len(x_k) for x_k in x]
    #selections from 0 in each stratum; time 1 is first sample
    T_k = np.zeros((np.sum(n) + 1, K), dtype = int)
    running_T_k = np.zeros(K, dtype = int)
    t = 0
    while np.any(running_T_k < n):
        t += 1
        next_k = allocation_func(x, running_T_k, n, N, eta, lam_func)
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
            #alpha = 1 - (1 - alpha)**(1/K), <- if we were using sidak correction (not necessary w evalues)
            alpha = alpha,
            N = N_k,
            breaks = breaks))
    T_k = selector(x, N, allocation_func, eta = None, lam_func = lam_func)
    running_lcbs = np.zeros((T_k.shape[0], K))
    for i in np.arange(T_k.shape[0]):
        running_lcbs[i,:] = np.array([lcbs[k][T_k[i, k]] for k in np.arange(K)])
    global_lcb = np.matmul(running_lcbs, w)
    return global_lcb


def intersection_mart(x, N, eta, lam_func = None, mixing_dist = None, allocation_func = Allocations.proportional_round_robin, combine = "product",  theta_func = None, log = True, WOR = False, return_selections = False):
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
        mixing_dist: a B by K np.array in [0,1]
            lambdas to mix over, B is just any positive integer (the size of the mixing distribution)
        theta_func: callable, a function from class Weights
            only relevant if combine == "sum", the weights to use when combining with weighted sum
        log: boolean
            return the log I-NNSM if true, otherwise return on original scale
        WOR: boolean
            should martingales be computed under sampling with or without replacement?
        return_selections: boolean
            return matrix of stratum sample sizes (T_k) if True, otherwise just return combined martingale
    Returns
    ----------
        the value of an intersection martingale that uses all the data (not running max)
    '''
    K = len(eta)
    assert (lam_func is None) or (mixing_dist is None), "Cannot specify both a mixing distribution and a predictable lambda function"
    assert (lam_func is not None) or ((combine == "product") and (mixing_dist is not None)), "Currently, mixing distribution can only be used with product combining"
    ws_log = False if combine == "sum" else log
    ws_N = N if WOR else np.inf*np.ones(K)

    if lam_func is not None:
        #compute within-stratum martingales using a predictable lambda sequence
        ws_marts = [mart(x[k], eta[k], lam_func, ws_N[k], ws_log) for k in np.arange(K)]
        #construct the interleaving
        T_k = selector(x, N, allocation_func, eta, lam_func)
        marts = np.zeros((T_k.shape[0], K))
        for i in np.arange(T_k.shape[0]):
            marts_i = np.array([ws_marts[k][T_k[i, k]] for k in np.arange(K)])
            #make all marts infinite if one is, when product is taken this enforces rule:
            #we reject intersection null if certainly False in one stratum
            marts[i,:] = marts_i if not any(np.isposinf(marts_i)) else np.inf * np.ones(K)
    elif mixing_dist is not None:
        B = mixing_dist.shape[0]
        T_k = selector(x, N, allocation_func, eta, lam_func = Bets.fixed) #this supplants a fixed bet to construct martingales in Allocations.prop_to_mart
        marts = np.zeros((B, T_k.shape[0], K))
        for b in range(B):
            ws_marts = [mart(x[k], eta[k], lambda x, eta: Bets.fixed(x, eta, c = mixing_dist[b,k]), ws_N[k], log = False) for k in np.arange(K)]
            for i in np.arange(T_k.shape[0]):
                marts_bi = np.array([ws_marts[k][T_k[i, k]] for k in np.arange(K)])
                marts[b,i,:] = marts_bi if not any(np.isposinf(marts_bi)) else np.inf * np.ones(K)
    if combine == "product":
        if lam_func is not None:
            int_mart = np.sum(marts, 1) if log else np.prod(marts, 1)
        elif mixing_dist is not None:
            #this take the product across strata and the mean across the mixing distribution
            int_mart = np.mean(np.prod(marts, 2), 0)
            int_mart = np.log(int_mart) if log else int_mart
    elif combine == "sum":
        assert theta_func is not None, "Need to specify a theta function from Weights if using sum"
        thetas = theta_func(eta)
        int_mart = np.log(np.sum(thetas * marts, 1)) if log else np.sum(thetas * marts, 1)
    elif combine == "fisher":
        pvals = np.exp(-np.maximum(0, marts)) if log else 1 / np.maximum(1, marts)
        fisher_stat = -2 * np.sum(np.log(pvals), 1)
        pval = 1 - chi2.cdf(fisher_stat, df = 2*K)
        pval = np.log(pval) if log else pval
    else:
        raise NotImplementedError("combine must be product, sum, or fisher")
    result = int_mart if combine != 'fisher' else pval
    if return_selections:
        return result, T_k
    else:
        return result


def plot_marts_eta(x, N, lam_func = None, mixture = None, allocation_func = Allocations.proportional_round_robin, combine = "product", theta_func = None, log = True, res = 1e-2):
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
        mixture: either "vertex" or "uniform"; cannot specify if lam_func is not None
        combine: string, either "product" or "sum"
            how to combine within-stratum martingales to test the intersection null
        allocation_func: callable, a function from class Allocations
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
    if mixture == "vertex":
        vertices = construct_vertex_etas(1/2, N)
        mixing_dist = 1 - np.array(vertices)
    elif mixture == "uniform":
        lam_grids = K * [np.linspace(0.01,0.99,5)]
        mixing_dist = np.array(list(itertools.product(*lam_grids)))
    elif mixture is None:
        mixing_dist = None
    else:
        stop("Specify a valid mixture method; either uniform or vertex")
    eta_grid = np.arange(res, 1-res, step=res)
    eta_xs, eta_ys, eta_zs, objs = [], [], [], []
    w = N / np.sum(N)
    if K == 2:
        for eta_x in eta_grid:
            eta_y = (1/2 - w[0] * eta_x) / w[1]
            if eta_y > 1 or eta_x < 0: continue
            obj = intersection_mart(x, N, np.array([eta_x,eta_y]), lam_func, mixing_dist, allocation_func, combine, theta_func, log)[-1]
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
                obj = intersection_mart(x, N, np.array([eta_x,eta_y,eta_z]), lam_func, mixing_dist, allocation_func, combine, theta_func, log)[-1]
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
    #assert np.dot(w, A_c) > 0.5, "global reported margin <= 1/2"
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

def construct_vertex_etas(eta_0, N):
    '''
    find all the null means that make up vertices of calC.
    Can be used to (relatively) efficiently compute the union-intersection martingale
    IF the betting and allocation strategies are eta-nonadaptive


    Parameters
    ----------
        eta_0: scalar in [0,1]
            the global null
        N: length-K list of ints
            the size of the population within each stratum
    Returns
    ----------
        a list of intersection nulls, to be passed into union_intersection_mart
    '''
    assert len(N) < 16, "Too many strata to compute vertices."
    w = N / np.sum(N)
    K = len(N)
    #define constraint set for pypoman projection
    A = np.concatenate((
        np.expand_dims(w, axis = 0),
        np.expand_dims(-w, axis = 0),
        -np.identity(K),
        np.identity(K)))
    b = np.concatenate((eta_0 * np.ones(2), np.zeros(K), np.ones(K)))
    vertices = np.stack(pypoman.compute_polytope_vertices(A, b), axis = 0)
    etas = vertices[np.matmul(vertices, w) == eta_0,]
    #etas = vertices
    etas = list(map(tuple, etas))
    return etas


def union_intersection_mart(x, N, etas, lam_func = None, allocation_func = Allocations.proportional_round_robin, mixture = None, combine = "product", theta_func = None, log = True, WOR = False, eta_0_mixture = 1/2):
    '''
    compute a UI-NNSM by minimizing I-NNSMs by brute force search over feasible eta, passed as etas

    Parameters
    ----------
        x: length-K list of length-n_k np.arrays with elements in [0,1]
            the data sampled from each stratum
        N: length-K list of ints
            the size of each stratum
        etas: list of length-K tuples
            intersection nulls over which the minimum will be taken
        lam_func: callable, a function from class Bets
            the function for setting the bets (lambda_{ki}) for each stratum / time
        allocation_func: callable, a function from the Allocations class
            function for allocation sample to strata for each eta
        mixture: string or None, either "vertex" or "uniform"
            if present, defines one of two mixing strategies for each I-NNSM and builds mixing_dist
                "vertex": mixes over a discrete uniform distribution on lambda = 1 - etas
                "uniform": mixes over a uniform distribution on [0,1]^K, gridded into 10 equally-spaced points
        combine: string, either "product" or "sum"
            how to combine within-stratum martingales to test the intersection null
        theta_func: callable, a function from class Weights
            only relevant if combine == "sum", the weights to use when combining with weighted sum
        log: Boolean
            return the log UI-NNSM if true, otherwise return on original scale
        WOR: Boolean
            should the intersection martingales be computed under sampling without replacement
        eta_0_mixture: float in [0,1]
            the global null mean, only used to construct the vertices of calC for mixture == "vertex";
            in other cases can be ignored. Defaults to 1/2
    Returns
    ----------
        the value of a union-intersection martingale using all data x
    '''
    assert (lam_func is None) or (mixture is None), "cannot specify both a mixture strategy and predictable lambda function"
    K = len(x)
    w = N / np.sum(N)
    if mixture == "vertex":
        vertices = construct_vertex_etas(eta_0_mixture, N)
        mixing_dist = 1 - np.array(vertices)
    elif mixture == "uniform":
        lam_grids = K * [np.linspace(0.01,0.99,10)]
        mixing_dist = np.array(list(itertools.product(*lam_grids)))
    elif mixture is None:
        mixing_dist = None
    else:
        stop("Specify a valid mixture method; either uniform or vertex")
    #evaluate intersection mart on every eta
    obj = np.zeros((len(etas), np.sum(N) + 1))
    sel = np.zeros((len(etas), np.sum(N) + 1, K))
    for i in np.arange(len(etas)):
        obj[i,:], sel[i,:,:] = intersection_mart(
            x = x,
            N = N,
            eta = etas[i],
            lam_func = lam_func,
            mixing_dist = mixing_dist,
            allocation_func = allocation_func,
            combine = combine,
            theta_func = theta_func,
            log = log,
            WOR = WOR,
            return_selections = True)
    opt_index = np.argmin(obj, 0) if combine != "fisher" else np.argmax(obj, 0)
    eta_opt = np.zeros((np.sum(N) + 1, len(x)))
    mart_opt = np.zeros(np.sum(N) + 1)
    global_sample_size = np.sum(np.max(sel, 0), 1)
    for i in np.arange(np.sum(N) + 1):
        eta_opt[i,:] = etas[opt_index[i]]
        mart_opt[i] = obj[opt_index[i],i]
    return mart_opt, eta_opt, global_sample_size


def simulate_comparison_audit(N, A_c, p_1, p_2, lam_func = None, allocation_func = Allocations.proportional_round_robin, mixture = None, method = "ui-nnsm", combine = "product", alpha = 0.05, WOR = False, reps = 30):
    '''
    simulate (repateadly, if desired) a comparison audit of a plurality contest
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
        mixture: string or None, either "vertex" or "uniform"
            Only works if method == "ui-nnsm"
            if present, defines one of two mixing strategies for each I-NNSM and builds mixing_dist
                "vertex": mixes over a discrete uniform distribution on lambda = 1 - etas
                "uniform": mixes over a uniform distribution on [0,1]^K, gridded into 10 equally-spaced points
        method: string, either "ui-nnsm" or "lcbs"
            the method for testing the global null
            either union-intersection testing or combining lower confidence bounds as in Wright's method
        combine: string, either "product", "sum", or "fisher"
            how to combine within-stratum martingales to test the intersection null
            only relevant when method == "ui-nnsm"
        alpha: float in (0,1)
            the significance level of the test
        WOR: boolean
            should the martingales be computed under sampling without replacement?
        reps: int
            the number of simulations of the audit to run
    Returns
    ----------
        two scalars, the expected stopping time of the audit and the global sample size of the audit;
        these are the same whenever the allocation rule is nonadaptive
    '''
    assert method == "ui-nnsm" or (method == "lcbs" and (mixture is None)), "lcb does not work with mixture"
    K = len(N)
    w = N/np.sum(N)
    A_c_global = np.dot(w, A_c)
    etas = construct_eta_grid_plurcomp(N, A_c)[0]
    x = []
    for k in np.arange(K):
        num_errors = [int(n_err) for n_err in saferound([N[k]*p_2[k], N[k]*p_1[k], N[k]*(1-p_2[k]-p_1[k])], places = 0)]
        x.append(1/2 * np.concatenate([np.zeros(num_errors[0]), np.ones(num_errors[1]) * 1/2, np.ones(num_errors[2])]))
    stopping_times = np.zeros(reps) #container for global stopping times
    sample_sizes = np.zeros(reps) #container for global sample sizes
    for r in np.arange(reps):
        X = [np.random.choice(x[k],  len(x[k]), replace = (not WOR)) for k in np.arange(K)]
        if method == "ui-nnsm":
            uinnsm, eta_min, global_ss = union_intersection_mart(X, N, etas, lam_func, allocation_func, mixture, combine, WOR, log = True)
            if combine == "fisher":
                stopping_times[r] = np.where(any(uinnsm < np.log(alpha)), np.argmax(uinnsm < np.log(alpha)), np.sum(N))
            else:
                stopping_times[r] = np.where(any(uinnsm > -np.log(alpha)), np.argmax(uinnsm > -np.log(alpha)), np.sum(N))
            sample_sizes[r] = global_ss[int(stopping_times[r])]
        elif method == "lcbs":
            eta_0 = (1/2 + 1 - A_c_global)/2 # this is the implied global null mean in the setup described in 3.2 of Sweeter than SUITE
            lcb = global_lower_bound(X, N, lam_func, allocation_func, alpha, breaks = 1000)
            stopping_times[r] = np.where(any(lcb > eta_0), np.argmax(lcb > eta_0), np.sum(N))
            sample_sizes[r] = stopping_times[r]
    return np.mean(stopping_times), np.mean(sample_sizes)



def random_truncated_gaussian(mean, sd, size):
    '''
    simulate from a gaussian truncated to [0,1]

    Parameters
    ----------
        mean: double in [0,1]
        sd: positive double
        size: the number of samples to draw
    Returns
    ----------
        length-size np.array of truncated gaussian draws
    '''
    assert 0 <= mean <= 1, "mean is not in [0,1]"
    samples = np.zeros(size)
    for i in range(size):
        while True:
            draw = np.random.normal(mean, sd, 1)
            if 0 <= draw <= 1:
                samples[i] = draw
                break
    return samples


def t_test(x, eta_0, N = np.inf, WOR = False):
    '''
    run a one-sample, one-sided t-test of the hypothesis that E(x) <= eta_0

    Parameters
    ----------
        x: length-n np.array
            n samples drawn with or without replacement
        eta_0: double
            a hypothesized null mean
        N: int or np.inf
            the size of the population, may be infinite
        WOR: boolean
            was x drawn with or without replacement
    Returns
    ----------
        a P-value for the hypothesis that the mean of the population from which x is drawn is less than eta_0
    '''
    n = x.size[0]
    #currently fpc is not employed so this is conservative for sampling WOR
    fpc = np.sqrt((N - n) / (N-1)) if WOR else 1
    SE = fpc * np.std(x) / np.sqrt(n)
    test_stat = (np.mean(x) - eta_0) / SE
    p_val = 1 - t.cdf(test_stat, n-1)
    return p_val

def stratified_t_test(x, eta_0, N):
    '''
    run a stratified, one-sample, one-sided t-test of the hypothesis that E(x) <= eta_0

    Parameters
    ----------
        x: length-K list of np.arrays
            n_k samples drawn with replacement from each stratum 1 to K
        eta_0: double
            the hypothesized global null mean
        N: length-K list of ints
            the size of each stratum, used to calculate stratum weights
    Returns
    ----------
        a P-value for the hypothesis that the global mean of the population from which x is drawn is less than eta_0
    '''
    n = np.array([x_k.size for x_k in x])
    w = N/np.sum(N)
    g = N * (N - n) / n #used to calculate effective degrees of freedom from Cochran (1977) p. 69
    sample_means = np.array([np.mean(x_k) for x_k in x])
    sample_vars = np.array([np.var(x_k) for x_k in x])
    mean_est = np.sum(w * sample_means)
    #SE_est = np.sqrt((1/np.sum(N)**2) * np.sum(g * sample_vars))
    SE_est = np.sqrt(np.sum((N / np.sum(N))**2 * sample_vars / n))
    #dof = (np.sum(g * sample_vars)**2) / np.sum((g**2 * sample_vars**2) / (n - 1))
    dof = np.min(n - 1) #simpler but conservative (always no larger than above)
    test_stat = (mean_est - eta_0) / SE_est
    p_val = 1 - t.cdf(test_stat, dof)
    return p_val


class PGD:
    '''
    class of helper functions to compute UI-NNSM for negative exponential bets by projected gradient descent
    currently everything is computed on assumption of sampling with replacement
    '''

    def log_mart(samples, past_samples, eta):
        '''
        return the log value of within-stratum martingale evaluated at eta_k
        bets are exponential in negative eta, offset by current sample mean
        '''
        lag_mean = np.mean(past_samples) if past_samples.size > 0 else 1/2
        if samples.size == 0:
            return 0
        else:
            return np.sum(np.log(1 + np.exp(lag_mean - eta) * (samples - eta)))

    def global_log_mart(samples, past_samples, eta):
        '''
        return the log value of the product-combined I-NNSM evaluated at eta
        '''
        return np.sum([PGD.log_mart(samples[k], past_samples[k], eta[k]) for k in np.arange(len(eta))])

    def partial(samples, past_samples, eta):
        '''
        return the partial derivative (WRT eta) of the log I-NNSM evaluated at eta_k
        '''
        lag_mean = np.mean(past_samples) if past_samples.size > 0 else 1/2
        if samples.size == 0:
            return 0
        else:
            return -np.sum(np.exp(lag_mean - eta) * (samples - eta + 1) / (1 + np.exp(lag_mean - eta) * (samples - eta)))

    def grad(samples, past_samples, eta):
        '''
        return the gradient (WRT eta) of the log I-NNSM evaluated at eta
        '''
        return np.array([PGD.partial(samples[k], past_samples[k], eta[k]) for k in np.arange(len(eta))])

#add a nonadaptive allocation rule that chooses the next sample
#based on the Kelly-optimal selection for the current minimimum I-TSM
#bets can still be convexifying / negative exponential
def negexp_ui_mart(x, N, allocation_func, eta_0 = 1/2, log = True):
    '''
    compute the union-intersection NNSM when bets are negative exponential:
    lambda = exp(barX - eta)
    currently only works for sampling with replacement

    Parameters
    ----------
    x: length-K list of np.arrays
        samples from each stratum in random order
    N: np.array of length K
        the number of elements in each stratum in the population
    allocation_func: callable, a function from class Allocations
        the desired allocation strategy. If "eta-adaptive" (e.g. predictable_kelly), the eta used is the
        last minimizing eta, employing a minimax-type selection strategy.
    eta_0: double in [0,1]
        the global null mean
        
    Returns
    --------
    the value of the union-intersection supermartingale
    '''

    w = N / np.sum(N) #stratum weights
    K = len(N) #number of strata
    n = [x[k].shape[0] for k in range(K)]
    #define constraint set for pypoman projection
    A = np.concatenate((
        np.expand_dims(w, axis = 0),
        np.expand_dims(-w, axis = 0),
        -np.identity(K),
        np.identity(K)))
    b = np.concatenate((eta_0 * np.ones(2), np.zeros(K), np.ones(K)))
    proj = lambda eta: pypoman.projection.project_point_to_polytope(point = eta, ineq = (A, b))
    delta = 1e-3 #tuning parameter for optimizer (size of gradient step)

    #this stores the samples available in each stratum at time i = 0,1,2,...,n
    samples_t = [[[] for _ in range(K)] for _ in range(np.sum(n))]
    #initialize with no samples
    uinnsms = [1] #uinnsm starts at 1 at time 0
    samples_t[0] = [np.array([]) for _ in range(K)]
    T_k = np.zeros(K, dtype = int)
    eta_stars = np.zeros((np.sum(n), K))

    for i in np.arange(1, np.sum(n)):
        #select next stratum
        S_i = allocation_func(x, T_k, n, N, eta = eta_stars[i-1], lam_func = Bets.smooth_predictable)
        T_k[S_i] += 1
        for k in np.arange(K):
            samples_t[i][k] = x[k][np.arange(T_k[k])] #update available samples
        #initial estimate of minimum by projecting current sample mean onto null space
        if any(T_k == 0):
            sample_means = [eta_0 for _ in range(K)]
        else:
            sample_means = np.array([np.mean(samples_t[i][k]) for k in range(K)])
        eta_l = proj(np.log(sample_means))
        step_size = 1
        counter = 0
        while step_size > 1e-5:
            counter += 1
            grad_l = PGD.grad(samples_t[i], samples_t[i-1], eta_l)
            next_eta = proj(eta_l - delta * grad_l)
            step_size = PGD.global_log_mart(samples_t[i], samples_t[i-1], eta_l) - PGD.global_log_mart(samples_t[i], samples_t[i-1], next_eta)
            eta_l = next_eta
        eta_stars[i] = eta_l
        log_mart = PGD.global_log_mart(samples_t[i], samples_t[i-1], eta_stars[i])
        if log:
            uinnsms.append(log_mart)
        else:
            uinnsms.append(np.exp(log_mart))
    return np.array(uinnsms), eta_stars
