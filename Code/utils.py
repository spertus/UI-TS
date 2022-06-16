import numpy as np
import scipy as sp
import math
from scipy.stats import bernoulli, multinomial
from scipy.stats.mstats import gmean


def sprt_mart(x : np.array, N : int, mu : float=1/2, eta: float=1-np.finfo(float).eps, \
              u: float=1, random_order = True):
    '''
    Finds the SPRT supermartingale sequence to test the hypothesis that the population
    mean is less than or equal to mu against the alternative that it is eta,
    for a population of size N of values in the interval [0, u].

    Generalizes Wald's SPRT for the Bernoulli to sampling without replacement and to bounded
    values rather than binary values.

    If N is finite, assumes the sample is drawn without replacement
    If N is infinite, assumes the sample is with replacement

    Data are assumed to be in random order. If not, the calculation for sampling without replacement is incorrect.

    Parameters:
    -----------
    x : binary list, one element per draw. A list element is 1 if the
        the corresponding trial was a success
    N : int
        population size for sampling without replacement, or np.infinity for
        sampling with replacement
    theta : float in (0,u)
        hypothesized population mean
    eta : float in (0,u)
        alternative hypothesized population mean
    random_order : Boolean
        if the data are in random order, setting this to True can improve the power.
        If the data are not in random order, set to False

    Returns
    -------
    terms : np.array
        sequence of terms that would be a supermartingale under the null
    '''
    if any((xx < 0 or xx > u) for xx in x):
        raise ValueError(f'Data out of range [0,{u}]')
    if np.isfinite(N):
        if not random_order:
            raise ValueError("data must be in random order for samples without replacement")
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        m = (N*mu-S)/(N-j+1)                   # mean of population after (j-1)st draw, if null is true
    else:
        m = mu
    with np.errstate(divide='ignore',invalid='ignore'):
        terms = np.cumprod((x*eta/m + (u-x)*(u-eta)/(u-m))/u) # generalization of Bernoulli SPRT
    terms[m<0] = np.inf                        # the null is surely false
    return terms

def shrink_trunc(x: np.array, N: int, mu: float=1/2, nu: float=1-np.finfo(float).eps, u: float=1, \
                     c: float=1/2, d: float=100, f: float=0, minsd: float=10**-6) -> np.array:
        '''
        apply the shrinkage and truncation estimator to an array

        sample mean is shrunk towards nu, with relative weight d compared to a single observation,
        then that combination is shrunk towards u, with relative weight f/(stdev(x)).

        The result is truncated above at u-u*eps and below at mu_j+e_j(c,j)

        The standard deviation is calculated using Welford's method.


        S_1 = 0
        S_j = \sum_{i=1}^{j-1} x_i, j > 1
        m_j = (N*mu-S_j)/(N-j+1) if np.isfinite(N) else mu
        e_j = c/sqrt(d+j-1)
        sd_1 = sd_2 = 1
        sd_j = sqrt[(\sum_{i=1}^{j-1} (x_i-S_j/(j-1))^2)/(j-2)] \wedge minsd, j>2
        eta_j =  ( [(d*nu + S_j)/(d+j-1) + f*u/sd_j]/(1+f/sd_j) \vee (m_j+e_j) ) \wedge u*(1-eps)

        Parameters
        ----------
        x : np.array
            input data
        mu : float in (0, 1)
            hypothesized population mean
        eta : float in (t, 1)
            initial alternative hypothethesized value for the population mean
        c : positive float
            scale factor for allowing the estimated mean to approach t from above
        d : positive float
            relative weight of nu compared to an observation, in updating the alternative for each term
        f : positive float
            relative weight of the upper bound u (normalized by the sample standard deviation)
        minsd : positive float
            lower threshold for the standard deviation of the sample, to avoid divide-by-zero errors and
            to limit the weight of u
        '''
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        m = (N*mu-S)/(N-j+1) if np.isfinite(N) else mu   # mean of population after (j-1)st draw, if null is true
        mj = [x[0]]                            # Welford's algorithm for running mean and running SD
        sdj = [0]
        for i, xj in enumerate(x[1:]):
            mj.append(mj[-1]+(xj-mj[-1])/(i+1))
            sdj.append(sdj[-1]+(xj-mj[-2])*(xj-mj[-1]))
        sdj = np.sqrt(sdj/j)
        sdj = np.insert(np.maximum(sdj,minsd),0,1)[0:-1] # threshold the sd, set first sd to 1
        weighted = ((d*nu+S)/(d+j-1) + f*u/sdj)/(1+f/sdj)
        return np.minimum(u*(1-np.finfo(float).eps), np.maximum(weighted,m+c/np.sqrt(d+j-1)))

def alpha_mart(x: np.array, N: int, mu: float=1/2, eta: float=1-np.finfo(float).eps, f: float=0, u: float=1, \
               estim: callable=shrink_trunc) -> np.array :
    '''
    Finds the ALPHA martingale for the hypothesis that the population
    mean is less than or equal to t using a martingale method,
    for a population of size N, based on a series of draws x.

    The draws must be in random order, or the sequence is not a martingale under the null

    If N is finite, assumes the sample is drawn without replacement
    If N is infinite, assumes the sample is with replacement

    Parameters
    ----------
    x : list corresponding to the data
    N : int
        population size for sampling without replacement, or np.infinity for sampling with replacement
    mu : float in (0,1)
        hypothesized fraction of ones in the population
    eta : float in (t,1)
        alternative hypothesized population mean
    estim : callable
        estim(x, N, mu, eta, u) -> np.array of length len(x), the sequence of values of eta_j for ALPHA

    Returns
    -------
    terms : array
        sequence of terms that would be a nonnegative supermartingale under the null
    '''
    S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
    j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
    m = (N*mu-S)/(N-j+1) if np.isfinite(N) else mu   # mean of population after (j-1)st draw, if null is true
    etaj = estim(x=x, N=N, mu=mu, nu=eta, f=f,u=u)
    with np.errstate(divide='ignore',invalid='ignore'):
        terms = np.cumprod((x*etaj/m + (u-x)*(u-etaj)/(u-m))/u)
    terms[m<0] = np.inf
    terms[m>u] = 0
    return terms, m

def stratum_selector(marts : list, mu : list, u : np.array, rule : callable, seed=None) -> np.array:
    '''
    select the order of strata from which the samples will be drawn to construct the test SM

    Parameters
    ----------
    marts: list of np.arrays
        each array is the test supermartingale for one stratum

    mu: list of np.arrays
        each array is the running null mean for a stratum

    u: np.array
        the known maximum values within each strata

    rule: callable
        maps three K-vectors (where K is the number of strata) to a value in {0, \ldots, K-1}, the stratum
        from which the next term will be included in the product SM.
        The rule should stop sampling from a stratum when that stratum is exhausted.
        The first K-vector is the current value of each stratumwise SM
        The second K-vector is the number of samples drawn from each stratum so far
        The third is the number of elements in each stratum, or np.inf for sampling with replacement

    Returns
    -------
    strata : np.array
        the series of strata from which the samples are included
    T : np.array
        the resulting product test SM
    '''
    strata = np.array([])
    T = np.array([1])
    running_T = np.ones(len(marts))  # current value of each stratumwise SM
    running_n = np.zeros(len(marts)) # current index of each stratumwise SM
    running_mu = np.asarray([item[0] for item in mu]) #current value of the conditional null mean
    ns = np.zeros(len(marts))        # assumes the martingales exhaust the strata, for testing
    for i in range(len(marts)):
        ns[i] = len(marts[i])
    t = 0
    while np.any(running_n < ns-1):
        t += 1
        next_s = rule(running_T, running_n, running_mu, u, ns)
        running_n[next_s] += 1
        running_T[next_s] = marts[next_s][int(running_n[next_s])]
        running_mu[next_s] = mu[next_s][int(running_n[next_s])]
        if np.isposinf(running_T[next_s]):
            T = np.append(T, np.ones(int(sum(ns) - sum(running_n))) * np.inf) #pad out with infinities
            strata = np.append(strata, np.ones(int(sum(ns) - sum(running_n))) * np.inf) #stratum = inf if no sample is drawn
            break
        elif np.all((running_mu >= u) | (running_n == ns-1)):
            T = np.append(T, np.ones(int(sum(ns) - sum(running_n))) * T[-1]) #pad out with last value of martingale and stop counting, that null is ture
            strata = np.append(strata, np.ones(int(sum(ns) - sum(running_n))) * np.inf)
            break
        elif np.any(running_mu <= 0):
            T = np.append(T, np.ones(int(sum(ns) - sum(running_n))) * np.inf) #pad out with infinities; that null is certainly false
            strata = np.append(strata, np.ones(int(sum(ns) - sum(running_n))) * np.inf)
            break
        else:
            T = np.append(T, np.prod(running_T))
            strata = np.append(strata, next_s)
    return strata, T


def multinomial_selector(running_T : np.array, running_n : np.array, running_mu : np.array, u : np.array, ns : np.array, prng : np.random.RandomState=None) -> int:
    '''
    find the next stratum from which to take a term for the product supermartingale test

    Parameters
    ----------
    running_t : np.array
        the current value of each stratumwise SM
    running_n : np.array
        the number of samples drawn from each stratum so far
    running_mu: np.array
        the current value of mu in each stratum
    u: np.array
        the known upper bound in each stratum
    ns : np.array
        the total number of items in each stratum, or np.inf for sampling with replacement
    prng : np.Random.RandomState
        a PRNG (or seed, or none)
    '''
    available = (running_n < ns-1) & (running_mu < u) # strata that aren't exhausted and where null isn't deterministically true
    if np.sum(available) == 0:
        raise ValueError(f'all strata are exhausted: {running_n=} {ns=}')
    geomean = gmean(running_T[available])
    if any(np.isposinf(running_T) & available):
        ratios = np.where(np.isposinf(running_T), 1, 0)
    else:
        ratios = running_T/geomean
    ratios = np.where(available, ratios, 0)
    probs = ratios/sum(ratios)
    return np.random.choice(len(ratios), p = probs)
    #return multinomial.rvs(1, ratios/np.sum(ratios), random_state=prng)


def get_global_pvalue(strata: list, u: np.array, v: np.array, rule: callable):
    '''
    returns a P-value (maximized over nuisance parameter) for the global null hypothesis that the mean of a population with 2 strata is equal to 1/2

    Parameters
    ----------
    strata: list of 2 np.arrays
        each np.array contains the values of a population within a stratum, to be sampled by SRSing
    u: np.array of length 2
        each value is the upper bound in the corresponding stratum in strata (e.g. u[0] is the known upper bound of strata[0])
    v: np.array of length 2
        the (reported) diluted margin in each stratum, used to set the tuning parameter eta_0 in ALPHA martingale
    rule: callable
        the stratum selection rule to be used, e.g., multinomial_selector

    Returns
    -------
    p_values: np.array of length N_1 + N_2
        the P-values for the entire sequence of samples comprised of the strata
    stratum_selections: np.array of length N_1 + N_2
        the stratum selected at each sample in the P-value-maximizing martingale (a different null corresponds to each index)
    null_selections: np.array
        the P-value-maximizing null in stratum 1 at each sample size
    '''
    assert len(strata) == 2, "Only works for 2 strata, input as list of 2 np.arrays." #only works for 2 strata, not clear how to scale efficiently yet

    shuffled_1 = np.random.permutation(strata[0])
    shuffled_2 = np.random.permutation(strata[1])
    N = np.concatenate((np.array([len(shuffled_1)]), np.array([len(shuffled_2)])))
    w = N/sum(N)
    epsilon = 1 / (2*np.max(N))
    theta_1_grid = np.arange(epsilon, u[0] - epsilon, epsilon) #sequence from epsilon to u[0] - epsilon
    theta_2_grid = (1/2 - w[0] * theta_1_grid) / w[1]
    strata_matrix = np.zeros((len(shuffled_1) + len(shuffled_2) - 1, len(theta_1_grid)))
    intersection_marts = np.zeros((len(shuffled_1) + len(shuffled_2), len(theta_1_grid)))
    for i in range(len(theta_1_grid)):
        mart_1, mu_1 = alpha_mart(x = shuffled_1, N = N[0], mu = theta_1_grid[i], eta = 1/(2-v[0]), f = .01, u = u[0])
        mart_2, mu_2 = alpha_mart(x = shuffled_2, N = N[1], mu = theta_2_grid[i], eta = 1/(2-v[1]), f = .01, u = u[1])
        strata_matrix[:,i], intersection_marts[:,i] = stratum_selector(
            marts = [mart_1, mart_2],
            mu = [mu_1, mu_2],
            u = u,
            rule = rule)
    null_index = np.argmin(intersection_marts, axis = 1)
    #stratum_selections = strata_matrix[1:sum(N), null_index]
    #minimized_martingale = intersection_marts[1:sum(N), null_index]
    minimized_martingale = np.ones(sum(N))
    stratum_selections = np.ones(sum(N) - 1) * np.inf
    for i in np.arange(sum(N) - 1):
        minimized_martingale[i] = intersection_marts[i,null_index[i]]
        stratum_selections[i] = strata_matrix[i,null_index[i]]
    p_values = 1 / np.maximum(1, minimized_martingale)
    null_selections = theta_1_grid[null_index]
    return p_values, stratum_selections, null_selections

def simulate_audits(strata: list, u: np.array, v: np.array, rule: callable, n_sims: int, alpha: float = 0.05):
    '''
    simulates n_sims audits by wrapping get_global_pvalue and returns stopping times at level alpha

    Parameters
    ----------
    strata: list of 2 np.arrays
        each np.array contains the values of a population within a stratum, to be sampled by SRSing
    u: np.array of length 2
        each value is the upper bound in the corresponding stratum in strata (e.g. u[0] is the known upper bound of strata[0])
    v: np.array of length 2
        the (reported) diluted margin in each stratum, used to set the tuning parameter eta_0 in ALPHA martingale
    rule: callable
        the stratum selection rule to be used, e.g., multinomial_selector
    n_sims: positive integer
        the number of simulations to run
    alpha: float in (0,1)
        the risk limit for each simulated audit to stop

    Returns
    -------
    stopping_times: np.array of length n_sims
        the stopping time for each simulated audit
    '''
    stopping_times = np.zeros(n_sims)
    for i in np.arange(n_sims):
        p_values, stratum_selections, null_selections = get_global_pvalue(strata = strata, u = u, v = v, rule = rule)
        if any(p_values < alpha):
            stopping_times[i] = np.min(np.where(p_values < alpha))
        else:
            stopping_times[i] = np.inf
    return stopping_times

##############################################################################

def test_shrink_trunc():
    epsj = lambda c, d, j: c/math.sqrt(d+j-1)
    Sj = lambda x, j: 0 if j==1 else np.sum(x[0:j-1])
    muj = lambda N, mu, x, j: (N*mu - Sj(x, j))/(N-j+1) if np.isfinite(N) else mu
    nus = [.51, .55, .6]
    mu = 1/2
    u = 1
    d = 10
    vrand =  sp.stats.bernoulli.rvs(1/2, size=20)
    v = [
        np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        vrand
    ]
    for nu in nus:
        c = (nu-mu)/2
        for x in v:
            N = len(x)
            xinf = shrink_trunc(x, np.inf, mu, nu, c=c, d=d)
            xfin = shrink_trunc(x, len(x), mu, nu, c=c, d=d)
            yinf = np.zeros(len(x))
            yfin = np.zeros(len(x))
            for j in range(1,len(x)+1):
                est = (d*nu + Sj(x,j))/(d+j-1)
                most = u*(1-np.finfo(float).eps)
                yinf[j-1] = np.minimum(np.maximum(mu+epsj(c,d,j), est), most)
                yfin[j-1] = np.minimum(np.maximum(muj(N,mu,x,j)+epsj(c,d,j), est), most)
            np.testing.assert_allclose(xinf, yinf)
            np.testing.assert_allclose(xfin, yfin)

def test_multinomial_selector():
    running_T = np.ones(3)
    running_n = np.zeros(3)
    pass  # fix me!

if __name__ == "__main__":
    test_shrink_trunc()
