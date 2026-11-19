"""
Shape of union-of-intersections test sequences (UI-TSs) as functions of the intersection null.

Companion code for the note "Betting rules that make the UI-TS convex" (draft).
Notation follows Spertus, Sridhar, and Stark, "Sequential stratified inference for the mean"
(arXiv:2409.06680) and the UI-TS repository (github.com/spertus/UI-TS):
    stratum k, observations X_{k1}, X_{k2}, ... in [0,1], null mean eta_k,
    betting TSM  M_{kt}(eta_k) = prod_i [1 + lambda_{ki}(eta_k) (X_{ki} - eta_k)],
    I-TSM        M_t(eta) = prod_k M_{k T_k(t)}(eta_k),
    UI-TS        min over eta in C = {eta in [0,1]^K : w . eta = eta_0}.

Everything here is for sampling WITH replacement and eta-oblivious selections, so that
log M_t(eta) = sum_k L_k(eta_k) is separable.  Each betting rule is vectorised over a grid
of eta values: it takes the data x (length n) and a grid eta (length G) and returns a
G-by-n array of bets lambda_{ki}(eta).  Conventions (lagged Welford estimates, default
tuning constants) match utils.Bets in the UI-TS repository; see test_against_repo.py.
"""
import os
import sys
import numpy as np

# The null-free fractions, the separable objectives and the water-filling solver live in the repository's
# Code/utils.py (tested in Code/tests.py). Import them from there, so that there is a single copy. The path is inserted
# after this script's own directory, so that modules here are not shadowed by modules in Code/.
_CODE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Code"))
if _CODE not in sys.path:
    sys.path.insert(1, _CODE)
from utils import kelly_fraction, taylor_fraction, growth_at, FractionObjective, MixtureObjective, waterfill_min

# ----------------------------------------------------------------------------------------
# lagged estimates (same convention as utils.Bets.lag_welford: mu_0 = 1/2, sd_0 = 1/4,
# sample SD with ddof = 1, and the SD after one observation replaced by sd_0)
# ----------------------------------------------------------------------------------------
def lag_mean_sd(x, mu_0=0.5, sd_0=0.25):
    x = np.asarray(x, float); n = len(x)
    t = np.arange(1, n + 1)
    mu = np.cumsum(x) / t
    ss = np.cumsum(x**2) - t * mu**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.maximum(ss, 0) / (t - 1))
    if n > 0:
        sd[0] = sd_0
    lag_mu = np.r_[mu_0, mu[:-1]]
    lag_sd = np.r_[sd_0, sd[:-1]]
    return lag_mu, lag_sd


# ----------------------------------------------------------------------------------------
# betting rules: bets(x, eta_grid, **kw) -> (G, n) array of lambda_{ki}(eta)
# ----------------------------------------------------------------------------------------
def bets_fixed(x, eta, c=0.75, **kw):
    """eta-oblivious fixed bet lambda = c (utils.Bets.fixed)."""
    return np.full((len(eta), len(x)), c)

def bets_agrapa(x, eta, c=0.75, sd_min=0.01, eps=1e-5, **kw):
    """aGRAPA: 0 v (mu_hat - eta)/(sd_hat^2 + (mu_hat - eta)^2) ^ c/eta (utils.Bets.agrapa)."""
    m, s = lag_mean_sd(x); s = np.maximum(s, sd_min)
    d = m[None, :] - eta[:, None]
    lam = d / (s[None, :]**2 + d**2)
    return np.maximum(0, np.minimum(lam, c / (eta[:, None] + eps)))

def bets_binary_kelly(x, eta, c=0.99, mu_0=0.5, **kw):
    """Kelly bet for a Bernoulli alternative with the lagged mean plugged in:
    lambda = 0 v (mu_hat - eta)/(eta (1 - eta)) ^ c/eta  (cf. utils.Bets.apriori_bernoulli)."""
    m, _ = lag_mean_sd(x, mu_0=mu_0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (m[None, :] - eta[:, None]) / (eta[:, None] * (1 - eta[:, None]))
    return np.clip(lam, 0, c / eta[:, None])

def bets_alpha(x, eta, eps_0=0.5, d=20, c=0.95, tol=1e-5, **kw):
    """ALPHA shrink-truncate bet (utils.Bets.predictable_bernoulli), mu_0 = (eta + 1)/2."""
    n = len(x); S = np.r_[0, np.cumsum(x)[:-1]]; j = np.arange(1, n + 1)
    mu_0 = (eta[:, None] + 1) / 2
    est = np.maximum((d * mu_0 + S[None, :]) / (d + j[None, :] - 1),
                     eta[:, None] + eps_0 / np.sqrt(d + j[None, :] - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.minimum((est / eta[:, None] - 1) / (1 - eta[:, None]), c / (eta[:, None] + tol))

def bets_negexp(x, eta, c=0.75, **kw):
    """Negative-exponential bet lambda = exp(1 - b eta), b = (1 - log(c - sd_hat))/mu_hat
    (utils.Bets.negative_exponential with its default branch)."""
    m, s = lag_mean_sd(x)
    with np.errstate(divide="ignore"):
        b = (1 - np.log(c - s)) / m                 # m = 0 gives b = inf and a zero bet, as in the repo
    with np.errstate(invalid="ignore"):
        lam = np.exp(1 - b[None, :] * eta[:, None])
    return np.nan_to_num(lam, nan=0.0)

def _kelly_root(D, hi):
    """argmax_{lam in [0, hi]} sum_i log(1 + lam D_i), vectorised over rows of D (G, m)."""
    score = lambda lam: (D / (1 + lam[:, None] * D)).sum(1)
    lo = np.zeros(D.shape[0]); up = hi.copy()
    pos0 = score(lo) > 0; posmax = score(up) > 0
    for _ in range(60):
        mid = (lo + up) / 2; s = score(mid) > 0
        lo = np.where(s, mid, lo); up = np.where(s, up, mid)
    lam = (lo + up) / 2
    lam = np.where(posmax, hi, lam)
    return np.where(pos0, lam, 0.0)

def bets_grapa(x, eta, c=0.99, **kw):
    """GRAPA: Kelly-optimal bet for the lagged empirical distribution, first bet 0
    (utils.Bets.grapa), capped at c/eta (utils.Bets.kelly_optimal caps at 1/eta)."""
    x = np.asarray(x, float); n = len(x)
    lam = np.zeros((len(eta), n)); hi = c / eta
    for t in range(1, n):
        lam[:, t] = _kelly_root(x[None, :t] - eta[:, None], hi)
    return lam

BETS = {"fixed": bets_fixed, "agrapa": bets_agrapa,
        "grapa": bets_grapa, "binary_kelly": bets_binary_kelly, "alpha": bets_alpha,
        "negexp": bets_negexp}


def log_wealth(x, eta, rule, **kw):
    """L(eta) = sum_i log(1 + lambda_i(eta)(x_i - eta)) on the grid eta, for a rule in BETS,
    or for the universal-portfolio mixture (rule='up') or hindsight Kelly (rule='hindsight')."""
    x = np.asarray(x, float)
    if rule == "up":
        return log_wealth_up(x, eta, **kw)
    if rule == "hindsight":
        lam = _kelly_root(x[None, :] - eta[:, None], kw.get("c", 0.99) / eta)
        return np.log1p(lam[:, None] * (x[None, :] - eta[:, None])).sum(1)
    lam = BETS[rule](x, eta, **kw)
    with np.errstate(divide="ignore"):
        return np.log1p(lam * (x[None, :] - eta[:, None])).sum(1)


# ----------------------------------------------------------------------------------------
# universal-portfolio (mixture) TSM:  W(eta) = int_0^1 prod_i (1 - c + c x_i/eta) pi(dc)
# computed exactly (it is a polynomial in c) with Gauss-Legendre / Gauss-Jacobi quadrature
# ----------------------------------------------------------------------------------------
def _quad_nodes(n, prior="uniform"):
    m = max(32, n // 2 + 2)
    if prior == "uniform":
        z, wq = np.polynomial.legendre.leggauss(m)
        return (z + 1) / 2, wq / 2
    if prior == "jeffreys":              # Beta(1/2, 1/2) prior on the fraction c
        from scipy.special import roots_jacobi
        z, wq = roots_jacobi(m, -0.5, -0.5)
        return (z + 1) / 2, wq / wq.sum()
    raise ValueError(prior)

def _up_terms(x, eta, prior="uniform"):
    cs, wq = _quad_nodes(len(x), prior)
    eta = np.atleast_1d(np.asarray(eta, float))
    # (G, m) log prod_i (1 - c + c x_i / eta)
    LL = np.log(1 - cs[None, None, :] + cs[None, None, :] * x[None, :, None] / eta[:, None, None]).sum(1)
    return cs, np.log(wq), LL

def log_wealth_up(x, eta, prior="uniform", **kw):
    if len(x) == 0:
        return np.zeros(np.size(eta))
    cs, lw, LL = _up_terms(x, eta, prior)
    a = LL + lw[None, :]; mx = a.max(1)
    return mx + np.log(np.exp(a - mx[:, None]).sum(1))

def dlog_wealth_up(x, eta, prior="uniform"):
    """d/d eta of log W(eta): posterior-weighted sum of per-observation derivatives."""
    eta = np.atleast_1d(np.asarray(eta, float))
    if len(x) == 0:
        return np.zeros_like(eta)
    cs, lw, LL = _up_terms(x, eta, prior)
    a = LL + lw[None, :]; a -= a.max(1, keepdims=True); post = np.exp(a); post /= post.sum(1, keepdims=True)
    e = eta[:, None, None]; cc = cs[None, None, :]; xx = x[None, :, None]
    dterm = (-(cc * xx / e**2) / (1 - cc + cc * xx / e)).sum(1)          # (G, m)
    return (post * dterm).sum(1)

def up_fraction(x_past, eta, prior="uniform"):
    """Predictable UP bet fraction c_t(eta) = posterior mean of c given x_past (bet = c_t/eta)."""
    if len(x_past) == 0:
        return np.full(np.size(eta), 0.5)
    cs, lw, LL = _up_terms(np.asarray(x_past, float), eta, prior)
    a = LL + lw[None, :]; a -= a.max(1, keepdims=True); post = np.exp(a); post /= post.sum(1, keepdims=True)
    return (post * cs[None, :]).sum(1)


# ----------------------------------------------------------------------------------------
# convexity diagnostics
# ----------------------------------------------------------------------------------------
def lower_envelope(x, y):
    """Greatest convex minorant of the points (x_j, y_j), evaluated on x (monotone chain)."""
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(y)
    order = np.argsort(x[ok]); xs, ys = x[ok][order], y[ok][order]; hull = []
    for i in range(len(xs)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            if (ys[i2] - ys[i1]) * (xs[i] - xs[i1]) >= (ys[i] - ys[i1]) * (xs[i2] - xs[i1]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, xs[hull], ys[hull])

def nonconvexity(x, y):
    """rho = max_j [y_j - envelope(x_j)], in the units of y (log-wealth)."""
    return float(np.nanmax(y - lower_envelope(x, y)))


# ----------------------------------------------------------------------------------------
# minimising over C
# ----------------------------------------------------------------------------------------
def segment_C(w, eta_0, G=801):
    """K = 2: C is a segment; returns eta_1 grid and eta_2 = (eta_0 - w_1 eta_1)/w_2."""
    lo = max(0.0, (eta_0 - w[1]) / w[0]); hi = min(1.0, eta_0 / w[0])
    e1 = np.linspace(lo, hi, G); return e1, (eta_0 - w[0] * e1) / w[1]

def grid_min_K3(Lfuns, grid, eta_0=0.5):
    """Exact minimum over the grid points of C for K = 3, equal weights (eta_3 = 3 eta_0 - eta_1 - eta_2).
    Lfuns: three arrays of L_k on `grid`; L_3 is linearly interpolated."""
    E1, E2 = np.meshgrid(grid, grid, indexing="ij"); E3 = 3 * eta_0 - E1 - E2
    tot = Lfuns[0][:, None] + Lfuns[1][None, :] + np.interp(E3, grid, Lfuns[2])
    tot = np.where((E3 >= grid[0]) & (E3 <= grid[-1]), tot, np.inf)
    i = np.unravel_index(np.argmin(tot), tot.shape)
    return float(tot[i]), np.array([grid[i[0]], grid[i[1]], E3[i]])

def waterfill(dL, w, eta_0, lo=1e-6, hi=1.0, tol=1e-10):
    """Minimise sum_k L_k(eta_k) over C when every L_k is convex and nonincreasing.
    dL: list of K callables, dL[k](eta) = L_k'(eta) (nondecreasing in eta, <= 0).
    KKT: eta_k = clip((L_k')^{-1}(nu w_k), lo, hi), with the multiplier nu found by bisection."""
    K = len(dL); w = np.asarray(w, float)
    def eta_of(nu):
        out = np.empty(K)
        for k in range(K):
            target = nu * w[k]
            if dL[k](lo) >= target: out[k] = lo; continue
            if dL[k](hi) <= target: out[k] = hi; continue
            a, b = lo, hi
            for _ in range(60):
                m = (a + b) / 2
                a, b = (m, b) if dL[k](m) < target else (a, m)
            out[k] = (a + b) / 2
        return out
    nu_lo = min(dL[k](lo) / w[k] for k in range(K)); nu_hi = 0.0
    for _ in range(80):
        nu = (nu_lo + nu_hi) / 2
        if w @ eta_of(nu) < eta_0: nu_lo = nu
        else: nu_hi = nu
        if nu_hi - nu_lo < tol: break
    return eta_of((nu_lo + nu_hi) / 2)

def uits_up(x_list, w, eta_0, prior="uniform"):
    """Universal-portfolio UI-TS: returns (log M_t, minimiser eta*) by water-filling."""
    dL = [lambda e, x=x: float(dlog_wealth_up(x, e, prior)[0]) for x in x_list]
    eta = waterfill(dL, w, eta_0)
    val = sum(float(log_wealth_up(x, np.array([e]), prior)[0]) for x, e in zip(x_list, eta))
    return val, eta


# ========================================================================================
# Betting fractions that do not depend on the null (convexifying: lambda = c / eta):
# kelly_fraction (S1) and taylor_fraction (S2) are imported from Code/utils.py
# ========================================================================================


# ========================================================================================
# A lattice on C (equal weights) and a sequential engine
# ========================================================================================
class Lattice:
    """Equal weights w_k = 1/K.  Coordinates on the grid g = (1, ..., m)/m, so that the
    constraint sum_k eta_k = K eta_0 keeps every coordinate on the grid (needs K eta_0 m integer).
    eta_k = 0 is excluded because convexifying log-wealth is +infinity there."""
    def __init__(self, K, eta_0=0.5, m=200):
        S = K * eta_0 * m
        assert abs(S - round(S)) < 1e-9, "K eta_0 m must be an integer"
        self.K, self.eta_0, self.m, self.S = K, eta_0, m, int(round(S))
        self.g = np.arange(1, m + 1) / m
        if K == 2:
            i = np.arange(max(1, self.S - m), min(m, self.S - 1) + 1)
            self.idx = [i - 1, self.S - i - 1]                  # zero-based indices into g
        elif K == 3:
            I, J = np.meshgrid(np.arange(1, m + 1), np.arange(1, m + 1), indexing="ij")
            L3 = self.S - I - J
            ok = (L3 >= 1) & (L3 <= m)
            self.idx = [I[ok] - 1, J[ok] - 1, L3[ok] - 1]
            self.ij = (I[ok] - 1, J[ok] - 1)
        else:
            raise NotImplementedError("K = 2 or 3")
        self.points = np.column_stack([self.g[ix] for ix in self.idx])   # (P, K) points of C

    def objective(self, Ls):
        """log M(eta) at every lattice point of C, from per-stratum log-wealth arrays on g."""
        return sum(L[ix] for L, ix in zip(Ls, self.idx))

    def argmin(self, Ls):
        obj = self.objective(Ls); p = int(np.argmin(obj))
        return float(obj[p]), self.points[p]


class StratumBettor:
    """State of one stratum's betting TSM on the grid g, for one rule.
    rule in: fixed, agrapa, grapa, binary_kelly, alpha (ST-Bern), negexp,
             kelly_ref, taylor_ref (fractions at a reference null), up (mixture)."""
    REF_RULES = ("kelly_ref", "taylor_ref")

    def __init__(self, rule, g, c_max=0.9, up_nodes=128):
        self.rule, self.g, self.c_max = rule, g, c_max
        self.x = []                                   # past observations in this stratum
        self.s1 = 0.0; self.s2 = 0.0                  # running sums for lagged mean and SD
        self.L = np.zeros(len(g))                     # log-wealth on the grid
        if rule == "up":
            z, wq = np.polynomial.legendre.leggauss(up_nodes)
            self.cs, self.lw = (z + 1) / 2, np.log(wq / 2)
            self.state = np.zeros((len(g), up_nodes))

    def _lag(self):
        """Lagged mean and SD with the conventions of utils.Bets.lag_welford."""
        n = len(self.x)
        if n == 0:
            return 0.5, 0.25
        mu = self.s1 / n
        sd = 0.25 if n == 1 else float(np.sqrt(max(self.s2 - n * mu * mu, 0.0) / (n - 1)))
        return mu, sd

    def fraction(self, eta_ref):
        """The null-free fraction c for the next draw (reference rules only)."""
        if self.rule == "kelly_ref":
            return kelly_fraction(self.x, eta_ref, self.c_max)
        mu, sd = self._lag()
        return taylor_fraction(mu, sd, eta_ref, self.c_max)

    def bets(self, eta_ref=None):
        """lambda(eta) on the grid for the next draw (not used for 'up')."""
        g = self.g
        if self.rule in self.REF_RULES:
            return np.full(len(g), self.fraction(eta_ref)) / g
        if self.rule == "grapa":
            if len(self.x) == 0:
                return np.zeros(len(g))
            return _kelly_root(np.asarray(self.x)[None, :] - g[:, None], 0.99 / g)
        # fast paths (O(G) per draw) for the rules used in long sequential runs
        if self.rule == "fixed":
            return np.full(len(g), 0.75)
        if self.rule == "agrapa":
            mu, sd = self._lag(); d = mu - g
            return np.maximum(0, np.minimum(d / (max(sd, 0.01) ** 2 + d ** 2), 0.75 / (g + 1e-5)))
        # closed-form rules: evaluate the vectorised rule on (past + dummy) and take the last bet
        xe = np.r_[self.x, 0.5]
        return BETS[self.rule](xe, g)[:, -1]

    def update(self, x, eta_ref=None):
        if self.rule == "up":
            self.state += np.log(1 - self.cs[None, :] + self.cs[None, :] * x / self.g[:, None])
            a = self.state + self.lw[None, :]; mx = a.max(1)
            self.L = mx + np.log(np.exp(a - mx[:, None]).sum(1))
        else:
            lam = self.bets(eta_ref)
            with np.errstate(divide="ignore"):
                self.L = self.L + np.log1p(lam * (x - self.g))
        self.x.append(float(x)); self.s1 += x; self.s2 += x * x




# growth_at (the hindsight Kelly growth estimate used by selection='greedy') is imported from Code/utils.py


def run_uits(data, rule, lattice, n_steps=None, alpha=0.05, c_max=0.9, stop=False, selection="round_robin",
             proportions=None, running_max="outer", ucb_mult=2.0, sd_min=0.05):
    """Sequential UI-TS over the lattice of C (K = 2 or 3), data[k] = stratum k's draws in order.

    Reference rules (S1, S2) bet the null-free fraction computed at eta_hat = argmin over C of the current I-TSM
    (the lagged estimate of the least-favourable null).  Selection rules (all eta-oblivious and predictable):
      round_robin  -- k = t mod K;
      greedy       -- round robin until every stratum has two draws, then the stratum with the largest upper
                      confidence bound mean + ucb_mult * max(sd, sd_min) / sqrt(n_k) on the per-draw log-growth
                      of the hindsight Kelly bet at eta_hat_k (the lagged minimiser);
      oracle       -- deterministic allocation tracking the given proportions (largest deficit first).
    running_max = "outer": the UI-TS is max_{j <= t} min_eta M_j(eta) (one minimisation per step; valid by Ville);
                  "inner": min_eta max_{j <= t} M_j(eta) (larger, needs the whole running-max surface).
    Returns dict: L (per-stratum log-wealth on the grid), objective, path (log UI-TS), tau (first t with
    path >= log 1/alpha), eta_hat, counts, fractions, bettors."""
    K = lattice.K
    bettors = [StratumBettor(rule, lattice.g, c_max) for _ in range(K)]
    T = n_steps if n_steps is not None else sum(len(d) for d in data)
    counts = np.zeros(K, int)
    runmax_surface = np.zeros(len(lattice.points)); runmax = 0.0
    path = np.empty(T); tau = None
    eta_hat = np.full(K, lattice.eta_0)
    thr = np.log(1 / alpha)
    p = None if proportions is None else np.asarray(proportions, float)
    obj = np.zeros(len(lattice.points)); t_done = 0
    for t in range(T):
        avail = counts < np.array([len(d) for d in data])
        if not avail.any():
            break
        if selection == "round_robin":
            k = t % K
            if not avail[k]:
                k = int(np.flatnonzero(avail)[0])
        elif selection == "greedy":
            if counts.min() < 2:
                k = int(np.argmin(np.where(avail, counts, np.iinfo(int).max)))
            else:
                ucb = np.full(K, -np.inf)
                for j in np.flatnonzero(avail):
                    m_, s_ = growth_at(data[j][:counts[j]], eta_hat[j], c_max)
                    ucb[j] = m_ + ucb_mult * max(s_, sd_min) / np.sqrt(counts[j])
                k = int(np.argmax(ucb))
        elif selection == "oracle":
            deficit = np.where(avail, p * (t + 1) - counts, -np.inf)
            k = int(np.argmax(deficit))
        else:
            raise ValueError(selection)
        x = data[k][counts[k]]; counts[k] += 1
        bettors[k].update(x, eta_ref=eta_hat[k])
        obj = lattice.objective([b.L for b in bettors])
        if running_max == "inner":
            runmax_surface = np.maximum(runmax_surface, obj); path[t] = runmax_surface.min()
        else:
            runmax = max(runmax, float(obj.min())); path[t] = runmax
        eta_hat = lattice.points[int(np.argmin(obj))]
        t_done = t + 1
        if tau is None and path[t] >= thr:
            tau = t + 1
            if stop:
                break
    path = path[:t_done]
    return dict(L=[b.L for b in bettors], objective=obj, path=path, tau=tau, eta_hat=eta_hat, counts=counts,
                fractions=[b.fraction(eta_hat[k]) if rule in StratumBettor.REF_RULES else None
                           for k, b in enumerate(bettors)], bettors=bettors)


# ========================================================================================
# Minimising a convex, separable UI-TS objective over C for any K: water-filling.
# FractionObjective (S1/S2), MixtureObjective (S3) and waterfill_min are imported from Code/utils.py
# ========================================================================================
