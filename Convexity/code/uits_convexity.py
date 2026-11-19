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
import numpy as np

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
# Betting fractions that do not depend on the null (convexifying: lambda = c / eta)
# ========================================================================================
def kelly_fraction(x_past, eta_ref, c_max=0.9):
    """Predictable Kelly fraction at a single reference null:
    argmax_{c in [0, c_max]} sum_i log(1 - c + c x_i / eta_ref).  Returns 0 with no data."""
    x_past = np.asarray(x_past, float)
    if len(x_past) == 0:
        return 0.0
    z = x_past / eta_ref - 1
    score = lambda c: np.sum(z / (1 + c * z))           # derivative in c; decreasing
    if score(0.0) <= 0:
        return 0.0
    if score(c_max) >= 0:
        return c_max
    lo, hi = 0.0, c_max
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
    return (lo + hi) / 2

def taylor_fraction(mu_hat, sd_hat, eta_ref, c_max=0.9, sd_min=0.01):
    """Second-order (Taylor) approximation to the Kelly fraction at eta_ref:
    c = eta_ref (mu_hat - eta_ref) / (sd_hat^2 + (mu_hat - eta_ref)^2), clipped to [0, c_max].
    This is eta_ref times the AGRAPA bet evaluated once, at the reference null."""
    d = mu_hat - eta_ref
    return float(np.clip(eta_ref * d / (max(sd_hat, sd_min) ** 2 + d ** 2), 0.0, c_max))


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




def growth_at(x_past, eta_ref, c_max=0.9):
    """Mean and SD of the per-draw log factors log(1 - c + c x_i / eta_ref) at the hindsight Kelly fraction c:
    an (optimistic) estimate of the stratum's Kelly growth rate kappa_k(eta_ref)."""
    x_past = np.asarray(x_past, float)
    if len(x_past) < 2:
        return 0.0, np.inf
    c = kelly_fraction(x_past, eta_ref, c_max)
    f = np.log(1 - c + c * x_past / eta_ref)
    return float(f.mean()), float(f.std(ddof=1))


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
# Minimising a convex, separable UI-TS objective over C for any K: water-filling
# ========================================================================================
def _pad(xs, fill=0.0):
    n = max(1, max(len(x) for x in xs)); K = len(xs)
    X = np.full((K, n), fill); mask = np.zeros((K, n), bool)
    for k, x in enumerate(xs):
        X[k, :len(x)] = x; mask[k, :len(x)] = True
    return X, mask

class FractionObjective:
    """L_k(eta) = sum_i log(1 - c_ki + c_ki x_ki / eta) for given null-free fractions c_ki (S1, S2).
    value, deriv, deriv2 take a K-vector eta and return K-vectors."""
    def __init__(self, xs, cs):
        self.X, self.mask = _pad(xs); self.C, _ = _pad(cs)
        self.C = np.where(self.mask, self.C, 0.0)
    def value(self, eta):
        e = np.asarray(eta, float)[:, None]
        return np.log(1 - self.C + self.C * self.X / e).sum(1)
    def deriv(self, eta):
        e = np.asarray(eta, float)[:, None]
        num = self.C * self.X / e ** 2
        return -(num / (1 - self.C + self.C * self.X / e)).sum(1)
    def deriv2(self, eta):
        e = np.asarray(eta, float)[:, None]
        D = 1 - self.C + self.C * self.X / e; A = self.C * self.X / e ** 2
        return (2 * self.C * self.X / (e ** 3 * D) - A ** 2 / D ** 2).sum(1)
    def d12(self, eta):
        e = np.asarray(eta, float)[:, None]
        D = 1 - self.C + self.C * self.X / e; A = self.C * self.X / e ** 2; r = A / D
        return -r.sum(1), (2 * r / e - r ** 2).sum(1)

class MixtureObjective:
    """S3: L_k(eta) = log int prod_i (1 - c + c x_ki / eta) dc (uniform prior on c), by Gauss-Legendre quadrature
    with m nodes (exact when m >= (n_k + 1) / 2; by default m is chosen so)."""
    def __init__(self, xs, m=None):
        self.X, self.mask = _pad(xs)
        m = m or max(8, int(np.ceil((self.X.shape[1] + 1) / 2)))
        z, wq = np.polynomial.legendre.leggauss(m); self.cs, self.lw = (z + 1) / 2, np.log(wq / 2)
    def _terms(self, eta):
        e = np.asarray(eta, float)[:, None, None]
        cc = self.cs[None, None, :]; xx = self.X[:, :, None]; mm = self.mask[:, :, None]
        a = np.where(mm, np.log(1 - cc + cc * xx / e), 0.0).sum(1) + self.lw[None, :]   # (K, m)
        return a, cc, xx, mm, e
    def value(self, eta):
        a = self._terms(eta)[0]; mx = a.max(1)
        return mx + np.log(np.exp(a - mx[:, None]).sum(1))
    def _slopes(self, eta):
        a, cc, xx, mm, e = self._terms(eta)
        post = np.exp(a - a.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        D = 1 - cc + cc * xx / e; A = cc * xx / e ** 2
        d1 = np.where(mm, -A / D, 0.0).sum(1)                                        # (K, m)
        d2 = np.where(mm, 2 * cc * xx / (e ** 3 * D) - A ** 2 / D ** 2, 0.0).sum(1)
        return post, d1, d2
    def deriv(self, eta):
        post, d1, _ = self._slopes(eta)
        return (post * d1).sum(1)
    def deriv2(self, eta):
        return self.d12(eta)[1]
    def d12(self, eta):
        post, d1, d2 = self._slopes(eta)
        m1 = (post * d1).sum(1)
        return m1, (post * (d2 + d1 ** 2)).sum(1) - m1 ** 2


def waterfill_min(obj, w, eta_0, eps=1e-6, newton_iters=40):
    """Minimise sum_k L_k(eta_k) subject to w . eta = eta_0, eps <= eta_k <= 1, for convex nonincreasing L_k.

    Water-filling: for a level nu <= 0, every stratum takes the eta_k(nu) at which its slope L_k'(eta_k) equals
    nu w_k (clipped to [eps, 1]).  Because L_k' is nondecreasing (convexity), eta_k(nu) is unique and nondecreasing
    in nu, so the total w . eta(nu) is monotone and the level with w . eta(nu) = eta_0 is a one-dimensional root.
    These are exactly the KKT conditions, which are sufficient for a global minimum because the problem is convex.
    Inner roots: safeguarded Newton on (-L_k')^(-1/2), all strata at once.  Outer root: Brent's method on log(-nu).
    Each slope evaluation costs O(total number of draws) (times the number of quadrature nodes for S3).
    Strata with no information (L_k' = 0) take up whatever null mass is left over.  Returns (minimum, minimiser)."""
    from scipy.optimize import brentq
    w = np.asarray(w, float); K = len(w)
    lo_b, hi_b = np.full(K, eps), np.ones(K)
    d_lo, d_hi = obj.deriv(lo_b), obj.deriv(hi_b)
    flat = np.abs(d_lo) < 1e-12
    if flat.all():
        eta = np.full(K, eta_0 / w.sum()); return float(obj.value(eta).sum()), eta
    def eta_of(nu):
        # Solve L_k'(eta) = nu w_k stratum by stratum.  Near 0 the slopes behave like -1/eta^2, so Newton is run on
        # phi(eta) = (-L_k'(eta))^(-1/2), which is increasing and close to linear (one step if L_k' = -A/eta^2).
        target = nu * w
        at_lo, at_hi = (d_lo >= target) | flat, d_hi <= target          # roots clipped to a bound: nothing to solve
        free = ~(at_lo | at_hi)
        phi_star = 1 / np.sqrt(np.maximum(-target, 1e-300))
        a, b = lo_b.copy(), hi_b.copy(); x = (a + b) / 2
        for _ in range(newton_iters if free.any() else 0):
            d1, d2 = obj.d12(x); d1 = np.minimum(d1, -1e-300)
            g = np.where(free, 1 / np.sqrt(-d1) - phi_star, 0.0)
            a = np.where(g < 0, x, a); b = np.where(g > 0, x, b)
            free = free & (np.abs(g) > 1e-12 * phi_star) & (b - a > 1e-13)     # converged strata stay put
            if not free.any():
                break
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                dg = 0.5 * d2 / (-d1) ** 1.5
                xn = x - g / dg
            ok = np.isfinite(xn) & (xn >= a) & (xn <= b)
            x = np.where(free, np.where(ok, xn, (a + b) / 2), x)
        x = np.where(at_lo, lo_b, np.where(at_hi, hi_b, x))
        return np.where(flat, eps, x)
    f = lambda s_: float(w @ eta_of(-np.exp(s_))) - eta_0        # nonincreasing in s
    s_min = -40.0
    if f(s_min) <= 0:                                             # informative strata can all sit at eta_k = 1
        nu = -np.exp(s_min)
    else:
        s_max = float(np.log(-np.min(d_lo / w)))
        nu = -np.exp(brentq(f, s_min, s_max, xtol=1e-12, rtol=4 * np.finfo(float).eps, maxiter=200))
    eta = eta_of(nu)
    short = eta_0 - w @ eta
    if short > 1e-12 and flat.any():                              # uninformative strata absorb the remaining null mass
        eta[flat] = np.minimum(1.0, eps + short / w[flat].sum())
    return float(obj.value(eta).sum()), eta
