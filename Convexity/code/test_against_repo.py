"""Cross-check the vectorised betting rules against utils.Bets in the UI-TS repository.
Usage: python test_against_repo.py /path/to/UI-TS/Code"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "../../UI-TS/Code")
import utils
from utils import Bets
import uits_convexity as uc

rng = np.random.default_rng(0)
x = rng.beta(2.6, 1.4, 40)
etas = np.array([0.2, 0.45, 0.6, 0.8])
checks = {
    "fixed":        (lambda e: Bets.fixed(x, e),                lambda: uc.bets_fixed(x, etas)),
    "agrapa":       (lambda e: Bets.agrapa(x, e),               lambda: uc.bets_agrapa(x, etas)),
    "alpha":        (lambda e: Bets.predictable_bernoulli(x, e), lambda: uc.bets_alpha(x, etas)),
    "negexp":       (lambda e: Bets.negative_exponential(x, e), lambda: uc.bets_negexp(x, etas)),
}
for name, (repo, mine) in checks.items():
    R = np.vstack([repo(e) for e in etas]); M = mine()
    print(f"{name:10s} max |repo - vectorised| = {np.max(np.abs(R - M)):.2e}")

# GRAPA: repo has no cap below 1/eta; compare where the repo bet is below 0.99/eta
R = np.vstack([Bets.grapa(x, e) for e in etas]); M = uc.bets_grapa(x, etas, c=1.0)
print(f"{'grapa':10s} max |repo - vectorised| = {np.max(np.abs(R - M)):.2e}")

# universal portfolio: product of the repo's predictable UP bets equals the mixture wealth
xs = x[:25]
for e in [0.3, 0.5]:
    lam = Bets.universal_portfolio(xs, e)
    repo_logW = np.sum(np.log1p(lam * (xs - e)))
    mine = uc.log_wealth_up(xs, np.array([e]))[0]
    print(f"UP eta={e}: repo predictable-bet log-wealth {repo_logW:.6f} | mixture {mine:.6f}")

# water-filling against the repository's style of solve (cvxopt interior point), on a small S1/S2-type problem
try:
    import simulations as sm
    w = rng.dirichlet(np.full(8, 5.0)); xs8 = [rng.beta(2.4, 1.6, 30) for _ in range(8)]
    cs8 = [rng.uniform(0.1, 0.8, 30) for _ in range(8)]
    fo = uc.FractionObjective(xs8, cs8)
    v_wf, _ = uc.waterfill_min(fo, w, 0.5); v_cv, status = sm._cvxopt_min(fo, w, 0.5)
    print(f"water-filling {v_wf:.8f} | cvxopt {v_cv:.8f} ({status})")
except ImportError as err:
    print("skipping water-filling check:", err)
