"""
Unit tests for raschpy._esf.cond_loglik (conditional log-likelihood, gradient and Hessian).

Checked against brute-force enumeration of every response pattern with the person's total (log-likelihood),
central finite differences of that brute-force log-likelihood (gradient and Hessian), and internal
consistencies: gradient with and without the Hessian, batching (a tiny memory budget forces many batches),
list vs NaN-padded threshold input, and persons who contribute nothing.
"""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from raschpy import _esf


def _design(Ks, N, miss, seed):
    rng = np.random.default_rng(seed)
    b = [np.sort(rng.normal(0, 1, K)) for K in Ks]
    X = np.array([[rng.integers(0, K + 1) for K in Ks] for _ in range(N)], float)
    X[rng.random(X.shape) < miss] = np.nan
    return b, X


def _brute_ll(X, b):
    le = [np.concatenate([[0.0], -np.cumsum(x)]) for x in b]
    Ks = [len(x) for x in b]
    total = 0.0
    for x in X:
        o = np.nonzero(~np.isnan(x))[0]
        if len(o) < 2:
            continue
        r = int(np.nansum(x))
        if r == 0 or r == sum(Ks[i] for i in o):
            continue
        den = [sum(le[i][c] for i, c in zip(o, cs))
               for cs in itertools.product(*[range(Ks[i] + 1) for i in o]) if sum(cs) == r]
        total += sum(le[i][int(x[i])] for i in o) - logsumexp(den)
    return total


def _split(v, Ks):
    off = np.concatenate([[0], np.cumsum(Ks)])
    return [v[off[i]:off[i + 1]] for i in range(len(Ks))]


DESIGNS = [
    ([1, 1, 1, 1, 1, 1], 80, 0.0, 1),     # dichotomous, complete
    ([1, 1, 1, 1, 1, 1], 80, 0.4, 2),     # dichotomous, missing
    ([2, 2, 2, 2, 2], 60, 0.0, 3),        # equal K, complete
    ([1, 3, 2, 4, 1], 60, 0.3, 4),        # unequal K, missing
    ([4, 4, 4, 4], 50, 0.25, 5),          # K = 4, missing
]


@pytest.mark.parametrize("Ks,N,miss,seed", DESIGNS)
def test_loglik_matches_brute_force(Ks, N, miss, seed):
    b, X = _design(Ks, N, miss, seed)
    ll, _, _ = _esf.cond_loglik(X, b)
    assert ll == pytest.approx(_brute_ll(X, b), abs=1e-10)


@pytest.mark.parametrize("Ks,N,miss,seed", DESIGNS)
def test_gradient_and_hessian_match_finite_differences(Ks, N, miss, seed):
    b, X = _design(Ks, N, miss, seed)
    v = np.concatenate(b)
    f = lambda u: _brute_ll(X, _split(u, Ks))
    _, g, H = _esf.cond_loglik(X, b, hessian=True)
    E = np.eye(len(v))
    h = 1e-5
    g_fd = np.array([(f(v + h * e) - f(v - h * e)) / (2 * h) for e in E])
    np.testing.assert_allclose(g, g_fd, atol=1e-7)
    h = 1e-3
    H_fd = np.array([[(f(v + h * (ei + ej)) - f(v + h * (ei - ej)) - f(v - h * (ei - ej)) + f(v - h * (ei + ej)))
                      / (4 * h * h) for ej in E] for ei in E])
    np.testing.assert_allclose(H, H_fd, atol=1e-5)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


@pytest.mark.parametrize("Ks,N,miss,seed", DESIGNS)
def test_gradient_same_with_and_without_hessian(Ks, N, miss, seed):
    b, X = _design(Ks, N, miss, seed)
    ll0, g0, H0 = _esf.cond_loglik(X, b)
    ll1, g1, _ = _esf.cond_loglik(X, b, hessian=True)
    assert H0 is None
    assert ll0 == pytest.approx(ll1, abs=1e-12)
    np.testing.assert_allclose(g0, g1, atol=1e-12)


def test_batching_does_not_change_results():
    b, X = _design([1, 3, 2, 4, 1, 2, 2, 3], 300, 0.35, 6)
    ref = _esf.cond_loglik(X, b, hessian=True)
    tiny = _esf.cond_loglik(X, b, hessian=True, budget=50)
    assert tiny[0] == pytest.approx(ref[0], abs=1e-10)
    np.testing.assert_allclose(tiny[1], ref[1], atol=1e-10)
    np.testing.assert_allclose(tiny[2], ref[2], atol=1e-10)


def test_padded_array_input_matches_list_input():
    b, X = _design([1, 3, 2, 4], 80, 0.2, 7)
    padded = np.full((4, 4), np.nan)
    for i, x in enumerate(b):
        padded[i, :len(x)] = x
    a, c = _esf.cond_loglik(X, b, hessian=True), _esf.cond_loglik(X, padded, hessian=True)
    assert a[0] == c[0]
    np.testing.assert_array_equal(a[1], c[1])
    np.testing.assert_array_equal(a[2], c[2])


def test_persons_without_information_contribute_nothing():
    b, X = _design([2, 2, 2, 2], 60, 0.0, 8)
    extra = np.array([[0, 0, 0, 0], [2, 2, 2, 2], [1, np.nan, np.nan, np.nan], [np.nan] * 4], float)
    a = _esf.cond_loglik(X, b, hessian=True)
    c = _esf.cond_loglik(np.vstack([X, extra]), b, hessian=True)
    assert c[0] == pytest.approx(a[0], abs=1e-12)
    np.testing.assert_allclose(c[1], a[1], atol=1e-12)
    np.testing.assert_allclose(c[2], a[2], atol=1e-12)


def test_extreme_parameters_stay_finite():
    # thresholds spread over +-40 logits: linear-space ESFs overflow here; log space must not
    rng = np.random.default_rng(9)
    b = [np.sort(rng.uniform(-40, 40, 3)) for _ in range(30)]
    X = rng.integers(0, 4, (200, 30)).astype(float)
    ll, g, H = _esf.cond_loglik(X, b, hessian=True)
    assert np.isfinite(ll) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))


def test_invalid_responses_raise():
    b, X = _design([2, 2, 2], 20, 0.0, 10)
    X[0, 0] = 3
    with pytest.raises(ValueError):
        _esf.cond_loglik(X, b)


def test_sample_given_totals_matches_conditional_distribution():
    from scipy.stats import chi2

    rng = np.random.default_rng(2)
    Ks, t = [2, 2, 2, 2], 3
    b = [np.sort(rng.normal(0, 1, K)) for K in Ks]
    X = np.tile(np.array([2, 1, 0, 0], float), (60000, 1))  # every person: total 3
    S = _esf.sample_given_totals(X, b, rng)
    assert np.all(S.sum(1) == t)
    le = [np.concatenate([[0.0], -np.cumsum(x)]) for x in b]
    pats = [c for c in itertools.product(*[range(K + 1) for K in Ks]) if sum(c) == t]
    lp = np.array([sum(le[i][c] for i, c in enumerate(cs)) for cs in pats])
    p = np.exp(lp - logsumexp(lp))
    counts = np.array([np.sum(np.all(S == np.array(cs), axis=1)) for cs in pats])
    stat = ((counts - p * len(S)) ** 2 / (p * len(S))).sum()
    assert chi2.sf(stat, len(pats) - 1) > 0.001


def test_sample_given_totals_keeps_missingness_and_totals():
    rng = np.random.default_rng(3)
    b, X = _design([1, 3, 2, 4], 200, 0.3, 11)
    S = _esf.sample_given_totals(X, b, rng)
    np.testing.assert_array_equal(np.isnan(S), np.isnan(X))
    np.testing.assert_array_equal(np.nansum(S, 1), np.nansum(X, 1))


def test_fit_cml_chord_reaches_the_newton_maximum():
    from raschpy import _rcl

    b, X = _design([1, 3, 2, 4, 2, 2], 400, 0.2, 12)
    Ks = [len(x) for x in b]
    p = sum(Ks)
    A = _rcl.zero_sum_basis(p)
    b0 = np.zeros(p)
    newton = _esf.fit_cml(X, A, b0, Ks)
    # a fixed matrix from nearby estimates (the start), as for same-design refits
    _, _, H0 = _esf.cond_loglik(X, [np.zeros(K) for K in Ks], hessian=True)
    chord = _esf.fit_cml(X, A, b0, Ks, fixed_hessian=H0)
    assert chord["solver"] == "chord" and chord["converged"] and chord["hessian"] is None
    np.testing.assert_allclose(chord["b"], newton["b"], atol=1e-6)
    assert chord["ll"] == pytest.approx(newton["ll"], abs=1e-9)


def test_prepared_structure_gives_identical_results():
    b, X = _design([1, 3, 2, 4, 2], 300, 0.3, 13)
    prep = _esf.prepare(X, [len(x) for x in b])
    for hess in (False, True):
        a = _esf.cond_loglik(X, b, hessian=hess)
        c = _esf.cond_loglik(X, b, hessian=hess, prepared=prep)
        assert a[0] == c[0]
        np.testing.assert_array_equal(a[1], c[1])
        if hess:
            np.testing.assert_array_equal(a[2], c[2])
    with pytest.raises(ValueError, match="different number of thresholds"):
        _esf.cond_loglik(X, [np.zeros(1)] * 5, prepared=prep)


def _brute_conditional(X, b):
    """Per-person P(X_i = k | r) by enumerating every pattern with the person's total."""
    Ks = [len(x) for x in b]
    le = [np.concatenate([[0.0], -np.cumsum(x)]) for x in b]
    probs = {}
    for n, x in enumerate(X):
        o = np.nonzero(~np.isnan(x))[0]
        if len(o) < 2:
            continue
        r = int(np.nansum(x))
        if r == 0 or r == sum(Ks[i] for i in o):
            continue
        pats = [cs for cs in itertools.product(*[range(Ks[i] + 1) for i in o]) if sum(cs) == r]
        lp = np.array([sum(le[i][c] for i, c in zip(o, cs)) for cs in pats])
        p = np.exp(lp - logsumexp(lp))
        own = sum(le[i][int(x[i])] for i in o)
        probs[n] = (o, pats, p, lp, own, r)
    return probs


def test_conditional_moments_match_brute_force():
    b, X = _design([1, 3, 2, 2, 1], 150, 0.25, 14)
    out = _esf.conditional_moments(X, b, tables=True)
    Ks = [len(x) for x in b]
    for n, (o, pats, p, _, _, r) in _brute_conditional(X, b).items():
        for j, i in enumerate(o):
            pk = np.array([p[[cs[j] == k for cs in pats]].sum() for k in range(Ks[i] + 1)])
            e = (pk * np.arange(Ks[i] + 1)).sum()
            d = np.arange(Ks[i] + 1) - e
            assert out["E"][n, i] == pytest.approx(e, abs=1e-12)
            assert out["V"][n, i] == pytest.approx((pk * d**2).sum(), abs=1e-12)
            assert out["M4"][n, i] == pytest.approx((pk * d**4).sum(), abs=1e-12)


def test_person_pattern_pvalues_match_exact():
    b, X = _design([1, 2, 2, 3, 1], 40, 0.2, 15)
    mc = _esf.person_pattern_pvalues(X, b, samples=20000, rng=1)
    brute = _brute_conditional(X, b)
    for n in range(len(X)):
        if n in brute:
            _, _, p, lp, own, _ = brute[n]
            assert mc[n] == pytest.approx(p[lp <= own + 1e-12].sum(), abs=0.02)
        else:
            assert np.isnan(mc[n])


def test_gk_gamma_matches_vcdextra():
    # vcdExtra::GKgamma(matrix(c(10, 4, 2, 3, 8, 5, 1, 2, 9), 3)): C = 401, D = 67
    t = np.array([[10, 3, 1], [4, 8, 2], [2, 5, 9]], float)
    g, se = _esf.gk_gamma(t)
    assert g == pytest.approx((401 - 67) / (401 + 67), abs=1e-12)
    assert se == pytest.approx(0.125936754996, abs=1e-9)  # vcdExtra sigma


def test_westfall_young_step_down():
    t_obs = np.array([3.0, 1.0, 2.0])
    t_sim = np.array([[0.5, 2.5, 0.1], [3.5, 0.2, 0.3], [0.1, 0.1, 2.2], [1.2, 0.9, 0.4]])
    p = _esf.westfall_young(t_obs, t_sim)
    # step 1 (item 0, |t| = 3): max over all items >= 3 in row 1 only -> (1 + 1) / 5
    # step 2 (item 2, |t| = 2): max over items {2, 1} >= 2 in rows 0, 2 -> 3 / 5
    # step 3 (item 1, |t| = 1): max over {1} >= 1 in row 0 -> 2 / 5, raised to 3 / 5 by monotonicity
    np.testing.assert_allclose(p, [2 / 5, 3 / 5, 3 / 5])
    marginal = (1 + (np.abs(t_sim) >= np.abs(t_obs)).sum(0)) / 5
    assert np.all(p >= marginal - 1e-12)


def test_hdi():
    lo, hi = _esf.hdi([5, 1, 2, 3, 4, 100], width=0.8)  # shortest interval holding 5 of 6 values
    assert (lo, hi) == (1, 5)


def test_partial_gamma_and_stratified_kendall():
    rng = np.random.default_rng(0)
    T = rng.integers(0, 20, (4, 3))
    np.testing.assert_allclose(_esf.partial_gk_gamma(T), _esf.gk_gamma(T))
    # partial gamma = brute-force pair counts within strata
    S = rng.integers(0, 5, (3, 3, 4)).astype(float)
    C = D = 0
    for s in S:
        cells = [(i, j) for i in range(3) for j in range(4) for _ in range(int(s[i, j]))]
        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                di, dj = cells[a][0] - cells[b][0], cells[a][1] - cells[b][1]
                C += di * dj > 0
                D += di * dj < 0
    assert _esf.partial_gk_gamma(S)[0] == pytest.approx((C - D) / (C + D), rel=1e-12)
    assert _esf.partial_gk_gamma_batch(S[None])[0] == pytest.approx((C - D) / (C + D), rel=1e-12)
    c, d, var = _esf.stratified_kendall_s(S)
    assert (c, d) == (C, D)
    # exact permutation variance of S = C - D against all within-stratum permutations (small strata)
    x = np.array([0, 1, 1, 2, 0, 2, 1]); y = np.array([0, 0, 1, 1, 2, 1, 0]); st = np.array([0, 0, 0, 0, 1, 1, 1])
    vals = []
    for p0 in itertools.permutations(range(4)):
        for p1 in itertools.permutations(range(3)):
            xx = x.copy()
            xx[:4] = x[:4][list(p0)]
            xx[4:] = x[4:][list(p1)]
            Tt = np.zeros((2, 3, 3))
            np.add.at(Tt, (st, xx, y), 1)
            cc, dd, _ = _esf.stratified_kendall_s(Tt)
            vals.append(cc - dd)
    Tt = np.zeros((2, 3, 3))
    np.add.at(Tt, (st, x, y), 1)
    assert _esf.stratified_kendall_s(Tt)[2] == pytest.approx(np.var(vals), rel=1e-12)
    assert np.mean(vals) == pytest.approx(0, abs=1e-12)


def test_targeting_core_reproduces_kreiner_christensen_example():
    # Rasch Models in Health (2013), ch. 4, Table 4.1 and section 4.6: the DE subscale's score
    # parameters gamma_r, its WLEs, and the population results for E(theta) = -0.516, SD = 0.855
    g = np.array([1.00, 15.70, 83.96, 276.58, 690.84, 1344.82, 2088.62, 2635.91, 2582.94,
                  1921.42, 1130.50, 510.38, 177.95, 46.78, 8.56, 1.00])
    wle_book = np.array([-3.512, -2.064, -1.437, -1.066, -0.787, -0.545, -0.316, -0.084, 0.156,
                         0.404, 0.655, 0.911, 1.185, 1.503, 1.933, 2.803])
    lg = np.log(g)
    wle = _esf.score_wle(lg)
    # gamma_r is printed to 2 decimals, which moves the top totals' estimates slightly
    np.testing.assert_allclose(wle[:14], wle_book[:14], atol=0.005)
    pp = _esf.population_properties(lg, -0.516, 0.855)
    assert pp["target"] == pytest.approx(-0.14, abs=0.005)
    assert pp["max_information"] == pytest.approx(4.05, abs=0.01)
    assert pp["expected_information"] == pytest.approx(3.26, abs=0.01)
    assert pp["expected_information_ratio"] > 0.8
    assert pp["expected_abs_bias"] == pytest.approx(0.011, abs=0.002)
    assert pp["expected_rmse"] == pytest.approx(0.592, abs=0.006)
    assert pp["rho1"] == pytest.approx(0.71, abs=0.005)
    assert pp["rho2"] == pytest.approx(pp["rho1"], abs=1e-9)  # independent given theta
    assert pp["rho6"] == pytest.approx(0.77, abs=0.01)
    # the score distribution and its moments
    E, V, _ = _esf.score_moments(lg, [0.3])
    P = _esf.score_distribution(lg, [0.3])[0]
    r = np.arange(16)
    assert E[0] == pytest.approx(P @ r) and V[0] == pytest.approx(P @ (r - E[0]) ** 2)
    np.testing.assert_allclose(_esf.log_gamma([np.array([0.2]), np.array([-0.5, 0.4])]),
                               np.log([1, np.exp(-0.2) + np.exp(0.5), np.exp(-0.2 + 0.5) + np.exp(0.1),
                                       np.exp(-0.2 + 0.1)]))
