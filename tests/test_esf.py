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
