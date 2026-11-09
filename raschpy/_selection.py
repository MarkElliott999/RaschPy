"""
RSM-within-PCM comparison for model_selection() without leaving the object's estimator.

One composite likelihood serves cp-full-rcl and PAIR (method='log-lik'): the full pattern set of every item pair,
(a, b) against (a - 1, b + 1), whose unsmoothed comparisons are _rcl.pcm_comparisons() (RSM's full_comparisons()
gives the same set). The statistic is twice the difference of that composite log-likelihood at the two models'
estimates (RSM's as delta_i + tau_h). Its null distribution is a weighted sum of chi-square(1) variables (Kent,
1982) with weights from the sensitivity H and the variability J (the person scores' covariance) at the PCM point;
the p-value is that sum's exact tail probability (weighted_chi2_sf), and the information
criteria use the penalty tr(J H^-1) (Varin and Vidoni, 2005). composite_andersen() is Andersen's test of
the same likelihood across groups of persons, for global_fit(method='andersen') under 'cp-full-rcl'.
"""

import warnings

import numpy as np
from scipy.stats import chi2

from raschpy import _esf, _rcl


def _ll(D, t, u, b):
    eta = D @ b
    return float((-t * np.logaddexp(0, -eta) - u * np.logaddexp(0, eta)).sum())


def _info(D, t, u, b):
    s = 1.0 / (1.0 + np.exp(-(D @ b)))
    w = (t + u) * s * (1.0 - s)
    return np.asarray((D.T @ D.multiply(w[:, None])).todense())


def bases(I, K):
    """(A0, E): RSM's directions delta_i + tau_h (zero-sum) and the item-specific threshold deviations, in the
    stacked uncentred thresholds (item by item)."""
    Zi, Zk = _rcl.zero_sum_basis(I), _rcl.zero_sum_basis(K)
    A0 = np.hstack([np.kron(Zi, np.ones((K, 1))), np.kron(np.ones((I, 1)), Zk)])
    return A0, np.kron(Zi, Zk)


def _pinv_sym(M, tol=1e-8):
    w, Q = np.linalg.eigh((M + M.T) / 2)
    k = w > tol * max(w.max(), 1e-300)
    return Q[:, k] @ np.diag(1.0 / w[k]) @ Q[:, k].T


def composite_lr(X, K, b_rsm, b_pcm, cpat=False):
    """
    Composite likelihood ratio test of RSM within PCM. X (persons x items, NaN missing), b_rsm and b_pcm the
    stacked uncentred thresholds (item by item); the comparisons are PCM's full pattern set, or CPAT's patterns
    with cpat=True ('cpat-rcl'; see _rcl.pcm_comparisons). Returns a dict: statistic, df (the tested directions the
    comparisons inform), scale and df of the Satterthwaite approximation (for reference), p, the weights' range, the two
    composite log-likelihoods, and CL-AIC / CL-BIC penalties tr(J H^-1) for each model.
    """
    X = np.asarray(X, float)
    N, I = X.shape
    Ks = [int(K)] * I
    D, t, u = _rcl.pcm_comparisons(X, Ks, cpat)
    b0 = np.asarray(b_rsm, float) - np.mean(b_rsm)
    b1 = np.asarray(b_pcm, float) - np.mean(b_pcm)
    ll0, ll1 = _ll(D, t, u, b0), _ll(D, t, u, b1)
    A0, E = bases(I, K)
    B = np.hstack([A0, E])
    q0 = A0.shape[1]

    def sens_var(b, Bm):
        S = _rcl.pcm_person_scores(X, Ks, b, cpat) @ Bm
        return Bm.T @ _info(D, t, u, b) @ Bm, np.cov(S, rowvar=False) * N

    H1, J1 = sens_var(b1, B)
    r = composite_lr_core(2.0 * (ll1 - ll0), H1, J1, q0)
    H0, J0 = sens_var(b0, A0)
    r.update(cl_rsm=ll0, cl_pcm=ll1, penalty_rsm=penalty(H0, J0), penalty_pcm=penalty(H1, J1))
    return r


def penalty(H, J):
    """CL-AIC / CL-BIC's parameter count tr(J H^-1) (Varin and Vidoni, 2005), H^-1 over the directions H informs."""
    return float(np.trace(J @ _pinv_sym(H)))


def _imhof_sf(x, w):
    """Imhof's (1961) inversion: P = 1/2 + (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du, theta(u) = sum arctan(w u)
    / 2 - x u / 2, rho(u) = prod (1 + w^2 u^2)^(1/4). The integrand decays like u^(-1 - d/2); used for d >= 3.
    Returns (p, absolute error estimate)."""
    from scipy.integrate import quad

    def f(u):
        th = 0.5 * np.arctan(w * u).sum() - 0.5 * x * u
        return np.sin(th) / (u * np.exp(0.25 * np.log1p((w * u) ** 2).sum()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quad's IntegrationWarning: its error estimate is returned instead
        v, err = quad(f, 0, np.inf, epsabs=1e-13, limit=1000)
    return float(min(max(0.5 + v / np.pi, 0.0), 1.0)), float(err / np.pi)


def _two_sf(x, a, b):
    """P(a U + b V > x), U, V chi-square(1), a >= b > 0: P(U > x/a) + int_0^(x/a) f_1(u) P(V > (x - a u)/b) du, the
    chi-square(1) density's u^(-1/2) singularity integrated exactly (quad's algebraic weight)."""
    from scipy.integrate import quad

    c = 1.0 / np.sqrt(2 * np.pi)
    v, _ = quad(lambda u: c * np.exp(-u / 2) * chi2.sf((x - a * u) / b, 1), 0, x / a, weight="alg",
                wvar=(-0.5, 0.0), epsabs=0, epsrel=1e-11, limit=200)
    return float(min(chi2.sf(x / a, 1) + v, 1.0))


def _ruben_sf(x, w, tol=1e-12, max_terms=20_000):
    """
    Ruben's (1962) series: with beta = min(w), sum_k a_k P(chi-square(d + 2k) > x / beta), a_0 = prod (beta /
    w_j)^(1/2), a_k = sum_{r<k} g_{k-r} a_r / (2k), g_m = sum_j (1 - beta / w_j)^m. Every a_k is non-negative and
    they sum to 1, so the series has no cancellation and keeps its relative accuracy in the far tail; the
    chi-square tails come from P(chi2(v + 2) > y) = P(chi2(v) > y) + (y/2)^(v/2) e^(-y/2) / Gamma(v/2 + 1). It
    converges geometrically at rate 1 - min(w) / max(w). Returns None if it has not converged in max_terms.
    """
    from scipy.special import gammaln

    d = len(w)
    beta = float(w.min())
    q = 1.0 - beta / w
    y = x / beta
    log_a0 = 0.5 * float(np.log(beta / w).sum())
    n = 256
    g, b = np.zeros(n), np.zeros(n)  # b_k = a_k / (a_0 e^log_s)
    b[0], log_s = 1.0, 0.0
    qm = np.ones(d)
    tk = float(chi2.sf(y, d))
    log_f = 0.5 * d * np.log(y / 2) - y / 2 - gammaln(d / 2 + 1)
    total, mass, prev = tk, 1.0, tk
    for k in range(1, max_terms):
        if k >= n:
            g, b, n = np.concatenate([g, np.zeros(n)]), np.concatenate([b, np.zeros(n)]), 2 * n
        qm = qm * q
        g[k] = qm.sum()
        b[k] = float(np.dot(g[k:0:-1], b[:k])) / (2 * k)
        tk = min(tk + np.exp(log_f), 1.0)
        log_f += np.log(y / 2) - np.log((d + 2 * k) / 2)
        term = b[k] * tk
        total += term
        mass += b[k]
        if mass > 1e200:
            b[: k + 1] *= 1e-200
            total, mass, term, prev = total * 1e-200, mass * 1e-200, term * 1e-200, prev * 1e-200
            log_s += 200 * np.log(10)
        ratio = term / prev if prev > 0 else 1.0
        prev = term
        if 1.0 - np.exp(log_a0 + log_s + np.log(mass)) < 1e-10 and ratio < 1 and term * ratio / (1 - ratio) < tol * total:
            return float(min(np.exp(log_a0 + log_s + np.log(total)), 1.0)) if total > 0 else 0.0
    return None


def weighted_chi2_sf(x, weights):
    """
    P(sum_j w_j chi-square(1) > x) for positive weights w, exactly up to numerical tolerance: one weight, or equal
    weights, chi-square; two weights, a one-dimensional integral (_two_sf); three or more with max(w) / min(w) <=
    300, Ruben's positive series (_ruben_sf; relative accuracy about 1e-12 in every tail); more unequal weights,
    Imhof's inversion (_imhof_sf; absolute accuracy about 1e-10, its integrand decaying like u^(-1 - d/2)), which
    is also used if Ruben's series has not converged (far tails with near-300 ratios). So with very unequal
    weights, p-values below about 1e-9 are accurate only to about 1e-10 and may be reported as 0.
    """
    w = np.sort(np.asarray(weights, float))
    d = len(w)
    if x <= 0:
        return 1.0
    x, w = x / w[-1], w / w[-1]  # the tail probability is unchanged by a common scale
    if d == 1 or w[-1] <= w[0] * (1 + 1e-12):
        return float(chi2.sf(x / w[0], d))
    if d == 2:
        return _two_sf(x, w[1], w[0])
    if w[-1] <= 300 * w[0]:
        p = _ruben_sf(x, w)
        if p is not None:
            return p
    return _imhof_sf(x, w)[0]


def composite_lr_core(diff, H, J, q0):
    """
    Composite LR test of a smaller model within a larger one, from twice the composite log-likelihood difference
    diff (larger minus smaller; floored at 0), and the larger model's sensitivity H and variability J in
    coordinates whose first q0 are the smaller model's directions and the rest the tested ones. The weights are
    the eigenvalues of (H^-1)_pp^-1 (H^-1 J H^-1)_pp over the tested directions the comparisons inform (Kent,
    1982); the p-value is that sum's exact tail probability (weighted_chi2_sf). Returns statistic,
    df, scale and df_satterthwaite (the Satterthwaite approximation's, for reference), p, weight_min, weight_max.
    """
    Hinv = _pinv_sym(H)
    G = Hinv @ J @ Hinv
    ps = slice(q0, H.shape[0])
    wp, Qp = np.linalg.eigh(Hinv[ps, ps])
    kp = wp > 1e-8 * max(wp.max(), 1e-300)
    lam = np.linalg.eigvals(np.diag(1.0 / wp[kp]) @ (Qp[:, kp].T @ G[ps, ps] @ Qp[:, kp])).real
    stat = max(float(diff), 0.0)
    scale = float((lam**2).sum() / lam.sum())
    nu = float(lam.sum() ** 2 / (lam**2).sum())
    return dict(
        statistic=stat,
        df=int(len(lam)),
        scale=scale,
        df_satterthwaite=nu,
        p=weighted_chi2_sf(stat, lam),
        weight_min=float(lam.min()),
        weight_max=float(lam.max()),
    )


def bt_restricted_fit(mat, A, tol=1e-10, max_iter=200):
    """
    Maximise PAIR's Bradley-Terry likelihood of its smoothed threshold-pair matrix, sum_{a != b} mat[a, b]
    log sigma(b_b - b_a) (cell a's strength is minus its uncentred threshold), over b = A th: the RSM fit
    within PAIR's own objective. Returns the stacked uncentred thresholds.
    """
    n = np.array(mat, float)
    np.fill_diagonal(n, 0.0)
    Nn = n + n.T
    th = np.zeros(A.shape[1])

    def ev(th):
        b = A @ th
        p = 1.0 / (1.0 + np.exp(-(b[None, :] - b[:, None])))
        ll = float((n * np.log(np.clip(p, 1e-300, None))).sum())
        r = n - Nn * p
        g = A.T @ (r.sum(0) - r.sum(1))
        w = Nn * p * (1.0 - p)
        np.fill_diagonal(w, 0.0)
        return ll, g, A.T @ (np.diag(w.sum(1)) - w) @ A

    ll, g, H = ev(th)
    for _ in range(max_iter):
        step = np.linalg.solve(H, g)
        a = 1.0
        while True:
            ll2, g2, H2 = ev(th + a * step)
            if ll2 >= ll - 1e-12 * abs(ll) or a < 1e-8:
                break
            a /= 2.0
        th = th + a * step
        change = float(np.abs(A @ (a * step)).max())
        ll, g, H = ll2, g2, H2
        if change < tol:
            break
    return A @ th


def threshold_difference(b_rsm, b_pcm, I, K):
    """Sum over items and thresholds of the squared difference between PCM's and RSM's fitted thresholds, each
    item's thresholds taken relative to that item's central location (b_rsm, b_pcm stacked uncentred thresholds,
    item by item)."""
    c0 = np.asarray(b_rsm, float).reshape(I, K)
    c1 = np.asarray(b_pcm, float).reshape(I, K)
    d = (c1 - c1.mean(1, keepdims=True)) - (c0 - c0.mean(1, keepdims=True))
    return float((d**2).sum())


def bootstrap_p(X, K, b_rsm_null, statistic, samples, rng, track=None, observed=None):
    """
    Conditional parametric bootstrap p-value: datasets simulated given each person's total and observed items
    under RSM with uncentred thresholds b_rsm_null (stacked), statistic(X') recomputed on each (the route's own
    fits), p = (1 + #{T* >= T}) / (B + 1) with B the resamples whose statistic could be computed. observed is
    the statistic of X when already computed (from the object's own fits); None computes statistic(X).
    """
    import pandas as pd

    names = list(X.columns)
    I = len(names)
    thr = [np.asarray(b_rsm_null, float)[i * K:(i + 1) * K] for i in range(I)]
    Xn = np.array(X, float)
    T = statistic(X) if observed is None else observed
    sims = []
    it = range(samples) if track is None else track(range(samples))
    for _ in it:
        Xs = pd.DataFrame(_esf.sample_given_totals(Xn, thr, rng), columns=names, index=X.index)
        try:
            with warnings.catch_warnings():  # the simulated fits' warnings repeat the observed data's
                warnings.simplefilter("ignore")
                sims.append(statistic(Xs))
        except Exception:
            pass
    sims = np.asarray(sims)
    return T, float((1 + (sims >= T).sum()) / (len(sims) + 1)), int(len(sims))


def composite_andersen(Xs, Ks, b_groups, b0, Bm):
    """
    Andersen's test on the same composite likelihood (global_fit(method='andersen') under 'cp-full-rcl'):
    groups of persons Xs (each persons x items, NaN missing; maximum scores Ks) with their own fits b_groups and
    the pooled fit b0 (stacked uncentred thresholds), in the coordinates Bm (the model's free directions). The
    composite LR is 2 (sum_g cl_g(b_g) - sum_g cl_g(b0)); the groups have different persons, so H and J are
    block-diagonal. Its null distribution is a weighted sum of chi-square(1) variables (Kent, 1982), the
    weights the non-zero eigenvalues of J^1/2 (H^+ - A (A'HA)^+ A') J^1/2 with A stacking identity blocks
    (p: its exact tail, weighted_chi2_sf). The adjusted LR (Pace, Salvan and Sartori, 2011) is LR x robust Wald / naive Wald on the
    group differences psi = (phi_g - phi_1, g >= 2), with df the rank of psi's sandwich covariance. H and J are
    evaluated at the group fits. Returns a dict: lr, adjusted, df, scale, df_satterthwaite, p (weighted chi-square tail),
    weight_min, weight_max, cl (per group at its own fit) and cl_pooled (per group at b0).
    """
    Ks = [int(k) for k in Ks]
    Hs, Js, phis, cl, cl0 = [], [], [], [], []
    for X, b in zip(Xs, b_groups):
        X = np.asarray(X, float)
        D, t, u = _rcl.pcm_comparisons(X, Ks)
        bc, bc0 = np.asarray(b, float) - np.mean(b), np.asarray(b0, float) - np.mean(b0)
        cl.append(_ll(D, t, u, bc))
        cl0.append(_ll(D, t, u, bc0))
        Hs.append(Bm.T @ _info(D, t, u, bc) @ Bm)
        Js.append(np.cov(_rcl.pcm_person_scores(X, Ks, bc) @ Bm, rowvar=False) * len(X))
        phis.append(np.linalg.lstsq(Bm, bc, rcond=None)[0])
    out = composite_andersen_core(max(2.0 * (sum(cl) - sum(cl0)), 0.0), Hs, Js, phis)
    out.update(cl=cl, cl_pooled=cl0)
    return out


def composite_andersen_core(lr, Hs, Js, phis):
    """
    composite_andersen()'s statistics from the composite LR and each group's pieces in the model's free
    coordinates: the information H_g and the persons' score covariance J_g at the group's fit, and its
    estimates phi_g. Returns a dict: lr, adjusted, df, scale and df_satterthwaite (the Satterthwaite
    approximation's, for reference), p (the weighted chi-square tail, weighted_chi2_sf), weight_min, weight_max.
    """
    from scipy.linalg import block_diag

    q, G = Hs[0].shape[0], len(Hs)
    H, J = block_diag(*Hs), block_diag(*Js)
    # Kent weights for the restriction b_g = b (every group)
    A = np.vstack([np.eye(q)] * G)
    M = _pinv_sym(H) - A @ _pinv_sym(A.T @ H @ A) @ A.T
    w, Q = np.linalg.eigh((J + J.T) / 2)
    Jh = Q @ np.diag(np.sqrt(np.clip(w, 0, None))) @ Q.T
    lam = np.linalg.eigvalsh(Jh @ M @ Jh)
    lam = lam[lam > 1e-8 * max(lam.max(), 1e-300)]
    # adjusted LR in the coordinates (phi_1, psi): C maps them to the stacked group coordinates
    C = np.zeros((G * q, G * q))
    for g in range(G):
        C[g * q:(g + 1) * q, :q] = np.eye(q)
        if g:
            C[g * q:(g + 1) * q, g * q:(g + 1) * q] = np.eye(q)
    Hc_inv = _pinv_sym(C.T @ H @ C)
    S = Hc_inv @ (C.T @ J @ C) @ Hc_inv
    ps = slice(q, G * q)
    psi = np.concatenate([phis[g] - phis[0] for g in range(1, G)])
    naive = float(psi @ _pinv_sym(Hc_inv[ps, ps]) @ psi)
    robust = float(psi @ _pinv_sym(S[ps, ps]) @ psi)
    Sp = S[ps, ps]
    df = int(np.linalg.matrix_rank(Sp, tol=1e-8 * max(np.abs(Sp).max(), 1e-300)))
    scale = float((lam**2).sum() / lam.sum())
    nu = float(lam.sum() ** 2 / (lam**2).sum())
    return dict(
        lr=lr,
        adjusted=lr * robust / naive if naive > 0 else 0.0,
        df=df,
        scale=scale,
        df_satterthwaite=nu,
        p=weighted_chi2_sf(lr, lam),
        weight_min=float(lam.min()),
        weight_max=float(lam.max()),
    )


def thin_categories(X, K):
    """
    The score categories RSM's thin-width rule acts on (_rcl.width_ridges: a category width with fewer than
    _rcl.THIN_COUNT patterns, or all its patterns on one side) and the middle categories nobody used
    (_rcl.empty_widths). Returns (thin widths, unused middle categories), widths numbered 1..K-1.
    """
    Xr = np.asarray(X, float)[None]
    counts = _rcl.pattern_counts(Xr, K)
    cat = _rcl.category_counts(Xr, K)
    unused = [j for j in range(1, K) if cat[j] == 0]
    rho = _rcl.width_ridges(counts[1], counts[2], K) if K - 1 >= 2 else np.zeros(K - 1)  # (W, Nn, Dd, V)
    thin = [k + 1 for k in np.nonzero(rho)[0]]
    return thin, unused
