"""
Conditional (elementary symmetric function, ESF) log-likelihood for SLM, PCM and RSM, with its exact
gradient and Hessian.

Each item i has thresholds b[i, h] (h = 1..K_i): P(X_i = k) is proportional to
exp(k theta - sum_{h <= k} b[i, h]). Given a person's total r over the items they responded to, theta cancels:

    P(x | r) = prod_i eps[i, x_i] / gamma_r,    log eps[i, k] = -sum_{h <= k} b[i, h],  log eps[i, 0] = 0,

where gamma_r is the ESF of order r: the coefficient of z^r in prod_i sum_k eps[i, k] z^k. Items may have
different numbers of categories. Persons with a single response or an extreme total contribute nothing.

Computation (exact, in log space, no subtractions):
- Persons are grouped by which items they responded to (bit-packed rows hashed), and each group's persons are
  counted by total ("cells": group x total). Groups with the same number of items are processed together in
  vectorised batches; each batch is limited by memory and by the spread of its groups' totals.
- gamma for each group from prefix and suffix products of the items' generating polynomials (log-space
  convolutions).
- Per-cell category probabilities P(X_i = k | r) = eps[i, k] gamma^(-i)_{r-k} / gamma_r from prefix (*) suffix,
  computed only at the totals the group's cells need.
- Gradient and Hessian need only sums over totals weighted by u_r = n_r / gamma_r (n_r persons with total r).
  These weights are pushed back through the items once per group, Q_b(q) = sum_t suffix_{b+1}[t] u_{q+t}, with
  Q_b(q) = sum_j eps[b+1, j] Q_{b+1}(q + j), so that

      sum_r n_r P(X_a = k | r) = eps[a, k] sum_s prefix_a[s] Q_a(s + k)
      sum_r n_r P(X_a = k, X_b = l | r) = eps[a, k] eps[b, l] sum_s M_ab[s] Q_b(s + k + l)

  with M_ab the running product of the items before b except a. Each item pair then costs a few inner products
  instead of a full leave-two-out convolution.
- Derivatives are taken with respect to log eps and mapped to b by d log eps[i, k] / d b[i, h] = -1{h <= k}
  (reversed cumulative sums).

The Hessian is minus the summed conditional covariance of the category indicators.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NEG = -np.inf
BUDGET = 1.5e8  # target size (floats) of the largest temporary array; batches are split to stay near it
MIN_BATCH = 16  # groups per batch below which batches are not split by total (Python overhead dominates)


def _lse(v, axis=-1):
    """log-sum-exp along axis (all -inf gives -inf)."""
    m = v.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.log(np.exp(v - m).sum(axis=axis)) + np.squeeze(m, axis)


def _logconv_small(g, h):
    """log(exp g (*) exp h) along the last axis, h short (one item's categories); leading axes broadcast."""
    a, b = g.shape[-1], h.shape[-1]
    lead = np.broadcast_shapes(g.shape[:-1], h.shape[:-1])
    out = np.full(lead + (a + b - 1,), NEG)
    with np.errstate(invalid="ignore"):
        for k in range(b):
            seg = out[..., k : k + a]
            np.logaddexp(seg, g + h[..., k : k + 1], out=seg)
    return out


def _selconv_range(P, S, lo, m):
    """Rows of log(exp P (*) exp S) at totals lo + 0..m-1 (P (R, a), S (R, c), lo (R,)) -> (R, m)."""
    if P.shape[1] > S.shape[1]:
        P, S = S, P
    R, a = P.shape
    c = S.shape[1]
    t = lo[:, None] + np.arange(m)[None, :]
    si = t[:, :, None] - np.arange(a)[None, None, :]
    ok = (si >= 0) & (si < c)
    Sg = np.take_along_axis(
        S[:, None, :], np.clip(si, 0, c - 1).reshape(R, 1, -1), axis=2
    ).reshape(R, m, a)
    return _lse(np.where(ok, P[:, None, :] + Sg, NEG), axis=2)


def _patterns(obs):
    """Unique rows of a boolean matrix by hashing bit-packed rows: (unique rows, inverse index)."""
    pk = np.ascontiguousarray(np.packbits(obs, axis=1))
    v = pk.view(np.dtype((np.void, pk.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return obs[first], inv.ravel()


def _as_threshold_list(thresholds):
    """List of 1-d threshold arrays from a list/tuple of arrays or a 2-d array (NaN-padded rows)."""
    if isinstance(thresholds, np.ndarray) and thresholds.ndim == 2:
        return [row[~np.isnan(row)].astype(float) for row in thresholds]
    return [np.atleast_1d(np.asarray(x, float)) for x in thresholds]


def cond_loglik(X, thresholds, hessian=False, budget=BUDGET):
    """
    Conditional log-likelihood of responses X given each person's total over the items they responded to.

    Parameters
    ----------
    X : array (n_persons, n_items)
        Integer responses 0..K_i, NaN = no response.
    thresholds : list of 1-d arrays, or 2-d array with NaN padding
        b[i, h], h = 1..K_i, for each item (uncentred: item location plus centred threshold).
    hessian : bool
        Also return the Hessian.
    budget : float
        Target size (floats) of the largest temporary array.

    Returns
    -------
    ll : float
    grad : array (p,)
        Gradient with respect to the thresholds stacked item by item (p = sum_i K_i).
    hess : array (p, p) or None
    """
    b_list = _as_threshold_list(thresholds)
    X = np.asarray(X, float)
    I = len(b_list)
    if X.shape[1] != I:
        raise ValueError(f"X has {X.shape[1]} items but thresholds has {I}.")
    Ks = np.array([len(x) for x in b_list])
    Kmax = int(Ks.max())
    K1 = Kmax + 1
    D = I * K1
    off = np.concatenate([[0], np.cumsum(Ks)])
    p = int(off[-1])
    LE = np.full((I, K1), NEG)
    for i, x in enumerate(b_list):
        LE[i, : Ks[i] + 1] = np.concatenate([[0.0], -np.cumsum(x)])
    bidx = np.full((I, Kmax), p)  # stacked index of b[i, h]; absent h -> dummy index p
    for i, K in enumerate(Ks):
        bidx[i, :K] = off[i] + np.arange(K)

    obs = ~np.isnan(X)
    Xi = np.where(obs, X, 0).astype(np.int64)
    if np.any(Xi < 0) or np.any(Xi > Ks[None, :]):
        raise ValueError(
            "Responses must be integers from 0 to each item's number of thresholds."
        )
    r_all = Xi.sum(1)
    maxr = (obs * Ks[None, :]).sum(1)
    keep = (obs.sum(1) >= 2) & (r_all > 0) & (r_all < maxr)
    obs, Xi, r_all = obs[keep], Xi[keep], r_all[keep]
    if not keep.any():
        return 0.0, np.zeros(p), (np.zeros((p, p)) if hessian else None)

    # observed part of the log-likelihood and gradient (log-eps space)
    pr, pc = np.nonzero(obs)
    xv = Xi[pr, pc]
    ll = LE[pc, xv].sum()
    gle = np.bincount(pc * K1 + xv, minlength=D).astype(float)

    # cells (group x total), groups ordered by size, spread of totals, lowest total
    keys, inv = _patterns(obs)
    G = len(keys)
    S = int(Ks.sum())
    ck, cw = np.unique(inv.astype(np.int64) * (S + 1) + r_all, return_counts=True)
    cg, cr = ck // (S + 1), ck % (S + 1)
    top = np.zeros(G, np.int64)
    np.maximum.at(top, cg, cr)
    low = np.full(G, S + 1)
    np.minimum.at(low, cg, cr)
    span = top - low
    order = np.lexsort((low, span, keys.sum(1)))
    rank = np.empty_like(order)
    rank[order] = np.arange(G)
    keys, span, top = keys[order], span[order], top[order]
    key2 = rank[cg] * (S + 1) + cr
    o2 = np.argsort(key2)
    cg, cr, cw = key2[o2] // (S + 1), key2[o2] % (S + 1), cw[o2].astype(float)
    nobs_g = keys.sum(1)

    Hflat = np.zeros((p + 1) * (p + 1)) if hessian else None
    acc_i, acc_w, acc_n = [], [], [0]

    def flush():
        if acc_i:
            Hflat[:] += np.bincount(
                np.concatenate(acc_i),
                weights=np.concatenate(acc_w),
                minlength=(p + 1) ** 2,
            )
            acc_i.clear()
            acc_w.clear()
            acc_n[0] = 0

    def scatter(ia, ib, C):
        """Add log-eps-space blocks C (B, K1, K1) for item pairs (ia, ib), mapped to b:
        B[h, j] = sum_{k > h, l > j} C[k, l]."""
        Bm = C[:, ::-1, ::-1].cumsum(1).cumsum(2)[:, ::-1, ::-1][:, 1:, 1:]
        ri = bidx[ia][:, :, None] * (p + 1) + bidx[ib][:, None, :]
        acc_i.append(ri.ravel())
        acc_w.append(Bm.ravel())
        acc_n[0] += ri.size
        if acc_n[0] > 3e7:
            flush()

    kk = np.arange(K1)
    km = kk[:, None] + kk[None, :]
    for n in np.unique(nobs_g):
        gs = np.nonzero(nobs_g == n)[0]
        nb = max(1, int(budget / (2 * (n + 1) * (n * Kmax + 1))))
        # batches: at most nb groups, a spread of totals at most about double the first group's, and a
        # highest total at most about 1.25x the first group's
        bounds, s0 = [], 0
        while s0 < len(gs):
            s1 = min(
                s0 + nb,
                s0
                + int(
                    np.searchsorted(
                        span[gs[s0 : s0 + nb]],
                        2 * span[gs[s0]] + 2 * Kmax + 1,
                        side="right",
                    )
                ),
            )
            tl = np.maximum.accumulate(top[gs[s0:s1]])
            s1t = s0 + int(
                np.searchsorted(tl, 1.25 * top[gs[s0]] + 2 * Kmax + 1, side="right")
            )
            s1 = max(
                min(s1, s0 + MIN_BATCH), s1t, s0 + 1
            )  # split by total only beyond MIN_BATCH groups
            bounds.append((s0, s1))
            s0 = s1
        for s0, s1 in bounds:
            gb = gs[s0:s1]
            Gb = len(gb)
            item_idx = np.nonzero(keys[gb])[1].reshape(Gb, n)
            le = LE[item_idx]  # (Gb, n, K1)
            c0, c1 = np.searchsorted(cg, gb[0]), np.searchsorted(
                cg, gb[-1], side="right"
            )
            lp = np.searchsorted(gb, cg[c0:c1])  # local group of each cell
            r, w = cr[c0:c1], cw[c0:c1]
            starts = np.concatenate([[0], np.nonzero(np.diff(lp))[0] + 1])
            rmin = np.minimum.reduceat(r, starts)
            rmax = np.maximum.reduceat(r, starts)
            pre = [np.zeros((Gb, 1))]
            for j in range(n):
                pre.append(_logconv_small(pre[-1], le[:, j, :]))
            lgc = pre[n][lp, r]  # log gamma_r per cell
            ll -= (w * lgc).sum()
            W = int(rmax.max()) + 1
            # weights pushed back through the items: Q[b](q) = sum_t suffix_{b+1}[t] u_{q+t}
            U = np.full((Gb, W), NEG)
            with np.errstate(divide="ignore"):
                U[lp, r] = np.log(w) - lgc
            Q = [None] * n
            Q[n - 1] = U
            padK = np.full((Gb, Kmax), NEG)
            for b in range(n - 2, -1, -1):
                Qp = np.concatenate([Q[b + 1], padK], axis=1)
                Q[b] = _lse(
                    np.stack(
                        [Qp[:, j : j + W] + le[:, b + 1, j, None] for j in range(K1)], 0
                    ),
                    axis=0,
                )
            if not hessian:
                # expected category counts: eps[a, k] sum_s prefix_a[s] Q_a(s + k)
                for a in range(n):
                    Mv = np.full((Gb, W), NEG)
                    x = pre[a][:, :W]
                    Mv[:, : x.shape[1]] = x
                    Qa = sliding_window_view(
                        np.concatenate([Q[a], padK], 1), W, axis=1
                    )  # (Gb, K1, W)
                    with np.errstate(invalid="ignore"):
                        E = np.nan_to_num(
                            np.exp(le[:, a, :] + _lse(Mv[:, None, :] + Qa, axis=2))
                        )
                    gle -= np.bincount(
                        (item_idx[:, a][:, None] * K1 + kk[None, :]).ravel(),
                        weights=E.ravel(),
                        minlength=D,
                    )
                continue
            # per-cell probabilities P(X_j = k | r) (needed for the covariance's product term)
            suf = [np.zeros((Gb, 1))]
            for j in range(n - 1, -1, -1):
                suf.append(_logconv_small(suf[-1], le[:, j, :]))
            suf = suf[::-1]
            lo1 = rmin - Kmax
            m1 = int((rmax - lo1).max() + 1)
            P = np.empty((len(r), n, K1))
            for j in range(n):
                V = _selconv_range(pre[j], suf[j + 1], lo1, m1)
                v = V[lp[:, None], r[:, None] - kk[None, :] - lo1[lp][:, None]]
                with np.errstate(invalid="ignore"):
                    P[:, j, :] = np.nan_to_num(np.exp(le[lp, j, :] + v - lgc[:, None]))
                gle -= np.bincount(
                    (item_idx[lp, j][:, None] * K1 + kk[None, :]).ravel(),
                    weights=(w[:, None] * P[:, j, :]).ravel(),
                    minlength=D,
                )
            del suf
            Pw = P * w[:, None, None]
            pad2 = np.full((Gb, 2 * Kmax), NEG)
            Qw = [
                sliding_window_view(np.concatenate([q, pad2], 1), W, axis=1) for q in Q
            ]  # (Gb, 2K+1, W)
            for a in range(n):
                nb_ = n - a
                C = -np.add.reduceat(
                    Pw[:, a, None, :, None] * P[:, a:, None, :], starts, axis=0
                )  # (Gb, nb_, K1, K1)
                C[:, 0] += np.add.reduceat(Pw[:, a, :], starts, axis=0)[
                    :, :, None
                ] * np.eye(K1)
                if a < n - 1:
                    Sab = np.empty((Gb, nb_ - 1, 2 * Kmax + 1))
                    Mv = np.full((Gb, W), NEG)
                    x = pre[a][:, :W]
                    Mv[:, : x.shape[1]] = x
                    for t_, b in enumerate(range(a + 1, n)):
                        Sab[:, t_, :] = _lse(Mv[:, None, :] + Qw[b], axis=2)
                        if b < n - 1:
                            Mv = _logconv_small(Mv, le[:, b, :])[:, :W]
                    with np.errstate(invalid="ignore"):
                        C[:, 1:] += np.nan_to_num(
                            np.exp(
                                le[:, a, None, :, None]
                                + le[:, a + 1 :, None, :]
                                + Sab[:, :, km]
                            )
                        )
                scatter(
                    np.repeat(item_idx[:, a], nb_),
                    item_idx[:, a:].ravel(),
                    -C.reshape(-1, K1, K1),
                )
                if a < n - 1:
                    scatter(
                        item_idx[:, a + 1 :].ravel(),
                        np.repeat(item_idx[:, a], nb_ - 1),
                        -np.swapaxes(C[:, 1:], 2, 3).reshape(-1, K1, K1),
                    )
    # gradient: map log-eps space to b (d log eps[i, k] / d b[i, h] = -1{h <= k})
    G2 = gle.reshape(I, K1)
    gpad = -G2[:, ::-1].cumsum(1)[:, ::-1][:, 1:]
    grad = np.concatenate([gpad[i, :K] for i, K in enumerate(Ks)])
    if not hessian:
        return float(ll), grad, None
    flush()
    return float(ll), grad, Hflat.reshape(p + 1, p + 1)[:p, :p]


NEWTON_MAX = (
    500  # largest number of free parameters fitted by Newton-Raphson; above it, L-BFGS
)


SOLVERS = (None, "newton", "chord", "lbfgs")


def fit_cml(
    X,
    A,
    b0,
    Ks,
    tol=1e-7,
    max_iter=100,
    newton_max=NEWTON_MAX,
    fixed_hessian=None,
    solver=None,
):
    """
    Conditional maximum likelihood: maximise cond_loglik over thresholds b = A @ theta, starting from b0.

    A (p, q) maps the free parameters theta to the stacked thresholds (it carries the identification, e.g. a
    zero-sum basis). Every solver stops when the largest change in b is below tol, so all reach the same
    maximum; they differ in cost:

    - 'newton': Newton-Raphson on the exact Hessian at each step (step halved while the log-likelihood falls).
    - 'chord': one exact Hessian at the start, then Newton steps with it as a fixed matrix and the exact
      gradient at each step (no further Hessians). Falls back to 'newton' (q <= newton_max) or 'lbfgs' if
      it does not converge in max_iter steps.
    - 'lbfgs': L-BFGS-B with the exact gradient.
    - None: 'chord' (fastest, or within about 10% of the fastest, in timing tests from 30 to 1,000
      items, complete and missing data).

    fixed_hessian (p, p), e.g. the Hessian of a fit to data of the same design at nearby estimates: the
    chord method with this matrix, so no Hessian is computed at all.

    The chord method stops at tol / 100 (its convergence is linear) for a final gradient comparable to
    Newton-Raphson's.

    Returns a dict: b, theta, ll, iterations, converged, solver (the one that finished), max_gradient
    (largest |A' grad| at the end), hessian ((p, p) at b for 'newton', else None), hessians (the number of
    Hessians computed).
    """
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    from scipy.optimize import minimize

    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}.")

    def unstack(b):
        off = np.concatenate([[0], np.cumsum(Ks)])
        return [b[off[i] : off[i + 1]] for i in range(len(Ks))]

    def result(theta, ll, g, it, converged, name, H, hessians):
        return dict(
            b=A @ theta,
            theta=theta,
            ll=ll,
            iterations=it,
            converged=converged,
            solver=name,
            max_gradient=float(np.max(np.abs(A.T @ g))) if A.shape[1] else 0.0,
            hessian=H,
            hessians=hessians,
        )

    theta = np.linalg.lstsq(A, np.asarray(b0, float), rcond=None)[0]
    q = A.shape[1]
    hessians = 0
    H0 = None
    if fixed_hessian is not None and fixed_hessian.shape[0] == A.shape[0]:
        H0 = fixed_hessian
    elif solver in (None, "chord"):
        _, _, H0 = cond_loglik(X, unstack(A @ theta), hessian=True)
        hessians += 1
    if H0 is not None:
        try:
            cf = cho_factor(-(A.T @ H0 @ A))
        except LinAlgError:
            cf = None
        if cf is not None:
            ll, g, _ = cond_loglik(X, unstack(A @ theta))
            for it in range(1, max_iter + 1):
                step = cho_solve(cf, A.T @ g)
                s = 1.0
                while True:
                    th_new = theta + s * step
                    ll_new, g_new, _ = cond_loglik(X, unstack(A @ th_new))
                    if ll_new >= ll - 1e-12 * abs(ll) or s < 1e-6:
                        break
                    s /= 2
                change = np.max(np.abs(A @ (th_new - theta)))
                theta, ll, g = th_new, ll_new, g_new
                # linear convergence: stop 100x tighter than Newton for a comparable final gradient
                if change < tol * 1e-2:
                    return result(theta, ll, g, it, True, "chord", None, hessians)
        solver = "fallback"  # chord did not converge (or the matrix was unusable)
    if solver == "newton" or (solver == "fallback" and q <= newton_max):
        ll, g, H = cond_loglik(X, unstack(A @ theta), hessian=True)
        hessians += 1
        converged, it = False, 0
        while it < max_iter:
            gt, Ht = A.T @ g, A.T @ H @ A
            try:
                step = cho_solve(cho_factor(-Ht), gt)
            except LinAlgError:
                break  # not negative definite: the data do not identify every parameter
            s = 1.0
            while True:
                th_new = theta + s * step
                ll_new, g_new, H_new = cond_loglik(X, unstack(A @ th_new), hessian=True)
                hessians += 1
                if ll_new >= ll - 1e-12 * abs(ll) or s < 1e-6:
                    break
                s /= 2
            it += 1
            change = np.max(np.abs(A @ (th_new - theta)))
            theta, ll, g, H = th_new, ll_new, g_new, H_new
            if change < tol:
                converged = True
                break
        return result(theta, ll, g, it, converged, "newton", H, hessians)

    def neg(th):
        ll, g, _ = cond_loglik(X, unstack(A @ th))
        return -ll, -(A.T @ g)

    res = minimize(
        neg,
        theta,
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=50 * max_iter, maxcor=20, ftol=1e-15, gtol=1e-8),
    )
    ll, g, _ = cond_loglik(X, unstack(A @ res.x))
    return result(
        res.x, ll, g, int(res.nit), bool(res.success), "lbfgs", None, hessians
    )


def cml_covariance(X, A, b, Ks, hessian=None):
    """Covariance of the CML thresholds b = A @ theta: A (-A' H A)^{-1} A', H the Hessian of cond_loglik at b
    (computed if not given)."""
    if hessian is None:
        off = np.concatenate([[0], np.cumsum(Ks)])
        _, _, hessian = cond_loglik(
            X, [b[off[i] : off[i + 1]] for i in range(len(Ks))], hessian=True
        )
    info = -(A.T @ hessian @ A)
    return A @ np.linalg.solve(info, A.T)


def screen_items(X, Ks, per_item_categories=True):
    """
    Items whose CML estimates exist, from responses X (n_persons, n_items; NaN = no response) with K_i
    thresholds each. Repeated until nothing changes (dropping an item changes which persons contribute):

    - among the persons who contribute (at least two responses, total neither 0 nor the maximum), an item with
      an unused category has no finite CML estimate for the thresholds next to it, and is dropped;
    - item i 'beats' item j if some person could move one score point from i to j with the same total
      (x_i >= 1 and x_j <= K_j - 1, both observed); CML estimates exist only within a strongly connected set
      of items, so items outside the largest such set are dropped.

    per_item_categories=False skips the first rule (RSM: the thresholds are shared, so an item may leave a
    category unused).

    Returns (keep, reasons): keep a boolean mask over items, reasons a dict {item index: reason} for the
    dropped items ('unused category k' or 'not connected').
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    X = np.asarray(X, float)
    Ks = np.asarray(Ks)
    keep = np.ones(X.shape[1], bool)
    reasons = {}
    while True:
        idx = np.nonzero(keep)[0]
        Xs = X[:, idx]
        obs = ~np.isnan(Xs)
        Xi = np.where(obs, Xs, 0).astype(np.int64)
        tot, mx = Xi.sum(1), (obs * Ks[idx][None, :]).sum(1)
        contrib = (obs.sum(1) >= 2) & (tot > 0) & (tot < mx)
        changed = False
        for j, i in enumerate(idx if per_item_categories else []):
            col = Xi[contrib & obs[:, j], j]
            counts = np.bincount(col, minlength=Ks[i] + 1)
            if np.any(counts == 0):
                keep[i] = False
                reasons[int(i)] = (
                    f"unused category {int(np.nonzero(counts == 0)[0][0])}"
                )
                changed = True
        if changed:
            continue
        up = csr_matrix((obs & (Xi >= 1)).astype(np.float64))
        down = csr_matrix((obs & (Xi <= Ks[idx][None, :] - 1)).astype(np.float64))
        beats = (up.T @ down).tocsr()
        beats.setdiag(0)
        beats.eliminate_zeros()
        n_comp, labels = connected_components(beats, directed=True, connection="strong")
        if n_comp > 1:
            main = np.bincount(labels).argmax()
            for j, i in enumerate(idx):
                if labels[j] != main:
                    keep[i] = False
                    reasons[int(i)] = "not connected"
            continue
        return keep, reasons


def sample_given_totals(X, thresholds, rng):
    """
    Simulate responses from the Rasch model given each person's total over the items they responded to:
    the conditional distribution P(x | r) = prod_i eps[i, x_i] / gamma_r does not involve the person
    location, so it is sampled exactly, item by item, using the suffix ESFs of the person's items:
    P(x_j = k | items j.. sum to t) = eps[j, k] gamma^(j+1..)_{t-k} / gamma^(j..)_t (log space).
    The missingness pattern and every person's total are those of X. Returns an array like X.
    """
    b_list = _as_threshold_list(thresholds)
    X = np.asarray(X, float)
    Ks = np.array([len(x) for x in b_list])
    Kmax = int(Ks.max())
    LE = np.full((len(b_list), Kmax + 1), NEG)
    for i, x in enumerate(b_list):
        LE[i, : Ks[i] + 1] = np.concatenate([[0.0], -np.cumsum(x)])
    obs = ~np.isnan(X)
    out = np.full(X.shape, np.nan)
    keys, inv = _patterns(obs)
    totals = np.where(obs, X, 0).sum(1).astype(np.int64)
    kk = np.arange(Kmax + 1)
    for g, key in enumerate(keys):
        rows = np.nonzero(inv == g)[0]
        items = np.nonzero(key)[0]
        if not len(items):
            continue
        le = LE[items]  # (n, K1)
        n = len(items)
        suf = [None] * (n + 1)
        suf[n] = np.zeros(1)
        for j in range(n - 1, -1, -1):
            suf[j] = _logconv_small(suf[j + 1], le[j])
        t = totals[rows].copy()
        for j in range(n):
            nxt = suf[j + 1]
            idx = t[:, None] - kk[None, :]  # remaining total after x_j = k
            ok = (idx >= 0) & (idx < len(nxt))
            lp = np.where(ok, le[j][None, :] + nxt[np.clip(idx, 0, len(nxt) - 1)], NEG)
            lp = lp - _lse(lp, axis=1)[:, None]
            with np.errstate(invalid="ignore"):
                p = np.nan_to_num(np.exp(lp))
            x = (p.cumsum(1) > rng.random((len(rows), 1))).argmax(1)
            out[rows, items[j]] = x
            t = t - x
    return out
