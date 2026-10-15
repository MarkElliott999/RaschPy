"""
Regularised composite likelihood CPAT (RCL-CPAT) estimation, shared by RSM and MFRM.

CPAT conditions on response patterns with equal totals: odd totals give item comparisons within a facet element
(and facet element comparisons within an item), even totals give the threshold patterns (k, k) vs (k-1, k+1).
RCL-CPAT fits all of these jointly, as one composite likelihood, with additive smoothing constants on the
pattern counts and ridge penalties on the facet element parameters, by Newton-Raphson.

Every comparison is a binomial in the difference of two cells of the location grid b[r, i, h] (facet element r,
item i, threshold h = 1..K, zero-based here):

    item comparison (r, boundary c, items i, j):  logit P(i higher) = b[r, j, c] - b[r, i, c]
    threshold pattern (r, width k, items i != j): logit P((k, k)) = b[r, j, k+1] - b[r, i, k]  vs (k-1, k+1)
    facet comparison (item i, boundary c, r, q):  logit P(r higher) = b[q, i, c] - b[r, i, c]

For the cell models b = delta_i + tau_h + s[r, i, h] is linear in the parameters (so the composite likelihood is
concave); for the stretch models b = lambda_r + omegaI_r * delta_i + omegaT_r * tau_h is bilinear.
RSM is the case R = 1 with no facet element parameters.

Full pattern set (estimation='cp-full-rcl'): every pair of response patterns (a, b) vs (a-1, b+1) with the same total, on item pairs within
a facet element and facet element pairs within an item, logit = b[., ., b] - b[., ., a-1] (zero-based thresholds).
CPAT's patterns are the d = |b - (a-1)| <= 1 subset. Smoothing constants go on the d <= 1 cells only; the farther
patterns (d >= 2) are left raw -- they never identify a parameter on their own, and their true log-odds are far from
the 0 a pseudo-count pulls towards. See full_comparisons().
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import null_space
from scipy.optimize import minimize

THIN_COUNT = 20  # a category width with fewer CPAT patterns than this is 'thin'
THIN_RIDGE = (
    0.5  # maximum ridge pulling a thin width toward the mean of the other widths
)
DENSE_LIMIT = (
    2.5e7  # largest dense Jacobian (cells x parameters) used for Newton-Raphson
)
SANDWICH_LIMIT = 2e8  # largest dense Jacobian formed for the sandwich after a quasi-Newton fit (1.6 GB)


class SandwichTooLarge(Exception):
    """The design is too large to form the sandwich's dense Jacobian and bread."""


def zero_sum_basis(n):
    """Orthonormal basis (n, n-1) of the zero-sum n-vectors."""
    if n <= 1:
        return np.zeros((n, 0))
    return null_space(np.ones((1, n)))


def threshold_basis(K):
    """tau = T @ D maps the K-1 category widths D to centred thresholds (np.diff(tau) == D)."""
    L = np.tril(np.ones((K, K - 1)), -1)
    return L - L.mean(axis=0)


# ----------------------------------------------------------------------------------------------------------------
# Pattern counts
# ----------------------------------------------------------------------------------------------------------------


def pattern_counts(X, K):
    """
    CPAT pattern counts from responses X (R, P, I), NaN = missing.

    Returns W (R, K, I, I): W[r, c, i, j] = persons scored c+1 on item i and c on item j by facet element r;
    Nn, Dd (R, K-1, I, I): (k, k) and (k-1, k+1) patterns on items (i, j), k = 1..K-1;
    V (I, K, R, R): V[i, c, r, q] = persons scored c+1 by r and c by q on item i.
    """
    R, P, I = X.shape
    # 0/1 indicators; float32 halves their memory and is exact while every count (at most P) is below 2**24
    dt = np.float32 if P < 2**24 else np.float64
    ind = [np.nan_to_num(X == c).astype(dt) for c in range(K + 1)]

    # batched matrix products (BLAS) rather than einsum, which loops over persons in C without BLAS
    def within(A, B):  # sum_p A[r,p,i] B[r,p,j] -> (R, I, I)
        return np.matmul(A.transpose(0, 2, 1), B).astype(np.float64)

    W = np.stack([within(ind[c + 1], ind[c]) for c in range(K)], 1)
    if K > 1:
        Nn = np.stack([within(ind[k], ind[k]) for k in range(1, K)], 1)
        Dd = np.stack([within(ind[k - 1], ind[k + 1]) for k in range(1, K)], 1)
    else:
        Nn = Dd = np.zeros((R, 0, I, I))
    idx = np.arange(I)
    for A in (W, Nn, Dd):
        A[..., idx, idx] = 0
    if R > 1:

        def across(A, B):  # -> (I, R, R)
            return np.matmul(A.transpose(2, 0, 1), B.transpose(2, 1, 0)).astype(
                np.float64
            )

        V = np.stack([across(ind[c + 1], ind[c]) for c in range(K)], 1)
        idx = np.arange(R)
        V[..., idx, idx] = 0
    else:
        V = np.zeros((I, K, 1, 1))
    return W, Nn, Dd, V


def category_counts(X, K):
    """Observed count of each score category 0..K, pooled over everything."""
    v = X[np.isfinite(X)].astype(int)
    return np.bincount(v, minlength=K + 1)[: K + 1]


def empty_widths(cat_counts, K):
    """Width indices (0..K-2) whose patterns involve an empty middle category: an empty category j identifies only
    the intersection of the category response curves of j-1 and j+1, so widths j-2, j-1, j are not identified by
    their own patterns."""
    empty = {j for j in range(1, K) if cat_counts[j] == 0}
    return sorted(b for b in range(K - 1) if empty & {b, b + 1, b + 2})


def width_ridges(Nn, Dd, K, exclude=()):
    """Per-width ridge weights: a width with fewer than THIN_COUNT patterns gets THIN_RIDGE * (1 - T/THIN_COUNT)
    toward the mean of the other widths, a one-sided width (all its patterns on one side) the full THIN_RIDGE.
    Needs at least two widths."""
    rho = np.zeros(K - 1)
    if K - 1 < 2:
        return rho
    E = Nn.sum(axis=(0, 2, 3))
    S = Dd.sum(axis=(0, 2, 3))
    T = (E + S) / 2.0  # ordered item pairs count each pattern twice
    one_sided = (E == 0) | (S == 0)
    rho = np.where(
        one_sided, THIN_RIDGE, THIN_RIDGE * np.clip(1 - T / THIN_COUNT, 0, None)
    )
    rho[list(exclude)] = 0.0
    return rho


def width_penalty_matrix(rho):
    """Q with sum_k rho_k (D_k - mean of the other widths)^2 = D' Q D."""
    n = len(rho)
    Q = np.zeros((n, n))
    for k in np.nonzero(rho)[0]:
        c = np.full(n, -1.0 / (n - 1))
        c[k] = 1.0
        Q += rho[k] * np.outer(c, c)
    return Q


# ----------------------------------------------------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------------------------------------------------


def smooth_counts(W, Nn, Dd, V, c_item, c_thr, c_facet, drop_widths=()):
    """Additive smoothing: c_item on each observed item-pair cell (both directions), c_facet likewise on facet
    element pairs, c_thr on each observed threshold-pattern cell. Patterns of the widths in drop_widths are removed.
    """
    W = W + c_item * ((W + np.swapaxes(W, -1, -2)) > 0)
    V = V + c_facet * ((V + np.swapaxes(V, -1, -2)) > 0)
    seen = (Nn + Dd) > 0
    Nn = Nn + c_thr * seen
    Dd = Dd + c_thr * seen
    if len(drop_widths):
        Nn = Nn.copy()
        Dd = Dd.copy()
        Nn[:, list(drop_widths)] = 0
        Dd[:, list(drop_widths)] = 0
    return W, Nn, Dd, V


def comparisons(W, Nn, Dd, V, R, I, K):
    """Flat comparison list (x, y, t, u): logit = b[x] - b[y], t 'successes', u 'failures', with flat grid index
    (r * I + i) * K + h."""

    def f(r, i, h):
        return (r * I + i) * K + h

    xs, ys, ts, us = [], [], [], []
    r, c, i, j = np.meshgrid(
        np.arange(R), np.arange(K), np.arange(I), np.arange(I), indexing="ij"
    )
    m = i < j
    r, c, i, j = r[m], c[m], i[m], j[m]
    t, u = W[r, c, i, j], W[r, c, j, i]
    k = (t + u) > 0
    xs.append(f(r[k], j[k], c[k]))
    ys.append(f(r[k], i[k], c[k]))
    ts.append(t[k])
    us.append(u[k])
    if K > 1:
        r, k_, i, j = np.meshgrid(
            np.arange(R), np.arange(K - 1), np.arange(I), np.arange(I), indexing="ij"
        )
        m = i != j
        r, k_, i, j = r[m], k_[m], i[m], j[m]
        t, u = Nn[r, k_, i, j], Dd[r, k_, i, j]
        k = (t + u) > 0
        xs.append(f(r[k], j[k], k_[k] + 1))
        ys.append(f(r[k], i[k], k_[k]))
        ts.append(t[k])
        us.append(u[k])
    if R > 1:
        i, c, r, q = np.meshgrid(
            np.arange(I), np.arange(K), np.arange(R), np.arange(R), indexing="ij"
        )
        m = r < q
        i, c, r, q = i[m], c[m], r[m], q[m]
        t, u = V[i, c, r, q], V[i, c, q, r]
        k = (t + u) > 0
        xs.append(f(q[k], i[k], c[k]))
        ys.append(f(r[k], i[k], c[k]))
        ts.append(t[k])
        us.append(u[k])
    x, y = np.concatenate(xs), np.concatenate(ys)
    t, u = np.concatenate(ts).astype(float), np.concatenate(us).astype(float)
    n = len(x)
    D = sparse.csr_matrix(
        (
            np.concatenate([np.ones(n), -np.ones(n)]),
            (np.tile(np.arange(n), 2), np.concatenate([x, y])),
        ),
        shape=(n, R * I * K),
    )
    return D, t, u


def _full_patterns(K, drop_widths=(), empty=()):
    """(a, b, d) for the full pattern set: a = 1..K, b = 0..K-1, d = |b - (a-1)|. The log-odds of pattern (a, b)
    contain tau_{b+1} - tau_a (one-based), the sum of the widths between those thresholds; a pattern is dropped if
    any of them is a split-out width (drop_widths, zero-based), so that no jointly fitted width is tied to a
    split-out one -- for d = 1 this is CPAT's own rule -- or if it involves an empty category (one side would be
    structurally zero)."""
    drop_widths, empty = set(drop_widths), set(empty)
    out = []
    for a in range(1, K + 1):
        for b in range(0, K):
            d = abs(b - (a - 1))
            lo, hi = sorted((a, b + 1))
            if drop_widths & set(range(lo - 1, hi - 1)):
                continue
            if d >= 2 and empty & {a, a - 1, b, b + 1}:
                continue
            out.append((a, b, d))
    return out


def full_comparisons(X, K, c_item, c_thr, c_facet, drop_widths=(), empty=()):
    """Comparison list (D, t, u) for the full pattern set (see module docstring) from responses X (R, P, I):
    logit = b[x] - b[y] with t persons on the (a, b) side and u on the (a-1, b+1) side. Smoothing: c_item (item pairs)
    and c_facet (facet element pairs) on observed d = 0 cells, c_thr on observed d = 1 cells, none farther out.
    """
    R, P, I = X.shape
    dt = np.float32 if P < 2**24 else np.float64
    ind = [np.nan_to_num(X == c).astype(dt) for c in range(K + 1)]

    def f(r, i, h):
        return (r * I + i) * K + h

    iu, ru = np.triu_indices(I, 1), np.triu_indices(R, 1)

    def within(A, B):  # (R, I, I)
        return np.matmul(A.transpose(0, 2, 1), B).astype(np.float64)

    def across(A, B):  # (I, R, R)
        return np.matmul(A.transpose(2, 0, 1), B.transpose(2, 1, 0)).astype(np.float64)

    xs, ys, ts, us = [], [], [], []
    for a, b, d in _full_patterns(K, drop_widths, empty):
        t = within(ind[a], ind[b])[:, iu[0], iu[1]].ravel()
        u = within(ind[a - 1], ind[b + 1])[:, iu[0], iu[1]].ravel()
        r_, i_, j_ = (
            np.repeat(np.arange(R), len(iu[0])),
            np.tile(iu[0], R),
            np.tile(iu[1], R),
        )
        k = (t + u) > 0
        c = c_item if d == 0 else (c_thr if d == 1 else 0.0)
        xs.append(f(r_[k], j_[k], b))
        ys.append(f(r_[k], i_[k], a - 1))
        ts.append(t[k] + c)
        us.append(u[k] + c)
        if R > 1:
            t = across(ind[a], ind[b])[:, ru[0], ru[1]].ravel()
            u = across(ind[a - 1], ind[b + 1])[:, ru[0], ru[1]].ravel()
            i_, r_, q_ = (
                np.repeat(np.arange(I), len(ru[0])),
                np.tile(ru[0], I),
                np.tile(ru[1], I),
            )
            k = (t + u) > 0
            c = c_facet if d == 0 else (c_thr if d == 1 else 0.0)
            xs.append(f(q_[k], i_[k], b))
            ys.append(f(r_[k], i_[k], a - 1))
            ts.append(t[k] + c)
            us.append(u[k] + c)
    x, y = np.concatenate(xs), np.concatenate(ys)
    t, u = np.concatenate(ts).astype(float), np.concatenate(us).astype(float)
    n = len(x)
    D = sparse.csr_matrix(
        (
            np.concatenate([np.ones(n), -np.ones(n)]),
            (np.tile(np.arange(n), 2), np.concatenate([x, y])),
        ),
        shape=(n, R * I * K),
    )
    return D, t, u


def empty_categories(cat_counts):
    """Score categories (0..K) never observed."""
    return [j for j, n in enumerate(cat_counts) if n == 0]


def _loglik(eta, t, u):
    return float((-t * np.logaddexp(0, -eta) - u * np.logaddexp(0, eta)).sum())


# ----------------------------------------------------------------------------------------------------------------
# Cell models (linear in the parameters)
# ----------------------------------------------------------------------------------------------------------------


class CellModel:
    """
    b[r, i, h] = delta_i + tau_h + s[r, i, h], with s built from the model's facet element blocks:
      global: lambda_r;  items: a_ri;  thresholds: t_rh;  bivector: a_ri + t_rh;  matrix: a_ri + t_rh + e_rih
    (a, t centred over facet elements; bivector/matrix t also zero-sum within each facet element; e the
    interaction, with item and threshold main effects removed within each facet element and centred over facet
    elements). Each block is parameterised in an orthonormal basis of its constraint space, so a ridge
    kappa * sum(block^2) is kappa * |coords|^2.
    """

    def __init__(self, model, R, I, K, ridge=0.0, interaction_ridge=0.0):
        self.model, self.R, self.I, self.K = model, R, I, K
        NR, NI, NK = zero_sum_basis(R), zero_sum_basis(I), zero_sum_basis(K)
        T = threshold_basis(K)
        self._NR, self._NI, self._NK, self._T = NR, NI, NK, T
        blocks = [("delta", I - 1, 0.0), ("tau", K - 1, 0.0)]
        if R > 1:
            if model == "global":
                blocks.append(("lambda", R - 1, ridge))
            if model in ("items", "bivector", "matrix"):
                blocks.append(("a", (R - 1) * I, ridge))
            if model == "thresholds":
                blocks.append(("t", (R - 1) * K, ridge))
            if model in ("bivector", "matrix"):
                blocks.append(("t2", (R - 1) * (K - 1), ridge))
            if model == "matrix":
                blocks.append(("e", (R - 1) * (I - 1) * (K - 1), interaction_ridge))
        self.blocks, self.slices, start = blocks, {}, 0
        for name, n, _ in blocks:
            self.slices[name] = slice(start, start + n)
            start += n
        self.p = start
        self.ridge = np.concatenate([np.full(n, pen) for _, n, pen in blocks])

    # forward map, batched: coords (n, B) -> (R, I, K, B)
    def _block_fwd(self, name, C):
        R, I, K = self.R, self.I, self.K
        NR, NI, NK, T = self._NR, self._NI, self._NK, self._T
        B = C.shape[-1]
        if name == "delta":
            return np.broadcast_to((NI @ C)[None, :, None, :], (R, I, K, B))
        if name == "tau":
            return np.broadcast_to((T @ C)[None, None, :, :], (R, I, K, B))
        if name == "lambda":
            return np.broadcast_to((NR @ C)[:, None, None, :], (R, I, K, B))
        if name == "a":
            X = np.einsum("rq,qib->rib", NR, C.reshape(R - 1, I, B), optimize=True)
            return np.broadcast_to(X[:, :, None, :], (R, I, K, B))
        if name == "t":
            X = np.einsum("rq,qhb->rhb", NR, C.reshape(R - 1, K, B), optimize=True)
            return np.broadcast_to(X[:, None, :, :], (R, I, K, B))
        if name == "t2":
            X = np.einsum(
                "rq,hk,qkb->rhb", NR, NK, C.reshape(R - 1, K - 1, B), optimize=True
            )
            return np.broadcast_to(X[:, None, :, :], (R, I, K, B))
        if name == "e":
            return np.einsum(
                "rq,ij,hk,qjkb->rihb",
                NR,
                NI,
                NK,
                C.reshape(R - 1, I - 1, K - 1, B),
                optimize=True,
            )
        raise KeyError(name)

    def _block_adj(self, name, G):
        NR, NI, NK, T = self._NR, self._NI, self._NK, self._T
        if name == "delta":
            return NI.T @ G.sum(axis=(0, 2))
        if name == "tau":
            return T.T @ G.sum(axis=(0, 1))
        if name == "lambda":
            return NR.T @ G.sum(axis=(1, 2))
        if name == "a":
            return (NR.T @ G.sum(axis=2)).ravel()
        if name == "t":
            return (NR.T @ G.sum(axis=1)).ravel()
        if name == "t2":
            return (NR.T @ G.sum(axis=1) @ NK).ravel()
        if name == "e":
            return np.einsum("rq,ij,hk,rih->qjk", NR, NI, NK, G, optimize=True).ravel()
        raise KeyError(name)

    def grid(self, theta):
        b = np.zeros((self.R, self.I, self.K))
        for name, n, _ in self.blocks:
            if n:
                b += self._block_fwd(name, theta[self.slices[name]][:, None])[..., 0]
        return b

    def grid_adjoint(self, G):
        return np.concatenate(
            [
                self._block_adj(name, G) if n else np.zeros(0)
                for name, n, _ in self.blocks
            ]
        )

    def jacobian(self):
        """Dense d b / d theta, (R*I*K, p), rows in flat grid order (r * I + i) * K + h, built from Kronecker
        products of the block bases (equal to applying the forward map to the identity).
        """
        R, I, K = self.R, self.I, self.K
        NR, NI, NK, T = self._NR, self._NI, self._NK, self._T

        def one(n):
            return np.ones((n, 1))

        def kron(*ms):
            return ms[0] if len(ms) == 1 else np.kron(ms[0], kron(*ms[1:]))

        mats = {
            "delta": lambda: kron(one(R), NI, one(K)),
            "tau": lambda: kron(one(R), one(I), T),
            "lambda": lambda: kron(NR, one(I), one(K)),
            "a": lambda: kron(NR, np.eye(I), one(K)),
            "t": lambda: kron(NR, one(I), np.eye(K)),
            "t2": lambda: kron(NR, one(I), NK),
            "e": lambda: kron(NR, NI, NK),
        }
        return np.hstack([mats[name]() for name, n, _ in self.blocks if n])

    def parts(self, theta):
        """delta (I,), tau (K,), s (R, I, K) and the named facet element blocks as arrays."""
        I, K = self.I, self.K

        def z(name, n):
            if name in self.slices:
                return theta[self.slices[name]][:, None]
            return np.zeros((n, 1))

        delta = (self._NI @ z("delta", I - 1))[:, 0]
        tau = (self._T @ z("tau", K - 1))[:, 0]
        s = self.grid(theta) - delta[None, :, None] - tau[None, None, :]
        out = dict(delta=delta, tau=tau, s=s)
        for name, n, _ in self.blocks[2:]:
            out[name] = self._block_fwd(name, theta[self.slices[name]][:, None])[..., 0]
        return out

    def start(self, delta=None, tau=None):
        theta = np.zeros(self.p)
        if delta is not None and self.I > 1:
            theta[self.slices["delta"]] = self._NI.T @ (
                np.asarray(delta, float) - np.mean(delta)
            )
        if tau is not None and self.K > 1:
            theta[self.slices["tau"]] = np.diff(np.asarray(tau, float))
        return theta


def fit_cell(
    model,
    counts,
    R,
    I,
    K,
    c_item,
    c_thr,
    c_facet,
    ridge=0.0,
    interaction_ridge=0.0,
    width_rho=None,
    fixed_widths=None,
    delta0=None,
    tau0=None,
    fixed_coords=None,
    max_iter=100,
    tol=1e-10,
    full=None,
):
    """
    Penalised composite likelihood fit of a cell model. counts = pattern_counts() output. width_rho: per-width
    thin-width ridge; fixed_widths: {width index: value} held fixed (their patterns should already be dropped);
    fixed_coords: {theta index: value} held fixed. full: None for CPAT's patterns, or dict(X=responses (R, P, I),
    empty=empty categories) for the full pattern set (full_comparisons()).
    Returns (CellModel, theta, info).
    """
    fixed_widths = dict(fixed_widths or {})
    if full is not None:
        D, t, u = full_comparisons(
            full["X"],
            K,
            c_item,
            c_thr,
            c_facet,
            drop_widths=list(fixed_widths),
            empty=full["empty"],
        )
    else:
        W, Nn, Dd, V = smooth_counts(
            *counts, c_item, c_thr, c_facet, drop_widths=list(fixed_widths)
        )
        D, t, u = comparisons(W, Nn, Dd, V, R, I, K)
    full_empty = None if full is None else list(full["empty"])
    cm = CellModel(model, R, I, K, ridge=ridge, interaction_ridge=interaction_ridge)
    p = cm.p
    Lam = cm.ridge.copy()
    # thin-width penalty: non-zero only on the K - 1 width coordinates, so kept as that block (Qw) alone
    st = cm.slices["tau"]
    Qw = (
        width_penalty_matrix(np.asarray(width_rho, float))
        if width_rho is not None and np.any(width_rho)
        else np.zeros((st.stop - st.start,) * 2)
    )
    theta = cm.start(delta0, tau0)
    free = np.ones(p, bool)
    for k, v in fixed_widths.items():
        theta[cm.slices["tau"].start + k] = v
        free[cm.slices["tau"].start + k] = False
    for k, v in (fixed_coords or {}).items():
        theta[k] = v
        free[k] = False

    def objective(th, order=1):
        b = cm.grid(th).ravel()
        eta = D @ b
        pen = float(th @ (Lam * th)) + float(th[st] @ Qw @ th[st])
        obj = _loglik(eta, t, u) - pen
        if order == 0:
            return obj, None
        s = 1 / (1 + np.exp(-eta))
        gb = D.T @ (t - (t + u) * s)
        g = cm.grid_adjoint(gb.reshape(R, I, K)) - 2 * Lam * th
        g[st] -= 2 * Qw @ th[st]
        return obj, (g, s)

    info = dict(iterations=0, converged=False, optimiser="newton-raphson")
    if R * I * K * p <= DENSE_LIMIT:
        J = cm.jacobian()
        th = theta
        obj, (g, s) = objective(th)
        dec = np.inf
        for it in range(1, max_iter + 1):
            w = (t + u) * s * (1 - s)
            Hb = (D.T @ sparse.diags(w) @ D).tocsr()
            HbJ = Hb.toarray() @ J if R * I * K <= 6000 else Hb @ J
            negH = J.T @ HbJ + 2 * np.diag(Lam)
            negH[st, st] += 2 * Qw
            nf = negH[np.ix_(free, free)]
            gf = g[free]
            try:
                Lc = np.linalg.cholesky(nf)
                step_f = np.linalg.solve(Lc.T, np.linalg.solve(Lc, gf))
            except np.linalg.LinAlgError:
                step_f = np.linalg.lstsq(nf, gf, rcond=None)[0]
            if not np.all(np.isfinite(step_f)):
                dec = np.inf  # numerical breakdown: stop, not converged
                break
            dec = float(gf @ step_f)
            if dec < tol:
                break
            step = np.zeros(p)
            step[free] = step_f
            a = 1.0
            while a > 1e-10:
                new, _ = objective(th + a * step, order=0)
                if np.isfinite(new) and new >= obj:
                    break
                a /= 2
            th = th + a * step
            obj, (g, s) = objective(th)
        info.update(iterations=it, converged=bool(dec < tol), decrement=dec)
        # the dense Jacobian and bread are not kept (for the matrix sub-model they grow with the square of the
        # number of parameters); the sandwich rebuilds them from these comparisons (cell_state())
        info["_lazy"] = dict(
            D=D,
            t=t,
            u=u,
            Lam=Lam,
            Qw=Qw,
            st=st,
            free=free,
            drop_widths=sorted(fixed_widths),
            full_empty=full_empty,
        )
        return cm, th, info

    # large problems: same (concave) objective, limited-memory quasi-Newton on the free coordinates
    fixed_vals = theta[~free]

    def f(v):
        th = theta.copy()
        th[free] = v
        th[~free] = fixed_vals
        obj, (g, _) = objective(th)
        return -obj, -g[free]

    # what the sandwich needs to form the bread later, on demand (cell_state())
    info["_lazy"] = dict(
        D=D,
        t=t,
        u=u,
        Lam=Lam,
        Qw=Qw,
        st=st,
        free=free,
        drop_widths=sorted(fixed_widths),
        full_empty=full_empty,
    )
    res = minimize(
        f,
        theta[free],
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=20000, maxfun=50000, ftol=1e-15, gtol=1e-7),
    )
    th = theta.copy()
    th[free] = res.x
    info.update(
        iterations=int(res.nit), converged=bool(res.success), optimiser="l-bfgs-b"
    )
    return cm, th, info


# ----------------------------------------------------------------------------------------------------------------
# Stretch models (bilinear in the parameters)
# ----------------------------------------------------------------------------------------------------------------


class StretchModel:
    """
    b[r, i, h] = lambda_r + omegaI_r * delta_i + omegaT_r * tau_h (centrality: omegaI = 1; pseudo_halo:
    omegaT = 1; bistretch: both free). Natural parameters phi = [delta, tau, lambda, omegaI, omegaT]; free
    parameters theta with delta = N_I a, tau = T D, lambda = N_R l, omegaI = 1 + N_R c, omegaT = 1 + N_R e,
    so phi = phi0 + M theta and the mean omega is 1. Ridge kappa * sum(omega - 1)^2 on the free omega axes.
    """

    def __init__(
        self,
        model,
        counts,
        R,
        I,
        K,
        c_item,
        c_thr,
        c_facet,
        kappa,
        width_rho=None,
        drop_widths=(),
        full=None,
    ):
        # full: None for CPAT's patterns, or dict(X=responses (R, P, I), empty=empty categories) for the full
        # pattern set, evaluated through the location grid (_evaluate_full())
        self._full = (
            None
            if full is None
            else full_comparisons(
                full["X"],
                K,
                c_item,
                c_thr,
                c_facet,
                drop_widths=drop_widths,
                empty=full["empty"],
            )
        )
        self.full_empty = None if full is None else list(full["empty"])
        W, Nn, Dd, V = smooth_counts(
            *counts, c_item, c_thr, c_facet, drop_widths=drop_widths
        )
        self.model, self.R, self.I, self.K, self.kappa = model, R, I, K, kappa
        self.useI = model in ("pseudo_halo", "bistretch")
        self.useT = model in ("centrality", "bistretch")
        oD, oT, oL, oI, oW = 0, I, I + K, I + K + R, I + K + 2 * R
        self.off, self.p_nat = (oD, oT, oL, oI, oW), I + K + 3 * R
        r, c, i, j = np.meshgrid(
            np.arange(R), np.arange(K), np.arange(I), np.arange(I), indexing="ij"
        )
        m = i < j
        r, c, i, j = r[m], c[m], i[m], j[m]
        t, u = W[r, c, i, j], W[r, c, j, i]
        k = (t + u) > 0
        self.A = dict(r=r[k], c=c[k], i=i[k], j=j[k], t=t[k], u=u[k])
        r, k_, i, j = np.meshgrid(
            np.arange(R),
            np.arange(max(K - 1, 0)),
            np.arange(I),
            np.arange(I),
            indexing="ij",
        )
        m = i != j
        r, k_, i, j = r[m], k_[m], i[m], j[m]
        t, u = Nn[r, k_, i, j], Dd[r, k_, i, j]
        k = (t + u) > 0
        self.B = dict(r=r[k], k=k_[k], i=i[k], j=j[k], t=t[k], u=u[k])
        i, c, r, q = np.meshgrid(
            np.arange(I), np.arange(K), np.arange(R), np.arange(R), indexing="ij"
        )
        m = r < q
        i, c, r, q = i[m], c[m], r[m], q[m]
        t, u = V[i, c, r, q], V[i, c, q, r]
        k = (t + u) > 0
        self.C = dict(i=i[k], c=c[k], r=r[k], q=q[k], t=t[k], u=u[k])
        NI, NR, T = zero_sum_basis(I), zero_sum_basis(R), threshold_basis(K)
        blocks = (
            [(oD, NI), (oT, T), (oL, NR)]
            + ([(oI, NR)] if self.useI else [])
            + ([(oW, NR)] if self.useT else [])
        )
        self.M = np.zeros((self.p_nat, sum(b.shape[1] for _, b in blocks)))
        col = 0
        self.tau_cols = None
        for off, b in blocks:
            self.M[off : off + b.shape[0], col : col + b.shape[1]] = b
            if off == oT:
                self.tau_cols = slice(col, col + b.shape[1])
            col += b.shape[1]
        self.p = self.M.shape[1]
        self.phi0 = np.zeros(self.p_nat)
        self.phi0[oI : oI + R] = 1.0
        self.phi0[oW : oW + R] = 1.0
        self.Q = np.zeros((self.p, self.p))
        if width_rho is not None and np.any(width_rho):
            self.Q[self.tau_cols, self.tau_cols] = width_penalty_matrix(
                np.asarray(width_rho, float)
            )

    def theta_from(self, delta, tau, lam, omegaI, omegaT):
        R = self.R
        oI, oW = self.off[3], self.off[4]
        phi = np.concatenate(
            [
                delta - np.mean(delta),
                tau - np.mean(tau),
                lam - np.mean(lam),
                omegaI - np.mean(omegaI) + 1,
                omegaT - np.mean(omegaT) + 1,
            ]
        )
        if not self.useI:
            phi[oI : oI + R] = 1
        if not self.useT:
            phi[oW : oW + R] = 1
        return np.linalg.lstsq(self.M, phi - self.phi0, rcond=None)[0]

    def unpack(self, theta):
        phi = self.phi0 + self.M @ theta
        oD, oT, oL, oI, oW = self.off
        I, K, R = self.I, self.K, self.R
        return (
            phi[oD : oD + I],
            phi[oT : oT + K],
            phi[oL : oL + R],
            phi[oI : oI + R],
            phi[oW : oW + R],
        )

    def _evaluate_full(self, theta, order=2):
        """evaluate() for the full pattern set, through the location grid b = lambda + omegaI delta + omegaT tau."""
        D, t, u = self._full
        R, I, K = self.R, self.I, self.K
        d, tau, lam, wI, wT = self.unpack(theta)
        oD, oT, oL, oI, oW = self.off
        p = self.p_nat
        b = (
            lam[:, None, None]
            + wI[:, None, None] * d[None, :, None]
            + wT[:, None, None] * tau[None, None, :]
        )
        eta = D @ b.ravel()
        ll = _loglik(eta, t, u)
        s = 1 / (1 + np.exp(-eta))
        G = (D.T @ (t - (t + u) * s)).reshape(R, I, K)
        g = np.zeros(p)
        g[oD : oD + I] = np.einsum("r,rih->i", wI, G)
        g[oT : oT + K] = np.einsum("r,rih->h", wT, G)
        g[oL : oL + R] = G.sum(axis=(1, 2))
        if self.useI:
            g[oI : oI + R] = np.einsum("i,rih->r", d, G)
        if self.useT:
            g[oW : oW + R] = np.einsum("h,rih->r", tau, G)
        if order == 2:
            # via the sparse cell-by-cell information D' W D, as in fit_cell(), not the dense comparisons-by-
            # parameters matrix D J (far larger when facet elements are many)
            Jb = stretch_grid_jacobian(self, theta)
            Hb = (D.T @ sparse.diags((t + u) * s * (1 - s)) @ D).tocsr()
            HF = -(Jb.T @ (Hb @ Jb))
            H2 = np.zeros((p, p))
            if self.useI:
                H2[oI : oI + R, oD : oD + I] += G.sum(axis=2)
            if self.useT:
                H2[oW : oW + R, oT : oT + K] += G.sum(axis=1)
            H2 = H2 + H2.T
        pen = 0.0
        for use, off, w_ in ((self.useI, oI, wI), (self.useT, oW, wT)):
            if use:
                pen += self.kappa * ((w_ - 1) ** 2).sum()
                g[off : off + R] -= 2 * self.kappa * (w_ - 1)
                if order == 2:
                    ii = np.arange(off, off + R)
                    HF[ii, ii] -= 2 * self.kappa
        M = self.M
        out = dict(
            obj=ll - pen - float(theta @ self.Q @ theta), g=M.T @ g - 2 * self.Q @ theta
        )
        if order == 2:
            out.update(H=M.T @ (HF + H2) @ M - 2 * self.Q, HF=M.T @ HF @ M - 2 * self.Q)
        return out

    def evaluate(self, theta, order=2):
        """Penalised composite log-likelihood, gradient and (full and Fisher) Hessian with respect to theta."""
        if self._full is not None:
            return self._evaluate_full(theta, order)
        d, tau, lam, wI, wT = self.unpack(theta)
        oD, oT, oL, oI, oW = self.off
        p = self.p_nat
        A, B, C = self.A, self.B, self.C
        terms = []
        eta = wI[A["r"]] * (d[A["j"]] - d[A["i"]])
        cols = np.stack([oD + A["j"], oD + A["i"], oI + A["r"]], 1)
        vals = np.stack([wI[A["r"]], -wI[A["r"]], d[A["j"]] - d[A["i"]]], 1)
        sec = [(oI + A["r"], oD + A["j"], 1.0), (oI + A["r"], oD + A["i"], -1.0)]
        terms.append((eta, A["t"], A["u"], cols, vals, sec))
        if len(B["k"]):
            Dk = tau[B["k"] + 1] - tau[B["k"]]
            eta = wI[B["r"]] * (d[B["j"]] - d[B["i"]]) + wT[B["r"]] * Dk
            cols = np.stack(
                [
                    oD + B["j"],
                    oD + B["i"],
                    oT + B["k"] + 1,
                    oT + B["k"],
                    oI + B["r"],
                    oW + B["r"],
                ],
                1,
            )
            vals = np.stack(
                [
                    wI[B["r"]],
                    -wI[B["r"]],
                    wT[B["r"]],
                    -wT[B["r"]],
                    d[B["j"]] - d[B["i"]],
                    Dk,
                ],
                1,
            )
            sec = [
                (oI + B["r"], oD + B["j"], 1.0),
                (oI + B["r"], oD + B["i"], -1.0),
                (oW + B["r"], oT + B["k"] + 1, 1.0),
                (oW + B["r"], oT + B["k"], -1.0),
            ]
            terms.append((eta, B["t"], B["u"], cols, vals, sec))
        if len(C["i"]):
            tc, di = tau[C["c"]], d[C["i"]]
            eta = (
                (lam[C["q"]] - lam[C["r"]])
                + (wI[C["q"]] - wI[C["r"]]) * di
                + (wT[C["q"]] - wT[C["r"]]) * tc
            )
            one = np.ones_like(eta)
            cols = np.stack(
                [
                    oL + C["q"],
                    oL + C["r"],
                    oI + C["q"],
                    oI + C["r"],
                    oD + C["i"],
                    oW + C["q"],
                    oW + C["r"],
                    oT + C["c"],
                ],
                1,
            )
            vals = np.stack(
                [
                    one,
                    -one,
                    di,
                    -di,
                    wI[C["q"]] - wI[C["r"]],
                    tc,
                    -tc,
                    wT[C["q"]] - wT[C["r"]],
                ],
                1,
            )
            sec = [
                (oI + C["q"], oD + C["i"], 1.0),
                (oI + C["r"], oD + C["i"], -1.0),
                (oW + C["q"], oT + C["c"], 1.0),
                (oW + C["r"], oT + C["c"], -1.0),
            ]
            terms.append((eta, C["t"], C["u"], cols, vals, sec))
        ll, g = 0.0, np.zeros(p)
        HF = np.zeros((p, p))
        H2 = np.zeros((p, p))
        for eta, t, u, cols, vals, sec in terms:
            ll += _loglik(eta, t, u)
            s = 1 / (1 + np.exp(-eta))
            e = t - (t + u) * s
            w = (t + u) * s * (1 - s)
            np.add.at(g, cols.ravel(), (vals * e[:, None]).ravel())
            if order == 2:
                n_, k_ = cols.shape
                Jm = sparse.csr_matrix(
                    (vals.ravel(), (np.repeat(np.arange(n_), k_), cols.ravel())),
                    shape=(n_, p),
                )
                HF -= (Jm.T @ sparse.diags(w) @ Jm).toarray()
                for a_, b_, sgn in sec:
                    np.add.at(H2, (a_, b_), sgn * e)
                    np.add.at(H2, (b_, a_), sgn * e)
        pen = 0.0
        for use, off, w_ in ((self.useI, oI, wI), (self.useT, oW, wT)):
            if use:
                pen += self.kappa * ((w_ - 1) ** 2).sum()
                g[off : off + self.R] -= 2 * self.kappa * (w_ - 1)
                if order == 2:
                    ii = np.arange(off, off + self.R)
                    HF[ii, ii] -= 2 * self.kappa
        M = self.M
        out = dict(
            obj=ll - pen - float(theta @ self.Q @ theta), g=M.T @ g - 2 * self.Q @ theta
        )
        if order == 2:
            out.update(H=M.T @ (HF + H2) @ M - 2 * self.Q, HF=M.T @ HF @ M - 2 * self.Q)
        return out

    def fit(self, theta0, fixed=None, max_iter=100, tol=1e-10):
        """Newton-Raphson (Fisher scoring when the Hessian is not negative definite), backtracking line search on
        the penalised objective, stopping on the Newton decrement. fixed: {theta index: value}.
        """
        free = np.ones(self.p, bool)
        th = theta0.copy()
        for k, v in (fixed or {}).items():
            th[k] = v
            free[k] = False
        cur = self.evaluate(th)
        dec = np.inf
        fisher = 0
        for it in range(1, max_iter + 1):
            g = cur["g"][free]
            try:
                negH = -cur["H"][np.ix_(free, free)]
                np.linalg.cholesky(negH)
                sf = _determined_step(negH, g)
            except np.linalg.LinAlgError:
                sf = np.linalg.lstsq(-cur["HF"][np.ix_(free, free)], g, rcond=None)[0]
                fisher += 1
            if not np.all(np.isfinite(sf)):
                dec = np.inf
                break
            dec = float(g @ sf)
            if dec < tol:
                break
            step = np.zeros(self.p)
            step[free] = sf
            a = 1.0
            while a > 1e-10:
                new = self.evaluate(th + a * step, order=1)
                if np.isfinite(new["obj"]) and new["obj"] >= cur["obj"]:
                    break
                a /= 2
            th = th + a * step
            cur = self.evaluate(th)
        return th, dict(
            iterations=it,
            converged=bool(dec < tol),
            decrement=dec,
            fisher_steps=fisher,
            objective=cur["obj"],
            optimiser="newton-raphson",
        )


# ----------------------------------------------------------------------------------------------------------------
# Sandwich covariance for the cell models
# ----------------------------------------------------------------------------------------------------------------


def _sig(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def person_grid_scores(b, X, K, drop_widths=()):
    """Each person's score with respect to the location grid b (R, I, K): the derivative of their own pattern
    contributions to the composite log-likelihood (smoothing pseudo-counts belong to no one). X (R, P, I).
    Returns (P, R, I, K)."""
    R, P, I = X.shape
    ind = [np.nan_to_num(X == c).astype(np.float64) for c in range(K + 1)]
    G = np.zeros((P, R, I, K))
    off = 1.0 - np.eye(I)
    for r in range(R):
        for c in range(K):
            H, L = ind[c + 1][r], ind[c][r]
            Qm = _sig(b[r, :, c][:, None] - b[r, :, c][None, :]) * off
            G[:, r, :, c] += L * (H @ Qm) - H * (L @ Qm.T)
        for kk in range(K - 1):
            if kk in drop_widths:
                continue
            k = kk + 1
            M, A, Bp = ind[k][r], ind[k - 1][r], ind[k + 1][r]
            eta = (
                b[r, :, kk + 1][None, :] - b[r, :, kk][:, None]
            )  # eta[i, j] = b[r, j, kk+1] - b[r, i, kk]
            S0 = _sig(eta) * off
            S1 = (1 - _sig(eta)) * off
            G[:, r, :, kk + 1] += M * (M @ S1) - Bp * (A @ S0)
            G[:, r, :, kk] -= M * (M @ S1.T) - A * (Bp @ S0.T)
    if R > 1:
        offR = 1.0 - np.eye(R)
        for i in range(I):
            for c in range(K):
                H, L = ind[c + 1][:, :, i].T, ind[c][:, :, i].T  # (P, R)
                Qm = _sig(b[:, i, c][:, None] - b[:, i, c][None, :]) * offR
                G[:, :, i, c] += L * (H @ Qm) - H * (L @ Qm.T)
    return G


def person_grid_bread(b, X, K, sl, drop_widths=()):
    """Each person's contribution to the (unpenalised) composite information with respect to the location grid,
    for persons X[:, sl] of a single facet element (R = 1). Returns (n, I*K, I*K)."""
    R, P, I = X.shape
    assert R == 1
    ind = [np.nan_to_num(X[0, sl] == c).astype(np.float64) for c in range(K + 1)]
    n = ind[0].shape[0]
    Hb = np.zeros((n, I, K, I, K))
    off = 1.0 - np.eye(I)
    ii = np.arange(I)

    def add(e, w, hx, hy):
        # comparisons logit = b[j, hx] - b[i, hy] with person weights e[n, i, j] and curvature w[i, j]
        v = e * w[None]
        Hb[:, ii, hx, ii, hx] += v.sum(axis=1)  # (j, hx) diagonal: sum over i
        Hb[:, ii, hy, ii, hy] += v.sum(axis=2)  # (i, hy) diagonal: sum over j
        Hb[:, :, hx, :, hy] -= v.transpose(0, 2, 1)  # [(j, hx), (i, hy)]
        Hb[:, :, hy, :, hx] -= v  # [(i, hy), (j, hx)]

    for c in range(K):
        H, L = ind[c + 1], ind[c]
        e = H[:, :, None] * L[:, None, :]  # i higher, j lower
        e = (e + e.transpose(0, 2, 1)) * off
        s = _sig(b[0, :, c][None, :] - b[0, :, c][:, None])
        w = s * (1 - s)
        iu = np.triu(np.ones((I, I)), 1)
        add(e * iu, w, c, c)
    for kk in range(K - 1):
        if kk in drop_widths:
            continue
        k = kk + 1
        M, A, Bp = ind[k], ind[k - 1], ind[k + 1]
        e = (M[:, :, None] * M[:, None, :] + A[:, :, None] * Bp[:, None, :]) * off
        s = _sig(b[0, :, kk + 1][None, :] - b[0, :, kk][:, None])
        add(e, s * (1 - s), kk + 1, kk)
    return Hb.reshape(n, I * K, I * K)


def full_person_grid_scores(b, X, K, drop_widths=(), empty=()):
    """person_grid_scores() for the full pattern set (full_comparisons()): each person's score with respect to the
    location grid b (R, I, K) from their own patterns. X (R, P, I). Returns (P, R, I, K).
    """
    R, P, I = X.shape
    ind = [np.nan_to_num(X == c).astype(np.float64) for c in range(K + 1)]
    G = np.zeros((P, R, I, K))
    upI, upR = np.triu(np.ones((I, I)), 1), np.triu(np.ones((R, R)), 1)
    for a, bb, d in _full_patterns(K, drop_widths, empty):
        for r in range(R):
            # logit[i, j] = b[r, j, bb] - b[r, i, a-1], items i < j; score e = 1(a, bb)(1 - s) - 1(a-1, bb+1) s
            s = _sig(b[r, :, bb][None, :] - b[r, :, a - 1][:, None])
            Cm, Sm = (1 - s) * upI, s * upI
            A1, B1, A0, B0 = ind[a][r], ind[bb][r], ind[a - 1][r], ind[bb + 1][r]
            G[:, r, :, bb] += B1 * (A1 @ Cm) - B0 * (A0 @ Sm)
            G[:, r, :, a - 1] -= A1 * (B1 @ Cm.T) - A0 * (B0 @ Sm.T)
        if R > 1:
            for i in range(I):
                # logit[r, q] = b[q, i, bb] - b[r, i, a-1], facet elements r < q
                s = _sig(b[:, i, bb][None, :] - b[:, i, a - 1][:, None])
                Cm, Sm = (1 - s) * upR, s * upR
                A1, B1 = ind[a][:, :, i].T, ind[bb][:, :, i].T
                A0, B0 = ind[a - 1][:, :, i].T, ind[bb + 1][:, :, i].T
                G[:, :, i, bb] += B1 * (A1 @ Cm) - B0 * (A0 @ Sm)
                G[:, :, i, a - 1] -= A1 * (B1 @ Cm.T) - A0 * (B0 @ Sm.T)
    return G


def full_person_grid_bread(b, X, K, sl, drop_widths=(), empty=()):
    """person_grid_bread() for the full pattern set (single facet element, persons X[:, sl]). Returns
    (n, I*K, I*K)."""
    R, P, I = X.shape
    assert R == 1
    ind = [np.nan_to_num(X[0, sl] == c).astype(np.float64) for c in range(K + 1)]
    n = ind[0].shape[0]
    Hb = np.zeros((n, I, K, I, K))
    ii = np.arange(I)
    up = np.triu(np.ones((I, I)), 1)
    for a, bb, d in _full_patterns(K, drop_widths, empty):
        e = (
            ind[a][:, :, None] * ind[bb][:, None, :]
            + ind[a - 1][:, :, None] * ind[bb + 1][:, None, :]
        ) * up
        s = _sig(b[0, :, bb][None, :] - b[0, :, a - 1][:, None])
        v = e * (s * (1 - s))[None]
        hx, hy = bb, a - 1
        Hb[:, ii, hx, ii, hx] += v.sum(axis=1)
        Hb[:, ii, hy, ii, hy] += v.sum(axis=2)
        Hb[:, :, hx, :, hy] -= v.transpose(0, 2, 1)
        Hb[:, :, hy, :, hx] -= v
    return Hb.reshape(n, I * K, I * K)


def warn_unidentified(stacklevel=3):
    """Warn that some estimates are not identified, so their standard errors are NaN."""
    warnings.warn(
        "The data do not identify some of the estimates -- for example the locations of items "
        "not linked to the others, a category width with no threshold patterns, or the location "
        "of a facet element (e.g. rater) never compared with the others -- so their standard "
        "errors are set to NaN.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _determined_step(A, g, rel_tol=1e-9):
    """Newton step A^-1 g for symmetric positive (semi-)definite A, with no step along the directions A does not
    determine (eigenvalues <= rel_tol x the largest, as in null_directions()). Such a direction is flat in both the
    data and the penalties -- e.g. the shift between groups of facet elements never compared with each other -- so a
    plain solve returns an arbitrary step there, which can be enormous; the estimate is left where it started.
    """
    w, Q = np.linalg.eigh((A + A.T) / 2)
    keep = w > rel_tol * max(w.max(), 1e-300)
    return Q[:, keep] @ ((Q[:, keep].T @ g) / w[keep])


def null_directions(B, rel_tol=1e-9):
    """Orthonormal basis (p, k) of the directions in which the symmetric bread B is (numerically) singular: parameter
    combinations the data and penalties do not identify."""
    w, Q = np.linalg.eigh((B + B.T) / 2)
    return Q[:, w <= rel_tol * max(w.max(), 1e-300)]


def facet_components(V_counts):
    """Connected components of the facet element comparison graph (facet elements r, q linked if any person was
    scored one point apart by them on the same item), from raw V counts (I, K, R, R). Returns a list of index lists.
    """
    R = V_counts.shape[-1]
    A = (V_counts.sum(axis=(0, 1)) + V_counts.sum(axis=(0, 1)).T) > 0
    seen, comps = np.zeros(R, bool), []
    for r in range(R):
        if seen[r]:
            continue
        stack, comp = [r], []
        seen[r] = True
        while stack:
            a = stack.pop()
            comp.append(a)
            for q in np.nonzero(A[a] & ~seen)[0]:
                seen[q] = True
                stack.append(q)
        comps.append(sorted(comp))
    return comps


def cell_sandwich(cm, theta, info, X, K, small_sample=False, kc_scores=None):
    """Sandwich covariance of the free cell-model parameters theta: bread = negative penalised Hessian at the
    estimate, meat = sum over persons of their score outer products. Returns (V, free, null) with V over the free
    coordinates and null a basis of the directions the bread does not identify (see null_directions()).
    small_sample=True applies the Kauermann-Carroll correction (single facet element only).
    """
    st = cell_state(cm, theta, info)
    J, negH, free, drop = st["J"], st["negH"], st["free"], st["drop_widths"]
    empty = st.get("full_empty")
    b = cm.grid(theta)
    G = (
        person_grid_scores(b, X, K, drop)
        if empty is None
        else full_person_grid_scores(b, X, K, drop, empty)
    )
    U = G.reshape(G.shape[0], -1) @ J[:, free]
    Bf = negH[np.ix_(free, free)]
    A = np.linalg.pinv(Bf)
    if small_sample:
        Jf = J[:, free]
        bread = (
            (lambda sl: person_grid_bread(b, X, K, sl, drop))
            if empty is None
            else (lambda sl: full_person_grid_bread(b, X, K, sl, drop, empty))
        )
        U = kc_scores(U, lambda sl: np.einsum("ap,nab,bq->npq", Jf, bread(sl), Jf), A)
    return A @ (U.T @ U) @ A.T, free, null_directions(Bf)


def cell_state(cm, theta, info):
    """The dense Jacobian and bread of a cell-model fit at its estimate, formed from the fit's stored comparisons
    and cached in info until release_state(), unless the Jacobian would exceed SANDWICH_LIMIT entries, which raises
    SandwichTooLarge."""
    if "_state" not in info:
        lz = info["_lazy"]
        n_cells = cm.R * cm.I * cm.K
        if n_cells * cm.p > SANDWICH_LIMIT:
            raise SandwichTooLarge(f"{n_cells:,} cells x {cm.p:,} parameters")
        J = cm.jacobian()
        D, t, u = lz["D"], lz["t"], lz["u"]
        s = 1 / (1 + np.exp(-(D @ cm.grid(theta).ravel())))
        Hb = (D.T @ sparse.diags((t + u) * s * (1 - s)) @ D).tocsr()
        negH = J.T @ (Hb @ J) + 2 * np.diag(lz["Lam"])
        negH[lz["st"], lz["st"]] += 2 * lz["Qw"]
        info["_state"] = dict(
            J=J,
            negH=negH,
            free=lz["free"],
            drop_widths=lz["drop_widths"],
            full_empty=lz.get("full_empty"),
        )
    return info["_state"]


def release_state(info):
    """Drop the cached Jacobian and bread of cell_state() once the standard errors no longer need them."""
    info.pop("_state", None)


def cell_null(cm, theta, info):
    """Directions the bread of a cell-model fit does not identify (see null_directions()), from the bread at the
    estimate -- without forming the persons' scores, which only the covariance itself needs.
    """
    st = cell_state(cm, theta, info)
    return null_directions(st["negH"][np.ix_(st["free"], st["free"])])


def stretch_null(sm, theta, fixed=()):
    """As cell_null() for a stretch-model fit (fixed: theta indices held fixed)."""
    free = np.ones(sm.p, bool)
    free[list(fixed)] = False
    return null_directions(-sm.evaluate(theta)["H"][np.ix_(free, free)])


def stretch_grid_jacobian(sm, theta):
    """d b / d phi at theta for a StretchModel, (R*I*K, p_nat), phi = [delta, tau, lambda, omegaI, omegaT]."""
    d, tau, lam, wI, wT = sm.unpack(theta)
    R, I, K = sm.R, sm.I, sm.K
    oD, oT, oL, oI, oW = sm.off
    Jn = np.zeros((R, I, K, sm.p_nat))
    r, i, h = np.meshgrid(np.arange(R), np.arange(I), np.arange(K), indexing="ij")
    Jn[r, i, h, oD + i] = wI[r]
    Jn[r, i, h, oT + h] = wT[r]
    Jn[r, i, h, oL + r] = 1.0
    if sm.useI:
        Jn[r, i, h, oI + r] = d[i]
    if sm.useT:
        Jn[r, i, h, oW + r] = tau[h]
    return Jn.reshape(R * I * K, sm.p_nat)


def stretch_sandwich(sm, theta, X, K, fixed=(), drop_widths=()):
    """Sandwich covariance of the free stretch-model parameters theta (fixed: theta indices held fixed). Bread =
    negative penalised Hessian at the estimate, meat = sum over persons of their score outer products.
    Returns (V, free, null) with V over the free coordinates and null as in cell_sandwich().
    """
    free = np.ones(sm.p, bool)
    free[list(fixed)] = False
    d, tau, lam, wI, wT = sm.unpack(theta)
    b = (
        lam[:, None, None]
        + wI[:, None, None] * d[None, :, None]
        + wT[:, None, None] * tau[None, None, :]
    )
    G = (
        person_grid_scores(b, X, K, drop_widths)
        if sm.full_empty is None
        else full_person_grid_scores(b, X, K, drop_widths, sm.full_empty)
    )
    J = stretch_grid_jacobian(sm, theta) @ sm.M[:, free]
    U = G.reshape(G.shape[0], -1) @ J
    Bf = -sm.evaluate(theta)["H"][np.ix_(free, free)]
    A = np.linalg.pinv(Bf)
    return A @ (U.T @ U) @ A.T, free, null_directions(Bf)
