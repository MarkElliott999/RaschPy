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
    their patterns."""
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
    split-out one -- for d = 1 this is CPAT's rule -- or if it involves an empty category (one side would be
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

    levels: {'global', 'item', 'threshold', 'interaction': weight} for the penalty (weight / 2) x sum of squares of,
    respectively, each facet element's global effect (its mean effect over the I x K cells), its item effects minus its
    global effect and its threshold effects minus its global effect (both summed over the I x K cells), and the
    interactions.
    With levels, ridge and interaction_ridge (v1.4's whole-block penalty, kappa * |coords|^2) should be 0.
    """

    def __init__(self, model, R, I, K, ridge=0.0, interaction_ridge=0.0, levels=None):
        self.model, self.R, self.I, self.K = model, R, I, K
        self.levels = {k: float(v) for k, v in (levels or {}).items() if v}
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

    # blocks of the facet element parameters, by penalty level
    _ITEM_BLOCKS, _THRESHOLD_BLOCKS = ("a",), ("t", "t2")

    def _level_terms(self, grids):
        """Penalty-level projections of the facet element block grids (each (R, I, K, B)): global effect (R, B) and the
        item, threshold and interaction grids, each minus its facet element mean where that applies."""
        R = self.R
        S = sum(grids.values())
        sev = S.reshape(R, -1, S.shape[-1]).mean(axis=1)

        def dev(names):
            gs = [grids[n] for n in names if n in grids]
            if not gs:
                return None
            G = sum(gs)
            return G - G.mean(axis=(1, 2), keepdims=True)

        e = grids.get("e")
        return sev, dev(self._ITEM_BLOCKS), dev(self._THRESHOLD_BLOCKS), e

    def level_penalty(self, theta):
        """(value, gradient) of the levels penalty at theta (0, 0 without levels)."""
        g = np.zeros(self.p)
        if not self.levels:
            return 0.0, g
        grids = {
            name: self._block_fwd(name, theta[self.slices[name]][:, None])
            for name, n, _ in self.blocks[2:]
            if n
        }
        if not grids:  # a single facet element: no facet element parameters
            return 0.0, g
        sev, item, thr, e = self._level_terms(grids)
        lv = self.levels
        val = 0.5 * lv.get("global", 0.0) * float((sev**2).sum())
        n_cells = self.I * self.K
        G_sev = np.broadcast_to(
            (lv.get("global", 0.0) * sev / n_cells)[:, None, None, :],
            (self.R, self.I, self.K, 1),
        )
        parts = {}
        for name in grids:
            G = G_sev.copy()
            if item is not None and name in self._ITEM_BLOCKS:
                G = G + lv.get("item", 0.0) * item
            if thr is not None and name in self._THRESHOLD_BLOCKS:
                G = G + lv.get("threshold", 0.0) * thr
            if name == "e":
                G = G + lv.get("interaction", 0.0) * e
            parts[name] = G[..., 0]
        if item is not None:
            val += 0.5 * lv.get("item", 0.0) * float((item**2).sum())
        if thr is not None:
            val += 0.5 * lv.get("threshold", 0.0) * float((thr**2).sum())
        if e is not None:
            val += 0.5 * lv.get("interaction", 0.0) * float((e**2).sum())
        for name, G in parts.items():
            g[self.slices[name]] = self._block_adj(name, G)
        return val, g

    def level_matrix(self):
        """Q (p, p) with levels penalty 0.5 * theta' Q theta (zeros without levels)."""
        Q = np.zeros((self.p, self.p))
        names = [name for name, n, _ in self.blocks[2:] if n]
        if not self.levels or not names:
            return Q
        grids = {
            name: self._block_fwd(name, np.eye(self.slices[name].stop - self.slices[name].start))
            for name in names
        }
        lv = self.levels
        R, I, K = self.R, self.I, self.K
        sl = [self.slices[n] for n in names]
        if lv.get("global"):
            L = np.zeros((R, self.p))
            for name, s in zip(names, sl):
                L[:, s] = grids[name].reshape(R, -1, grids[name].shape[-1]).mean(axis=1)
            Q += lv["global"] * L.T @ L
        for level, blocks in (
            ("item", self._ITEM_BLOCKS),
            ("threshold", self._THRESHOLD_BLOCKS),
            ("interaction", ("e",)),
        ):
            if not lv.get(level):
                continue
            P = np.zeros((R * I * K, self.p))
            for name, s in zip(names, sl):
                if name in blocks:
                    G = grids[name]
                    if level != "interaction":
                        G = G - G.mean(axis=(1, 2), keepdims=True)
                    P[:, s] = G.reshape(R * I * K, -1)
            Q += lv[level] * P.T @ P
        return Q

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


def _width_centring(nw, fixed=()):
    """(nw, nw) centring matrix over the category widths not in fixed (zero rows / columns for those)."""
    keep = [k for k in range(nw) if k not in set(fixed)]
    C = np.zeros((nw, nw))
    if len(keep) > 1:
        C[np.ix_(keep, keep)] = np.eye(len(keep)) - 1.0 / len(keep)
    return C


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
    levels=None,
    item_ridge=0.0,
    width_ridge=0.0,
):
    """
    Penalised composite likelihood fit of a cell model. item_ridge: (item_ridge / 2) x sum of squared centred item
    locations; width_ridge: (width_ridge / 2) x sum over the fitted category widths of (width - their mean)^2.
    counts = pattern_counts() output. levels: the three-level
    penalty weights (see CellModel); ridge, interaction_ridge: v1.4's whole-block penalty. width_rho: per-width
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
    cm = CellModel(
        model, R, I, K, ridge=ridge, interaction_ridge=interaction_ridge, levels=levels
    )
    p = cm.p
    Lam = cm.ridge.copy()
    # thin-width penalty: non-zero only on the K - 1 width coordinates, so kept as that block (Qw) alone
    st = cm.slices["tau"]
    Qw = (
        width_penalty_matrix(np.asarray(width_rho, float))
        if width_rho is not None and np.any(width_rho)
        else np.zeros((st.stop - st.start,) * 2)
    )
    if item_ridge:
        Lam[cm.slices["delta"]] += 0.5 * float(item_ridge)  # delta = N_I coords, orthonormal
    if width_ridge:
        Qw = Qw + 0.5 * float(width_ridge) * _width_centring(st.stop - st.start, fixed_widths)
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
        lev, glev = cm.level_penalty(th)
        pen = float(th @ (Lam * th)) + float(th[st] @ Qw @ th[st]) + lev
        obj = _loglik(eta, t, u) - pen
        if order == 0:
            return obj, None
        s = 1 / (1 + np.exp(-eta))
        gb = D.T @ (t - (t + u) * s)
        g = cm.grid_adjoint(gb.reshape(R, I, K)) - 2 * Lam * th - glev
        g[st] -= 2 * Qw @ th[st]
        return obj, (g, s)

    info = dict(iterations=0, converged=False, optimiser="newton-raphson")
    if R * I * K * p <= DENSE_LIMIT:
        J = cm.jacobian()
        Qlev = cm.level_matrix()
        th = theta
        obj, (g, s) = objective(th)
        dec = np.inf
        for it in range(1, max_iter + 1):
            w = (t + u) * s * (1 - s)
            Hb = (D.T @ sparse.diags(w) @ D).tocsr()
            HbJ = Hb.toarray() @ J if R * I * K <= 6000 else Hb @ J
            negH = J.T @ HbJ + 2 * np.diag(Lam) + Qlev
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
    so phi = phi0 + M theta and the mean omega is 1. Ridge kappa * sum(omega - 1)^2 on the free omega axes, and
    (lambda_ridge / 2) * sum(lambda^2) on the facet element global effects.
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
        lambda_ridge=0.0,
        item_ridge=0.0,
        width_ridge=0.0,
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
            if off == oL:
                self.lambda_cols = slice(col, col + b.shape[1])
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
        if lambda_ridge:
            # lambda = N_R l with N_R orthonormal, so sum(lambda^2) = |l|^2
            lc = self.lambda_cols
            self.Q[lc, lc] += 0.5 * float(lambda_ridge) * np.eye(lc.stop - lc.start)
        if item_ridge:
            self.Q[: I - 1, : I - 1] += 0.5 * float(item_ridge) * np.eye(I - 1)
        if width_ridge:
            tc = self.tau_cols
            self.Q[tc, tc] += 0.5 * float(width_ridge) * _width_centring(tc.stop - tc.start, drop_widths)

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
    location grid b (R, I, K) from their patterns. X (R, P, I). Returns (P, R, I, K).
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


def person_comparison_terms(R, I, K, drop_widths=(), empty=None):
    """The comparisons a person's ratings enter in the composite likelihood (any number of facet elements), as
    arrays over terms: first and second cell (facet element x item, flattened r * I + i), the two category pairs
    (alpha1, beta1) and (alpha2, beta2) whose indicators give the person's weight in the term, e = 1(x_first =
    alpha1, x_second = beta1) + 1(x_first = alpha2, x_second = beta2), and the grid entries (flattened (r, i, k))
    of the logit gx - gy. empty=None: CPAT's patterns ('cpat-rcl', person_grid_scores); else the full pattern set
    with these empty categories ('cp-full-rcl', full_person_grid_scores). Pairs of cells are those of the scores:
    the items of one facet element, and the facet elements of one item.
    """
    drop_widths = set(drop_widths)
    within = [(r * I + i, r * I + j) for r in range(R) for i in range(I) for j in range(i + 1, I)]
    between = [(r * I + i, q * I + i) for i in range(I) for r in range(R) for q in range(r + 1, R)]
    rows = []  # (first, second, a1, b1, a2, b2, thr of second, thr of first)
    if empty is None:
        for f, sc in within + between:
            for c in range(K):
                rows.append((f, sc, c + 1, c, c, c + 1, c, c))
        for f, sc in within:
            for kk in range(K - 1):
                if kk in drop_widths:
                    continue
                k = kk + 1
                for a, bcell in ((f, sc), (sc, f)):
                    rows.append((a, bcell, k, k, k - 1, k + 1, kk + 1, kk))
    else:
        for a, bb, d in _full_patterns(K, drop_widths, empty):
            for f, sc in within + between:
                rows.append((f, sc, a, bb, a - 1, bb + 1, bb, a - 1))
    if not rows:
        z = np.zeros(0, int)
        return dict(first=z, second=z, a1=z, b1=z, a2=z, b2=z, gx=z, gy=z)
    t = np.asarray(rows, dtype=int)
    return dict(first=t[:, 0], second=t[:, 1], a1=t[:, 2], b1=t[:, 3], a2=t[:, 4], b2=t[:, 5],
                gx=t[:, 1] * K + t[:, 6], gy=t[:, 0] * K + t[:, 7])


def kc_grid_scores(U, b, X, K, J, bread_inv, drop_widths=(), empty=None, chunk=2000):
    """
    Kauermann-Carroll (2001) correction of the composite-likelihood scores U (persons x p, in the coordinates
    whose grid Jacobian is J (R I K x p)) of an RCL fit with any number of facet elements: each person's score u_n
    becomes (I - H_n)^(-1/2) u_n, H_n = B_n B^+, as Rasch._kc_scores, with B_n = J' (sum over the person's
    comparisons of e w (delta_x - delta_y)(delta_x - delta_y)') J their contribution to the unpenalised information
    (w = s (1 - s) at the grid b (R, I, K); for a grid not linear in the coordinates, the Gauss-Newton part, the
    usual choice). The inverse square root is taken from the eigendecomposition of S' B_n S (S S' = bread_inv),
    whose eigenvalues are those of H_n. X (R, P, I); bread_inv the pseudo-inverse of the penalised bread
    in the same coordinates; empty as person_comparison_terms. Persons with the same ratings are computed once. A
    person whose I - H_n is near-singular keeps their score.
    """
    U = np.asarray(U, dtype=float)
    R, P, I = X.shape
    J = np.asarray(J.todense() if hasattr(J, "todense") else J, dtype=float)
    terms = person_comparison_terms(R, I, K, drop_widths, empty)
    if not len(terms["gx"]):
        return U.copy()
    A = np.asarray(bread_inv, dtype=float)
    w_e, Q = np.linalg.eigh((A + A.T) / 2)
    keep = w_e > 1e-10 * max(w_e.max(), 1e-300)
    Qk = Q[:, keep]
    S, T = Qk * np.sqrt(w_e[keep]), Qk / np.sqrt(w_e[keep])
    bf = np.asarray(b, dtype=float).ravel()
    s = _sig(bf[terms["gx"]] - bf[terms["gy"]])
    w = s * (1 - s)
    JS = J @ S  # (grid, p_eff)
    gx, gy = terms["gx"], terms["gy"]
    SdJ = JS[gx] - JS[gy]  # (terms, p_eff): S' (J_x - J_y) for each term
    Xp = np.transpose(X, (1, 0, 2)).reshape(P, R * I)
    rows, inv = np.unique(np.where(np.isnan(Xp), -1.0, Xp), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    groups = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv, minlength=len(rows)))[:-1])
    out = U.copy()
    f, sc = terms["first"], terms["second"]
    pe = SdJ.shape[1]
    D2 = None
    if pe * pe * len(w) <= 2e7:  # every term's outer product, so each person's S' B_n S is one matrix product
        D2 = (SdJ[:, :, None] * SdJ[:, None, :]).reshape(len(w), pe * pe)
    chunk = max(1, min(chunk, int(5e6 // (pe * pe))))  # bounds the stack of S' B_n S matrices
    for start in range(0, len(rows), chunk):
        Xr = rows[start: start + chunk]
        E = ((Xr[:, f] == terms["a1"]) & (Xr[:, sc] == terms["b1"])).astype(float) + (
            (Xr[:, f] == terms["a2"]) & (Xr[:, sc] == terms["b2"])
        )
        Ew = E * w
        if D2 is not None:
            M = (Ew @ D2).reshape(-1, pe, pe)
        else:
            M = np.empty((Xr.shape[0], pe, pe))
            for g in range(Xr.shape[0]):
                nz = np.nonzero(Ew[g])[0]
                cells, loc = np.unique(np.concatenate([gx[nz], gy[nz]]), return_inverse=True)
                if len(nz) <= len(cells):  # few comparisons: sum their outer products directly
                    C = SdJ[nz] * np.sqrt(Ew[g, nz])[:, None]
                    M[g] = C.T @ C
                else:
                    # many comparisons: through the grid cells they touch, S'B_nS = (JS)' L (JS), with L the
                    # person's weighted comparison Laplacian over those cells, L (JS) = D' (w (JS_x - JS_y))
                    # for the comparisons' incidence matrix D (+1 at x, -1 at y)
                    m = len(nz)
                    D = sparse.csr_matrix(
                        (np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], loc)),
                        shape=(m, len(cells)),
                    )
                    M[g] = JS[cells].T @ (D.T @ (SdJ[nz] * Ew[g, nz][:, None]))
        M = (M + np.transpose(M, (0, 2, 1))) / 2
        lam_all, V_all = np.linalg.eigh(M)  # S' B_n S = V diag(lam) V'; lam are the eigenvalues of H_n
        lam_all = np.clip(lam_all, 0.0, None)
        ok = np.isfinite(lam_all).all(1) & (lam_all.max(1) < 1 - 1e-8) & (lam_all.max(1) > 0)
        gs = np.nonzero(ok)[0]
        if not len(gs):
            continue
        members = np.concatenate([groups[start + g] for g in gs])
        gi = np.repeat(np.arange(len(gs)), [len(groups[start + g]) for g in gs])
        fac = (1 - lam_all[gs]) ** -0.5 - 1
        V = V_all[gs]
        blk = max(1, int(5e6 // (pe * pe)))
        for i0 in range(0, len(members), blk):
            mb, gb = members[i0: i0 + blk], gi[i0: i0 + blk]
            u = U[mb]
            y = u @ S
            Vb = V[gb]
            z = y + np.einsum("nij,nj->ni", Vb, np.einsum("nji,nj->ni", Vb, y) * fac[gb])
            adj = z @ T.T + (u - (u @ Qk) @ Qk.T)
            out[mb] = np.where(np.isfinite(adj), adj, u)
    return out


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
    small_sample=True applies the Kauermann-Carroll correction (kc_grid_scores; kc_scores is not used).
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
        U = kc_grid_scores(U, b, X, K, J[:, free], A, drop, empty)
    return A @ (U.T @ U) @ A.T, free, null_directions(Bf)


def cell_posterior(cm, theta, info):
    """Inverse of the negative penalised Hessian of a cell-model fit at its estimate, over the free coordinates
    (the spread the penalties allow; std_errors(no_data_se='posterior') for parameters the data do not inform).
    Returns (V, free)."""
    st = cell_state(cm, theta, info)
    free = st["free"]
    return np.linalg.pinv(st["negH"][np.ix_(free, free)]), free


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
        negH = J.T @ (Hb @ J) + 2 * np.diag(lz["Lam"]) + cm.level_matrix()
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


def width_ridge_rows(K, unused, p, tau_slice):
    """Rows P (on theta, p coordinates, the K - 1 category widths at tau_slice) of the unused-category width
    ridge, (weight / 2) |P theta|^2: each width next to an unused score category (unused, 0..K) minus the mean
    of the other widths; with only two categories used, each such width itself (towards 0). Returns (P, the
    0-based widths penalised)."""
    nw = K - 1
    adj = sorted({m for j in unused for m in (j - 2, j - 1, j) if 0 <= m < nw})
    P = np.zeros((len(adj), p))
    if K + 1 - len(set(unused)) <= 2:
        # only two categories used: every width is next to an unused category, and the likelihood pushes them all
        # down (making the unused categories ever less likely); pulling them towards each other would leave their
        # common size free to run off, so pull each towards 0 instead
        for row, m in enumerate(adj):
            P[row, tau_slice.start + m] = 1.0
        return P, adj
    others = [m for m in range(nw) if m not in adj] or list(range(nw))
    for row, m in enumerate(adj):
        P[row, tau_slice.start + m] += 1.0
        P[row, [tau_slice.start + o for o in others]] -= 1.0 / len(others)
    return P, adj


def null_structure(K, used):
    """unused_categories='null' for shared thresholds with categories 0..K of which used (bool, K + 1, at least
    two True) occur (Wilson and Masters, 1993): marks (K,) -- -inf below the lowest used category, +inf at an
    unused category's threshold, 0 at the free ones (as _esf.unused_category_offset); mult (K,) -- how many
    thresholds each free threshold holds (1 + the run of unused categories just below it; 0 at the markers);
    G (K, K) with the free threshold h holding the sum of thresholds G[h] @ b (rows of the markers zero)."""
    from raschpy import _esf

    marks = _esf.unused_category_offset(K, used)
    G = np.zeros((K, K))
    mult = np.zeros(K)
    run = []
    for h in range(K):
        if marks[h] == np.inf:
            run.append(h)
        elif marks[h] == 0:
            G[h, run + [h]] = 1.0
            mult[h] = len(run) + 1
            run = []
        else:
            run = []
    return marks, mult, G


class MarkedDesign:
    """A cell or stretch model's location grid under unused_categories='null': the K thresholds of each (facet
    element, item) cell grouped by G (null_structure) with the markers as offsets, in reduced coordinates phi
    (theta = V phi, V an orthonormal basis of the directions the grouped grid identifies). For fit_cml_stretch
    (stretch base designs) and for mapping estimates back (theta(phi))."""

    def __init__(self, base, marks, G, V):
        self.base, self.R, self.I, self.K = base, base.R, base.I, base.K
        self.marks, self.G, self.V = marks, G, V
        self.Gc = np.kron(np.eye(base.R * base.I), G)
        self.offset = np.tile(np.where(marks == 0, 0.0, marks), base.R * base.I)
        self.p = V.shape[1]

    def theta(self, phi):
        return self.V @ phi

    def grid(self, phi):
        return (self.Gc @ self.base.grid(self.V @ phi).ravel() + self.offset).reshape(self.R, self.I, self.K)

    def grid_jacobian(self, phi):
        return self.Gc @ self.base.grid_jacobian(self.V @ phi) @ self.V

    def bilinear_hessian(self, phi, Gm):
        Gb = (self.Gc.T @ np.nan_to_num(Gm).ravel()).reshape(self.R, self.I, self.K)
        return self.V.T @ self.base.bilinear_hessian(self.V @ phi, Gb) @ self.V


def identified_basis(Aeff, mult_cells, tol=1e-9):
    """Orthonormal basis V of the coordinates identified by the grouped (marked) grid Aeff (cells x p): its row
    space, less the direction that adds mult_h to every cell's threshold h (a shift the conditional likelihood
    cannot see, offset by the persons' origin), when the coordinates can produce it."""
    _, sv, Vt = np.linalg.svd(Aeff, full_matrices=False)
    r = int((sv > tol * max(sv.max(), 1.0)).sum())
    V = Vt[:r].T
    d, *_ = np.linalg.lstsq(Aeff @ V, mult_cells, rcond=None)
    if np.linalg.norm(Aeff @ V @ d - mult_cells) < 1e-8 * max(np.linalg.norm(mult_cells), 1.0) and np.linalg.norm(d) > 0:
        d = d / np.linalg.norm(d)
        V = V @ null_space(d[None, :])
    return V


class StretchDesign:
    """The stretch models' parameters for CML (as StretchModel's, without the RCL comparisons): natural
    phi = [delta, tau, lambda, omegaI, omegaT] = phi0 + M theta, offsets off; theta's blocks delta (N_I),
    widths (T), lambda (N_R), omegaI - 1 and omegaT - 1 (N_R, where the model has them)."""

    def __init__(self, model, R, I, K):
        self.model, self.R, self.I, self.K = model, R, I, K
        self.useI = model in ("pseudo_halo", "bistretch")
        self.useT = model in ("centrality", "bistretch")
        oD, oT, oL, oI, oW = 0, I, I + K, I + K + R, I + K + 2 * R
        self.off, self.p_nat = (oD, oT, oL, oI, oW), I + K + 3 * R
        NI, NR, T = zero_sum_basis(I), zero_sum_basis(R), threshold_basis(K)
        blocks = (
            [("delta", oD, NI), ("tau", oT, T), ("lambda", oL, NR)]
            + ([("omegaI", oI, NR)] if self.useI else [])
            + ([("omegaT", oW, NR)] if self.useT else [])
        )
        self.M = np.zeros((self.p_nat, sum(b.shape[1] for _, _, b in blocks)))
        self.slices, col = {}, 0
        for name, off, b in blocks:
            self.M[off : off + b.shape[0], col : col + b.shape[1]] = b
            self.slices[name] = slice(col, col + b.shape[1])
            col += b.shape[1]
        self.tau_cols = self.slices["tau"]
        self.p = self.M.shape[1]
        self.phi0 = np.zeros(self.p_nat)
        self.phi0[oI : oI + R] = 1.0
        self.phi0[oW : oW + R] = 1.0

    def theta_from(self, delta, tau, lam, omegaI, omegaT):
        phi = np.concatenate([delta - np.mean(delta), tau - np.mean(tau), lam - np.mean(lam),
                              omegaI - np.mean(omegaI) + 1, omegaT - np.mean(omegaT) + 1])
        R, (_, _, _, oI, oW) = self.R, self.off
        if not self.useI:
            phi[oI : oI + R] = 1
        if not self.useT:
            phi[oW : oW + R] = 1
        return np.linalg.lstsq(self.M, phi - self.phi0, rcond=None)[0]

    def unpack(self, theta):
        phi = self.phi0 + self.M @ theta
        oD, oT, oL, oI, oW = self.off
        R = self.R
        return phi[oD:oT], phi[oT:oL], phi[oL:oI], phi[oI : oI + R], phi[oW : oW + R]

    def grid(self, theta):
        d, tau, lam, wI, wT = self.unpack(theta)
        return lam[:, None, None] + wI[:, None, None] * d[None, :, None] + wT[:, None, None] * tau[None, None, :]

    def grid_jacobian(self, theta):
        """(R*I*K, p) d grid / d theta at theta."""
        d, tau, lam, wI, wT = self.unpack(theta)
        R, I, K = self.R, self.I, self.K
        oD, oT, oL, oI, oW = self.off
        r, i, h = np.meshgrid(np.arange(R), np.arange(I), np.arange(K), indexing="ij")
        Jn = np.zeros((R, I, K, self.p_nat))
        Jn[r, i, h, oD + i] = wI[r]
        Jn[r, i, h, oT + h] = wT[r]
        Jn[r, i, h, oL + r] = 1.0
        Jn[r, i, h, oI + r] = d[i]
        Jn[r, i, h, oW + r] = tau[h]
        return Jn.reshape(R * I * K, self.p_nat) @ self.M

    def bilinear_hessian(self, theta, G):
        """The second-derivative term of the bilinear grid in theta, given the gradient G (R, I, K) of the
        log-likelihood in the grid: M' (S + S') M."""
        R, I, K = self.R, self.I, self.K
        oD, oT, oL, oI, oW = self.off
        S = np.zeros((self.p_nat, self.p_nat))
        S[oD : oD + I, oI : oI + R] = G.sum(axis=2).T
        S[oT : oT + K, oW : oW + R] = G.sum(axis=1).T
        return self.M.T @ (S + S.T) @ self.M


def fit_cml_stretch(X, design, theta0, Q, tol=1e-7, max_iter=200, linear=None, prep=None, quiet=False):
    """Penalised CML for a stretch model: maximise cond_loglik(X, grid) - theta' Q theta / 2 by Newton-Raphson
    on the exact Hessian (grid Jacobian term plus the bilinear term), with step halving; a gradient step where
    the penalised information is not positive definite. X: persons x (facet element, item) virtual items.
    linear (flat grid vector): maximise cond_loglik(X, grid) - linear . grid - theta' Q theta / 2 instead (the
    conditional log-likelihood averaged over other responses with the same raw scores; ll then includes the
    linear term). prep: _esf.prepare(X, ...) if already computed; quiet: no progress reporting.
    Returns dict(theta, b, ll, objective, converged, iterations, max_gradient, info (negative penalised
    Hessian), hessian (of cond_loglik in the grid))."""
    from raschpy import _esf

    R, I, K = design.R, design.I, design.K
    if prep is None:
        prep = _esf.prepare(X, [K] * (R * I))

    def evaluate(theta):
        b = design.grid(theta)
        ll, g, H = _esf.cond_loglik(X, list(b.reshape(R * I, K)), hessian=True, prepared=prep)
        g, H = np.nan_to_num(g), np.nan_to_num(H)  # unused-category markers (inf) have no derivatives
        if linear is not None:
            ll, g = ll - float(linear @ b.ravel()), g - linear
        J = design.grid_jacobian(theta)
        gt = J.T @ g - Q @ theta
        F = J.T @ H @ J - Q  # without the bilinear term: negative definite where the data and penalty inform theta
        Ht = F + design.bilinear_hessian(theta, g.reshape(R, I, K))
        return ll, ll - 0.5 * theta @ Q @ theta, gt, -(Ht + Ht.T) / 2, H, -(F + F.T) / 2

    from raschpy import _progress

    theta = np.asarray(theta0, float).copy()
    ll, obj, g, info, H, fisher = evaluate(theta)
    converged, it = False, 0
    rep = None if quiet else _progress.Reporter("CML fit", unit="iterations").__enter__()
    for it in range(1, max_iter + 1):
        try:
            ev = np.linalg.eigvalsh(info)
            if ev.min() > 0:
                step = np.linalg.solve(info, g)
            elif np.linalg.eigvalsh(fisher).min() > 0:
                # the exact Hessian is indefinite away from the optimum (its bilinear term): a Gauss-Newton step
                step = np.linalg.solve(fisher, g)
            else:
                step = g / max(np.abs(g).max(), 1.0) * 0.1
        except np.linalg.LinAlgError:
            break
        s = 1.0
        while True:
            new = theta + s * step
            out = evaluate(new)
            if (np.isfinite(out[1]) and out[1] >= obj - 1e-12 * abs(obj)) or s < 1e-8:
                break
            s /= 2
        d_grid = design.grid(new) - design.grid(theta)
        change = np.max(np.abs(d_grid[np.isfinite(d_grid)]), initial=0.0)
        theta, (ll, obj, g, info, H, fisher) = new, out
        if rep is not None:
            rep.update(note=f"largest change {change:.2g}")
        if change < tol:
            converged = True
            break
    if rep is not None:
        rep.close()
    return dict(theta=theta, b=design.grid(theta).ravel(), ll=ll, objective=obj, converged=converged,
                iterations=it, max_gradient=float(np.abs(g).max()), info=info, hessian=H)


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


def stretch_sandwich(sm, theta, X, K, fixed=(), drop_widths=(), small_sample=False):
    """Sandwich covariance of the free stretch-model parameters theta (fixed: theta indices held fixed). Bread =
    negative penalised Hessian at the estimate, meat = sum over persons of their score outer products.
    Returns (V, free, null) with V over the free coordinates and null as in cell_sandwich(). small_sample=True
    applies the Kauermann-Carroll correction (kc_grid_scores).
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
    if small_sample:
        U = kc_grid_scores(U, b, X, K, J, A, drop_widths, sm.full_empty)
    return A @ (U.T @ U) @ A.T, free, null_directions(Bf)


# ----------------------------------------------------------------------------------------------------------------
# PCM: full-pattern pairwise conditional likelihood over item pairs (estimation='cp-full-rcl')
# ----------------------------------------------------------------------------------------------------------------


def pcm_comparisons(X, Ks, cpat=False):
    """
    Comparison list (D, t, u) for PCM's full pattern set from responses X (persons x items, NaN missing) with
    maximum scores Ks: for every item pair (i < j), a = 1..K_i and b = 0..K_j - 1, t persons with
    (x_i, x_j) = (a, b) and u with (a - 1, b + 1) (the same pair total), logit P((a, b)) = d[j, b+1] - d[i, a]
    (uncentred thresholds, one-based; d stacked item by item). D (n, sum Ks) is sparse with +1 and -1 at those
    thresholds; comparisons with t + u = 0 are left out. cpat=True keeps CPAT's patterns only ('cpat-rcl'):
    |b - (a - 1)| <= 1, the same threshold on the two items (odd pair totals) or adjacent ones (even totals).
    """
    return pcm_comparisons_from_counts(pair_counts(X, Ks), Ks, cpat)


def pair_counts(X, Ks):
    """C[(i, a), (j, b)] = number of persons with x_i = a and x_j = b (rows and columns stacked item by item,
    categories 0..K_i)."""
    X = np.asarray(X, float)
    Ks = [int(k) for k in Ks]
    coff = np.concatenate([[0], np.cumsum([k + 1 for k in Ks])]).astype(int)
    Z = np.zeros((X.shape[0], coff[-1]))
    for i, K in enumerate(Ks):
        x = X[:, i]
        obs = ~np.isnan(x)
        Z[np.flatnonzero(obs), coff[i] + x[obs].astype(int)] = 1.0
    return Z.T @ Z


def pcm_comparisons_from_counts(C, Ks, cpat=False):
    """pcm_comparisons' (D, t, u) from the pair counts C (pair_counts); C may hold expected counts."""
    Ks = [int(k) for k in Ks]
    off = np.concatenate([[0], np.cumsum(Ks)]).astype(int)
    coff = np.concatenate([[0], np.cumsum([k + 1 for k in Ks])]).astype(int)
    xs, ys, ts, us = [], [], [], []
    for i in range(len(Ks)):
        for j in range(i + 1, len(Ks)):
            a = np.repeat(np.arange(1, Ks[i] + 1), Ks[j])
            b = np.tile(np.arange(0, Ks[j]), Ks[i])
            t = C[coff[i] + a, coff[j] + b]
            u = C[coff[i] + a - 1, coff[j] + b + 1]
            k = (t + u) > 0
            if cpat:
                k &= np.abs(b - (a - 1)) <= 1
            xs.append(off[j] + b[k])
            ys.append(off[i] + a[k] - 1)
            ts.append(t[k])
            us.append(u[k])
    x, y = np.concatenate(xs), np.concatenate(ys)
    n = len(x)
    D = sparse.csr_matrix(
        (np.r_[np.ones(n), -np.ones(n)], (np.tile(np.arange(n), 2), np.r_[x, y])),
        shape=(n, off[-1]),
    )
    return D, np.concatenate(ts), np.concatenate(us)


def fit_pcm(X, Ks, b0, Pb=None, tol=1e-9, max_iter=100, max_abs=30.0, cpat=False):
    """
    PCM full-pattern RCL (CPAT's patterns with cpat=True, 'cpat-rcl'): maximise the composite log-likelihood of
    pcm_comparisons() minus b' Pb b / 2 over
    zero-sum uncentred thresholds b, by Newton-Raphson (concave) from b0, halving the step while the objective
    falls. Returns dict(b, ll, objective, converged, iterations, info): info the negative Hessian of the
    objective in the zero-sum coordinates, A the basis.
    """
    from scipy.linalg import cho_factor, cho_solve, LinAlgError

    D, t, u = pcm_comparisons(X, Ks, cpat)
    p = D.shape[1]
    A = zero_sum_basis(p)
    Pb = np.zeros((p, p)) if Pb is None else np.asarray(Pb, float)
    n = t + u
    b0 = np.asarray(b0, float)
    th = A.T @ (b0 - b0.mean())

    def evaluate(th):
        b = A @ th
        eta = D @ b
        s = 1.0 / (1.0 + np.exp(-eta))
        ll = float((-t * np.logaddexp(0, -eta) - u * np.logaddexp(0, eta)).sum())
        g = A.T @ (D.T @ (t - n * s) - Pb @ b)
        W = sparse.diags(n * s * (1 - s))
        info = A.T @ (np.asarray((D.T @ W @ D).todense()) + Pb) @ A
        return ll - 0.5 * b @ Pb @ b, g, info, ll

    f, g, info, ll = evaluate(th)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        try:
            step = cho_solve(cho_factor(info), g)
        except LinAlgError:
            break  # not identified: no finite maximum
        s = 1.0
        while True:
            f2, g2, info2, ll2 = evaluate(th + s * step)
            if f2 >= f - 1e-12 * abs(f) or s < 1e-8:
                break
            s /= 2
        change = np.max(np.abs(A @ (s * step)))
        th = th + s * step
        f, g, info, ll = f2, g2, info2, ll2
        if change < tol:
            converged = True
            break
        if np.max(np.abs(A @ th)) > max_abs:
            break  # diverging: no finite maximum
    b = A @ th
    return dict(b=b, ll=ll, objective=f, converged=bool(converged), iterations=it, info=info, A=A, cpat=cpat)


def pcm_person_scores(X, Ks, b, cpat=False):
    """Each person's contribution to the gradient of the composite log-likelihood of pcm_comparisons() (with the
    same cpat) with respect to the stacked uncentred thresholds b, (n_persons, sum Ks)."""
    X = np.asarray(X, float)
    Ks = [int(k) for k in Ks]
    off = np.concatenate([[0], np.cumsum(Ks)]).astype(int)
    S = np.zeros((X.shape[0], off[-1]))
    rows = np.arange(X.shape[0])
    for i in range(len(Ks)):
        for j in range(i + 1, len(Ks)):
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            r = rows[ok]
            a, c = X[ok, i].astype(int), X[ok, j].astype(int)
            # as the (a, c) side of comparison (a, c) vs (a - 1, c + 1): weight 1 - sigma(eta)
            m = (a >= 1) & (c <= Ks[j] - 1)
            if cpat:
                m &= np.abs(c - (a - 1)) <= 1
            pj, pi = off[j] + c[m], off[i] + a[m] - 1
            w = 1.0 - 1.0 / (1.0 + np.exp(-(b[pj] - b[pi])))
            np.add.at(S, (r[m], pj), w)
            np.add.at(S, (r[m], pi), -w)
            # as the (a, c) side of comparison (a + 1, c - 1) vs (a, c): weight -sigma(eta)
            m = (a + 1 <= Ks[i]) & (c >= 1)
            if cpat:
                m &= np.abs((c - 1) - a) <= 1
            pj, pi = off[j] + c[m] - 1, off[i] + a[m]
            w = -1.0 / (1.0 + np.exp(-(b[pj] - b[pi])))
            np.add.at(S, (r[m], pj), w)
            np.add.at(S, (r[m], pi), -w)
    return S


def pcm_person_bread(X, Ks, b, cpat=False):
    """Each person's contribution to the negative Hessian of the composite log-likelihood of pcm_comparisons()
    (same cpat) with respect to the stacked uncentred thresholds b, (n_persons, sum Ks, sum Ks): the sum over the
    person's comparisons of s (1 - s) d d', d = e_j - e_i the comparison's threshold contrast."""
    X = np.asarray(X, float)
    Ks = [int(k) for k in Ks]
    off = np.concatenate([[0], np.cumsum(Ks)]).astype(int)
    p, n = int(off[-1]), X.shape[0]
    rows = np.arange(n)
    R_, J_, I_, V_ = [], [], [], []
    for i in range(len(Ks)):
        for j in range(i + 1, len(Ks)):
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            r = rows[ok]
            a, c = X[ok, i].astype(int), X[ok, j].astype(int)
            for m, pj, pi in (
                ((a >= 1) & (c <= Ks[j] - 1) & ((np.abs(c - (a - 1)) <= 1) if cpat else True),
                 off[j] + c, off[i] + a - 1),
                ((a + 1 <= Ks[i]) & (c >= 1) & ((np.abs((c - 1) - a) <= 1) if cpat else True),
                 off[j] + c - 1, off[i] + a),
            ):
                m = np.broadcast_to(m, a.shape)
                pj_, pi_ = pj[m], pi[m]
                s = 1.0 / (1.0 + np.exp(-(b[pj_] - b[pi_])))
                R_.append(r[m]); J_.append(pj_); I_.append(pi_); V_.append(s * (1 - s))
    r, pj, pi, v = (np.concatenate(x) for x in (R_, J_, I_, V_))
    base = r * (p * p)
    idx = np.concatenate([base + pj * p + pj, base + pi * p + pi, base + pj * p + pi, base + pi * p + pj])
    w = np.concatenate([v, v, -v, -v])
    return np.bincount(idx, weights=w, minlength=n * p * p).reshape(n, p, p)


def pcm_sandwich(X, Ks, b, info, A, cpat=False, small_sample=False, kc_scores=None):
    """Sandwich (Godambe) covariance of the PCM RCL thresholds b = A theta: A info^{-1} M info^{-1} A', info the
    negative Hessian of the (penalised) objective from fit_pcm(), M the sum over persons of the outer products of
    their scores (pcm_person_scores, once per distinct response row). small_sample=True applies the
    Kauermann-Carroll correction to the scores (kc_scores: Rasch._kc_scores), with each person's bread from
    pcm_person_bread."""
    rows, counts = np.unique(np.where(np.isnan(X), -1.0, X), axis=0, return_counts=True)
    rows = np.where(rows < 0, np.nan, rows)
    U = pcm_person_scores(rows, Ks, b, cpat) @ A
    Binv = np.linalg.inv(info)
    if small_sample:
        U = kc_scores(U, lambda sl: A.T @ pcm_person_bread(rows[sl], Ks, b, cpat) @ A, Binv)
    M = U.T @ (counts[:, None] * U)
    return A @ Binv @ M @ Binv @ A.T


# ----------------------------------------------------------------------------------------------------------------
# 'mixed': each facet element on its own representation
# ----------------------------------------------------------------------------------------------------------------

# two internal representations, used only inside per_rater_model_selection(test='LR') and never reported: a full
# set of effects on one side beside a stretch on the other (item effects + threshold stretch, threshold effects +
# item stretch)
_MIXED_HIDDEN = {"items_centrality": "items", "thresholds_pseudo_halo": "thresholds"}
_MIXED_ITEM_EFFECTS = ("items", "bivector", "matrix", "items_centrality")
_MIXED_THRESHOLD_EFFECTS = ("thresholds", "bivector", "matrix", "thresholds_pseudo_halo")
_MIXED_ITEM_STRETCH = ("pseudo_halo", "bistretch", "thresholds_pseudo_halo")
_MIXED_THRESHOLD_STRETCH = ("centrality", "bistretch", "items_centrality")


def cut_representation(kinds):
    """The representation an anchor's cuts imply, from the kinds of its cut cells ('item', 'threshold', 'cell'):
    item cuts only -> 'items', threshold cuts only -> 'thresholds', both -> 'bivector' (item and threshold shifts
    add), any item x threshold cell -> 'matrix'."""
    kinds = set(kinds)
    if "cell" in kinds:
        return "matrix"
    if kinds == {"item"}:
        return "items"
    if kinds == {"threshold"}:
        return "thresholds"
    return "bivector"


class MixedDesign:
    """
    The location grid of MFRM 'mixed', each facet element on its own representation, in the frame of the anchors
    (anchors: facet element indices; None means all of them, the average facet element). Natural parameters
    phi = [delta (I), tau (K), lambda (R), A (R, I), T (R, K), E (R, I, K), omegaI (R), omegaT (R)], grid
        b[r, i, h] = lambda_r + c_r delta_i + d_r tau_h + A_ri + T_rh + E_rih,
    with A (item effects; items / bivector / matrix) summing to 0 over items within each facet element, T
    (threshold effects; thresholds / bivector / matrix) summing to 0 over thresholds, E (interactions; matrix)
    summing to 0 over both, and omegaI / omegaT the stretch multipliers (pseudo_halo, bistretch / centrality,
    bistretch; 1 for the others). The frame (frame='part', the default): the anchors' global effects sum to zero,
    and in each part (item: A and omegaI; threshold: T and omegaT; interaction: E) the anchors whose
    representation lacks that part (e.g. those on 'global') agree with the instrument there by definition, so the
    anchors with the part are free in it. Where every anchor has the part (and always with frame='average', an
    anchor without it counting as zero), the anchors' average deviation from the instrument (delta_i + tau_h)
    is zero in that part -- sum over the anchors of lambda, of A_ri + (c_r - 1) delta_i for each item, of
    T_rh + (d_r - 1) tau_h for each threshold, and of E. c_r = omegaI_r for the item-stretch facet elements and 1
    for the others, except that when an anchor has item effects, the first such anchor carries the anchors'
    item stretch, c = 1 - sum over the anchor item-stretch facet elements of (omegaI - 1), which keeps the item
    constraint linear; when no anchor has item effects the anchor multipliers average 1. d_r likewise. With every
    facet element on one representation this is that representation's CellModel / StretchDesign space.
    weights: {'global', 'interaction', 'stretch': float, 'item' / 'threshold': {representation: float}}; the
    penalty 0.5 theta' Q theta is, per facet element with its own representation's weights,
    0.5 [global lambda^2 + item K |A|^2 + threshold I |T|^2 + interaction |E|^2 + stretch (omega - 1)^2] (the
    CellModel levels / StretchDesign penalties). grid / grid_jacobian / bilinear_hessian as StretchDesign, for
    fit_cml_stretch and fit_mixed_composite.
    cuts: {anchor index: [cell, ...]} -- cells where that anchor is not an anchor, each ('item', i), ('threshold',
    h) or ('cell', i, h) (0-based). The anchor (on 'global') gets one free shift S per cut cell, added to its grid
    over the item's row, the threshold's column or the one cell, so the zero point there is the other anchors'
    average and its global effect lambda is defined by its uncut cells (natural parameters phi = [..., S]).
    """

    def __init__(self, models, I, K, anchors=None, weights=None, cuts=None, frame="part"):
        self.models = list(models)
        R = self.R = len(self.models)
        self.I, self.K = I, K
        S = list(range(R)) if anchors is None else sorted(int(a) for a in anchors)
        self.anchors = S
        if frame not in ("part", "average"):
            raise ValueError(f"frame must be 'part' or 'average', got {frame!r}.")
        self.frame = frame
        # frame='part': in each part (item, threshold, interaction), the anchors whose representation lacks it
        # (e.g. those on 'global') agree with the instrument there by definition, so they fix it, and the anchors
        # with that part are free in it (they enter the frame through their global effect only); only when every
        # anchor has the part is it the anchors' average. frame='average': the anchors' average in every part, an
        # anchor without the part counting as zero (a lone anchor with the part is then held at zero).
        carriers = dict(
            item=set(_MIXED_ITEM_EFFECTS) | set(_MIXED_ITEM_STRETCH),
            threshold=set(_MIXED_THRESHOLD_EFFECTS) | set(_MIXED_THRESHOLD_STRETCH),
            interaction={"matrix"},
        )
        self.part_free = {
            # an anchor has the part and another lacks it
            part: frame == "part" and any(self.models[r] not in reps for r in S)
            and any(self.models[r] in reps for r in S)
            for part, reps in carriers.items()
        }
        S_part = {part: ([] if free else S) for part, free in self.part_free.items()}
        sel = np.zeros(R)
        sel[S] = 1.0
        NI, NK = zero_sum_basis(I), zero_sum_basis(K)
        oD, oT, oL = 0, I, I + K
        oA = oL + R
        oTr = oA + R * I
        oE = oTr + R * K
        oI = oE + R * I * K
        oW = oI + R
        self.off = (oD, oT, oL, oA, oTr, oE, oI, oW)
        self.p_nat = oW + R
        blocks = []

        def add(name, basis, row_off):
            B = np.zeros((self.p_nat, basis.shape[1]))
            B[row_off : row_off + basis.shape[0]] = basis
            blocks.append((name, B))

        def effects(raters, inner, n, S):
            """Basis over (R, n): per facet element in raters inner's space, the sum over anchors S zero."""
            if not raters:
                return np.zeros((R * n, 0))
            E = np.zeros((R, len(raters)))
            E[raters, np.arange(len(raters))] = 1.0
            B = np.kron(E, inner)
            anchored = [k for k, r in enumerate(raters) if r in S]
            if anchored:
                C = np.kron(np.eye(len(raters))[anchored].sum(axis=0, keepdims=True), np.eye(n))
                B = B @ null_space(C @ np.kron(np.eye(len(raters)), inner))
            return B

        add("delta", NI, oD)
        add("tau", threshold_basis(K), oT)
        add("lambda", null_space(sel[None, :]) if R > 1 else np.zeros((R, 0)), oL)
        rA = [r for r, m in enumerate(self.models) if m in _MIXED_ITEM_EFFECTS]
        rT = [r for r, m in enumerate(self.models) if m in _MIXED_THRESHOLD_EFFECTS]
        rE = [r for r, m in enumerate(self.models) if m == "matrix"]
        add("A", effects(rA, NI, I, S_part["item"]), oA)
        add("T", effects(rT, NK, K, S_part["threshold"]), oTr)
        add("E", effects(rE, np.kron(NI, NK), I * K, S_part["interaction"]), oE)
        self.Cm = {}
        for name, stretch, free, o, S in (
            ("omegaI", _MIXED_ITEM_STRETCH, rA, oI, S_part["item"]),
            ("omegaT", _MIXED_THRESHOLD_STRETCH, rT, oW, S_part["threshold"]),
        ):
            rs = [r for r, m in enumerate(self.models) if m in stretch]
            Cm = np.zeros((R, R))
            Cm[rs, rs] = 1.0
            Er = np.zeros((R, len(rs)))
            Er[rs, np.arange(len(rs))] = 1.0
            anchor_free = [r for r in free if r in S]
            anchor_stretch = [r for r in rs if r in S]
            if anchor_free and anchor_stretch:
                Cm[anchor_free[0], anchor_stretch] = -1.0
                basis = Er
            elif anchor_stretch:
                Cs = np.zeros((1, R))
                Cs[0, anchor_stretch] = 1.0
                basis = Er @ null_space(Cs @ Er)
            else:
                basis = Er
            self.Cm[name] = Cm
            add(name, basis, o)
        self.cut_cells, Sg = self._cut_cells(cuts, self.anchors, R, I, K)
        self.oS, self.Sg = oW + R, Sg
        # the representation each facet element is reported under: an anchor with cells cut is fitted on 'global'
        # with a free shift at each cut cell, which is the representation its cuts imply with the uncut parts
        # held at its global effect
        self.reported = list(self.models)
        for r in {c[0] for c in self.cut_cells}:
            self.reported[r] = cut_representation(c[1] for c in self.cut_cells if c[0] == r)
        if self.cut_cells:
            # the shifts at cut cells are free (no anchor constraint), after the stretch multipliers
            self.p_nat = self.oS + len(self.cut_cells)
            blocks = [(n, np.vstack([B, np.zeros((len(self.cut_cells), B.shape[1]))])) for n, B in blocks]
            Bs = np.zeros((self.p_nat, len(self.cut_cells)))
            Bs[self.oS:] = np.eye(len(self.cut_cells))
            blocks.append(("S", Bs))
        self.M = np.hstack([B for _, B in blocks])
        self.slices, c = {}, 0
        for name, B in blocks:
            self.slices[name] = slice(c, c + B.shape[1])
            c += B.shape[1]
        self.p = self.M.shape[1]
        self.tau_cols = self.slices["tau"]
        self.phi0 = np.zeros(self.p_nat)
        self.phi0[oI : oW + R] = 1.0  # the stretch multipliers; the cut cells' shifts (after them) start at 0
        if self.cut_cells and any(self.part_free.values()):
            self._check_cuts_identified()
        w = weights or {}
        P = np.zeros(self.p_nat)
        for r, m in enumerate(self.models):
            P[oL + r] = w.get("global", 0.0)
            mw = _MIXED_HIDDEN.get(m, m)  # a hidden representation takes its full-set side's ridge
            if m in _MIXED_ITEM_EFFECTS:
                P[oA + r * I : oA + (r + 1) * I] = w.get("item", {}).get(mw, 0.0) * K
            if m in _MIXED_THRESHOLD_EFFECTS:
                P[oTr + r * K : oTr + (r + 1) * K] = w.get("threshold", {}).get(mw, 0.0) * I
            if m == "matrix":
                P[oE + r * I * K : oE + (r + 1) * I * K] = w.get("interaction", 0.0)
            if m in _MIXED_ITEM_STRETCH:
                P[oI + r] = w.get("stretch", 0.0)
            if m in _MIXED_THRESHOLD_STRETCH:
                P[oW + r] = w.get("stretch", 0.0)
        P[self.oS:] = w.get("global", 0.0)
        self.Q = self.M.T @ (P[:, None] * self.M)

    def _cut_cells(self, cuts, S, R, I, K):
        """[(anchor, kind, index), ...] in a fixed order and the grid map of their shifts (R I K, n)."""
        cells = []
        for r, cs in sorted((cuts or {}).items()):
            r = int(r)
            if r not in S:
                raise ValueError(f"cuts: facet element {r} is not an anchor; only anchors' cells can be cut.")
            if self.models[r] != "global":
                raise ValueError(
                    f"cuts: anchor {r} is on {self.models[r]!r}; cells can be cut only for anchors on 'global'."
                )
            for c in dict.fromkeys(tuple(c) for c in cs):
                kind = c[0]
                ok = (kind == "item" and len(c) == 2 and 0 <= c[1] < I) or (
                    kind == "threshold" and len(c) == 2 and 0 <= c[1] < K) or (
                    kind == "cell" and len(c) == 3 and 0 <= c[1] < I and 0 <= c[2] < K)
                if not ok:
                    raise ValueError(f"cuts: {c!r} is not a cell ('item', i), ('threshold', h) or ('cell', i, h).")
                cells.append((r,) + c)
        Sg = np.zeros((R, I, K, len(cells)))
        for n, (r, kind, *ix) in enumerate(cells):
            if kind == "item":
                Sg[r, ix[0], :, n] = 1.0
            elif kind == "threshold":
                Sg[r, :, ix[0], n] = 1.0
            else:
                Sg[r, ix[0], ix[1], n] = 1.0
        for r in {c[0] for c in cells}:
            items = {c[2] for c in cells if c[0] == r and c[1] == "item"}
            thr = {c[2] for c in cells if c[0] == r and c[1] == "threshold"}
            if len(items) == I or len(thr) == K:
                raise ValueError(
                    f"cuts: anchor {r} has every {'item' if len(items) == I else 'threshold'} cut, so it no longer "
                    "anchors anything; drop it from the anchors instead."
                )
        if not cells:
            return cells, Sg.reshape(R * I * K, 0)
        if not any(self.part_free.values()):
            # the anchors' average in every part: some anchor must remain at each cell to define the zero point
            freed = Sg.sum(axis=3) > 0
            held = (~freed[S]).any(axis=0)
            if not held.all():
                i, h = np.argwhere(~held)[0]
                raise ValueError(
                    f"cuts: every anchor is cut at item {i}, threshold {h + 1}, so the zero point there is undefined; "
                    "at least one anchor must remain at every cell."
                )
        # (frame='part' with an anchor free in some part: the anchors that fix each part are checked once the design
        # is built, _check_cuts_identified)
        return cells, Sg.reshape(R * I * K, len(cells))

    def _check_cuts_identified(self):
        """frame='part' with cuts: the instrument is fixed in each part by the anchors without it, so whether a cut
        cell's shift is identified depends on the representations; it is if the grid's Jacobian has full rank (at a
        generic point, so that the stretch coefficients' directions are not degenerate)."""
        theta = np.random.default_rng(0).normal(0.0, 0.3, self.p)
        J = self.grid_jacobian(theta)
        null = null_directions(J.T @ J)
        if null.size == 0:
            return
        load = np.abs(null[self.slices["S"]]).max(axis=1) if null.ndim > 1 else np.abs(null[self.slices["S"]])
        bad = [self.cut_cells[n] for n in np.flatnonzero(load > 1e-6)]
        what = "; ".join(f"anchor {r} at " + (f"item {ix[0]}" if kind == "item" else f"threshold {ix[0] + 1}"
                                               if kind == "threshold" else f"item {ix[0]}, threshold {ix[1] + 1}")
                         for r, kind, *ix in bad)
        raise ValueError(
            f"cuts: no remaining anchor fixes the instrument at the cut cells ({what}), so their shifts are undefined "
            "with these representations; an anchor without that part (e.g. on 'global') must remain uncut there."
        )

    def unpack(self, theta):
        """Natural parameters: delta, tau, lam, A, T, E, omegaI, omegaT and the coefficients c, d on delta, tau."""
        phi = self.phi0 + self.M @ theta
        oD, oT, oL, oA, oTr, oE, oI, oW = self.off
        R, I, K = self.R, self.I, self.K
        wI, wT = phi[oI:oW], phi[oW : oW + R]
        return dict(S=phi[self.oS:],
            delta=phi[oD:oT], tau=phi[oT:oL], lam=phi[oL:oA], A=phi[oA:oTr].reshape(R, I),
            T=phi[oTr:oE].reshape(R, K), E=phi[oE:oI].reshape(R, I, K), omegaI=wI, omegaT=wT,
            c=1.0 + self.Cm["omegaI"] @ (wI - 1.0), d=1.0 + self.Cm["omegaT"] @ (wT - 1.0),
        )

    def grid(self, theta):
        u = self.unpack(theta)
        R, I, K = self.R, self.I, self.K
        return (
            u["lam"][:, None, None]
            + u["c"][:, None, None] * u["delta"][None, :, None]
            + u["d"][:, None, None] * u["tau"][None, None, :]
            + u["A"][:, :, None]
            + u["T"][:, None, :]
            + u["E"]
            + (self.Sg @ u["S"]).reshape(R, I, K)
        )

    def _jacobian_parts(self):
        """The pieces of grid_jacobian that do not depend on theta, computed once: the constant part (global, item,
        threshold and interaction effects and cut-cell shifts, each entering the grid with coefficient 1), the rows
        of M for delta and tau (entering times c_r and d_r), and the stretch multipliers' rows of M mapped through
        Cm (entering times delta_i and tau_h)."""
        parts = self.__dict__.get("_jparts")
        if parts is None:
            R, I, K, p, M = self.R, self.I, self.K, self.p, self.M
            oD, oT, oL, oA, oTr, oE, oI, oW = self.off
            J0 = (M[oL : oL + R][:, None, None, :] + M[oA:oTr].reshape(R, I, 1, p)
                  + M[oTr:oE].reshape(R, 1, K, p) + M[oE:oI].reshape(R, I, K, p)).reshape(R * I * K, p)
            if self.Sg.shape[1]:
                J0 = J0 + self.Sg @ M[self.oS:]
            parts = self._jparts = (J0, M[oD : oD + I], M[oT : oT + K], self.Cm["omegaI"] @ M[oI:oW],
                                    self.Cm["omegaT"] @ M[oW : oW + R])
        return parts

    def grid_jacobian(self, theta):
        """(R*I*K, p) d grid / d theta at theta: d b[r, i, h] = d lambda_r + c_r d delta_i + delta_i d c_r
        + d_r d tau_h + tau_h d d_r + d A_ri + d T_rh + d E_rih (+ d S), each d phi = M d theta."""
        u = self.unpack(theta)
        R, I, K = self.R, self.I, self.K
        J0, MD, MT, CI, CT = self._jacobian_parts()
        Jv = (u["c"][:, None, None, None] * MD[None, :, None, :] + u["d"][:, None, None, None] * MT[None, None, :, :]
              + u["delta"][None, :, None, None] * CI[:, None, None, :]
              + u["tau"][None, None, :, None] * CT[:, None, None, :])
        return J0 + Jv.reshape(R * I * K, self.p)

    def bilinear_hessian(self, theta, G):
        """The second-derivative term of the bilinear grid, given the gradient G (R, I, K) of the
        log-likelihood in the grid: M' (S + S') M, where S is zero except two blocks (delta against omegaI and tau
        against omegaT), so only those blocks' rows of M enter."""
        oD, oT, oL, oA, oTr, oE, oI, oW = self.off
        I, K, M = self.I, self.K, self.M
        H = M[oD : oD + I].T @ ((G.sum(axis=2).T @ self.Cm["omegaI"]) @ M[oI:oW])
        H = H + M[oT : oT + K].T @ ((G.sum(axis=1).T @ self.Cm["omegaT"]) @ M[oW : oW + self.R])
        return H + H.T

    def theta_from(self, delta, tau, lam=None):
        """Coordinates of shared item locations delta, thresholds tau and global effects lam (centred over the
        anchors), every other parameter at its null value."""
        phi = self.phi0.copy()
        oD, oT, oL, *_ = self.off
        phi[oD:oT] = np.asarray(delta, float) - np.mean(delta)
        phi[oT:oL] = np.asarray(tau, float) - np.mean(tau)
        if lam is not None:
            lam = np.asarray(lam, float)
            phi[oL : oL + self.R] = lam - lam[self.anchors].mean()
        return np.linalg.lstsq(self.M, phi - self.phi0, rcond=None)[0]


def _chol_solve(A, g):
    """A^-1 g by Cholesky factorisation, or None if A is not positive definite."""
    from scipy.linalg import cho_factor, cho_solve

    try:
        return cho_solve(cho_factor(A, check_finite=False), g, check_finite=False)
    except np.linalg.LinAlgError:
        return None


def fit_mixed_composite(design, D, t, u, Q, theta0, tol=1e-10, max_iter=100, fixed=()):
    """
    Penalised composite likelihood fit on a nonlinear design (MixedDesign): maximise
    sum t log sigma(D b) + u log(1 - sigma(D b)) - theta' Q theta / 2, b = design.grid(theta), by Newton-Raphson
    on the exact Hessian (Jacobian term plus the bilinear term) with step halving; a gradient step where the
    penalised information is not positive definite. The coordinates in fixed stay at their theta0 values. Returns
    dict(theta, b, ll, objective, converged, iterations, info (negative penalised Hessian, over every
    coordinate)).
    """
    R, I, K = design.R, design.I, design.K
    D = sparse.csr_matrix(D)
    free = np.ones(len(theta0), bool)
    free[list(fixed)] = False

    def evaluate(theta, order=2):
        eta = D @ design.grid(theta).ravel()
        ll = float(-(t @ np.logaddexp(0, -eta)) - u @ np.logaddexp(0, eta))
        obj = ll - 0.5 * theta @ Q @ theta
        if order == 0:
            return obj
        s = 1 / (1 + np.exp(-eta))
        gb = D.T @ (t - (t + u) * s)
        J = design.grid_jacobian(theta)
        g = J.T @ gb - Q @ theta
        Hb = D.T @ sparse.diags((t + u) * s * (1 - s)) @ D
        fisher = J.T @ (Hb @ J) + Q
        info = fisher - design.bilinear_hessian(theta, gb.reshape(R, I, K))
        return obj, ll, g, (info + info.T) / 2, (fisher + fisher.T) / 2

    theta = np.asarray(theta0, float).copy()
    obj, ll, g, info, fisher = evaluate(theta)
    converged, it = False, 0
    ix = np.ix_(free, free)
    for it in range(1, max_iter + 1):
        gf, inf_f, fis_f = g[free], info[ix], fisher[ix]
        if not (np.isfinite(inf_f).all() and np.isfinite(fis_f).all()):
            break
        # a Cholesky factorisation succeeds exactly when the matrix is positive definite, so it both tests that and
        # gives the Newton step
        sf = _chol_solve(inf_f, gf)
        pd_ = sf is not None
        if not pd_:
            sf = _chol_solve(fis_f, gf)  # indefinite away from the optimum: Gauss-Newton
            if sf is None:
                sf = gf / max(np.abs(gf).max(), 1.0) * 0.1
        if pd_ and float(gf @ sf) < tol:
            converged = True
            break
        step = np.zeros_like(theta)
        step[free] = sf
        a = 1.0
        while a > 1e-10:
            new = evaluate(theta + a * step, order=0)
            if np.isfinite(new) and new >= obj:
                break
            a /= 2
        theta = theta + a * step
        obj, ll, g, info, fisher = evaluate(theta)
    return dict(theta=theta, b=design.grid(theta).ravel(), ll=ll, objective=obj, converged=converged,
                iterations=it, max_gradient=float(np.abs(g[free]).max()), info=info)
