"""
check_anchor_homogeneity()'s check of an MFRM anchor set on joint fits ('cp-full-rcl', 'cpat-rcl', 'cml'), using the
anchors' responses only. One fit gives every anchor its own location at every item x threshold cell ('matrix').

1. Overall agreement: all anchors on 'global' against all on 'items' and all on 'thresholds' (composite likelihood
   ratio with Kent's weights, or the conditional likelihood ratio for 'cml'), and a Wald test of equal global
   effects, Holm across the three. Reported only; the suggestions come from steps 2 and 3.
2. Global effect: an anchor's global effect is its mean location over the cells. Every subset of the anchors (up to
   12; with more, a multi-start search) is tested, largest first: a subset agrees when each member's global effect
   minus the mean of the other members' is within delta (close-fit test, corrected across the members). The largest
   agreeing subsets are suggested, ties ranked by the shift of the zero point from the mean of all the anchors given.
3. Pattern, within each suggested subset: per_rater_model_selection(scheme='one-pass') on the subset's responses
   only, the subset as the anchors, so each member is tested by the selection's steps with the other members on
   'global' as the reference, at the check's delta (converted to logits by the width of the fit anchored on the
   subset; delta_max twice delta, as the selection's default). An anchor selected off 'global' anchors through its
   global effect only, its pattern over the cells estimated freely (AnchorSet.pattern_free). An anchor with cells cut
   stays on 'global' in the selection (cuts need it), so it is tested instead against delta by a close-fit test of
   the root mean square of its departures over its cells not cut (corrected across such anchors). The size reported
   for every member: its location at each cell minus the mean of the other members' there, less the weighted mean of
   these over the cells (so a difference in global effect does not count), root mean square with each cell weighted
   by the inverse sampling variance of the anchor's location there (as per_rater_model_selection sizes a departure).

delta_units='width': the median category width of the joint fit of every facet element's responses (per_rater_model_
selection's starting fit: the anchors on 'global', the others on 'matrix'), anchored on the anchors given for step 2
and on the subset being tested for step 3, so the check and the selection measure departures in the same unit.

The suggestions are never applied by the check itself: the check object's cut / drop / restore / free_pattern /
hold_pattern / apply_suggested methods return a new check on the adjusted anchor set.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as fdist, nct, norm, t as tdist

from raschpy._anchors import AnchorSet, threshold_label


def corrected_reject(p, alpha, correction):
    """{name: rejected} for p-values {name: p}: correction None (each at alpha), 'holm' or 'bh'."""
    names = list(p)
    v = np.array([p[a] for a in names], dtype=float)
    v = np.where(np.isnan(v), 1.0, v)  # no test (NaN): not rejected
    m = len(v)
    rej = np.zeros(m, bool)
    order = np.argsort(v)
    if correction is None:
        rej = v <= alpha
    elif correction == "holm":
        for j, o in enumerate(order):
            if v[o] > alpha / (m - j):
                break
            rej[o] = True
    elif correction == "bh":
        ok = [j for j in range(m) if v[order[j]] <= (j + 1) * alpha / m]
        if ok:
            rej[order[: max(ok) + 1]] = True
    else:
        raise ValueError(f"correction must be 'bh', 'holm' or None, got {correction!r}.")
    return {a: bool(r) for a, r in zip(names, rej)}


class AnchorCheck:
    """
    The result of check_anchor_homogeneity() on joint fits. Tables:

    overall : pandas.DataFrame
        The overall agreement test of the anchors: for 'items', 'thresholds' (likelihood ratio) and 'global effect'
        (Wald), the statistic, its degrees of freedom and p; the overall p (Holm across the three) is overall_p.
    global_effects : pandas.DataFrame
        Each anchor given: its global effect minus the mean of the other anchors given (logits), SE, close-fit p at
        delta, whether flagged (corrected across the anchors), and whether it is in the suggested set.
    pattern : pandas.DataFrame
        Each anchor in the suggested set: the root mean square of its departures over the cells from the other
        members, each cell weighted by the inverse sampling variance of the anchor's location there (logits, and with
        sampling noise removed, 'RMS (bias-corrected)'), the same in tolerance units (pattern.attrs['width']: the
        median category width used, of the fit anchored on the suggested set), 'Representation' (what
        per_rater_model_selection's steps choose for it on the set's responses, the other members on 'global';
        'global (cells cut)' for an anchor with cells cut), and 'Pattern free' (selected off 'global': it anchors
        through its global effect only). With an anchor with cells cut, also 'p': that anchor's close-fit p at delta
        over its cells not cut (its pattern-free test).
    suggestions : pandas.DataFrame
        The suggested changes: anchor, action ('drop' or 'free pattern'), reason ('global effect' or 'pattern'), the
        shift (a dropped anchor's global effect minus the mean of the suggested set's) or RMS departure, SE, p, the
        representation selected (a freed pattern), and the pull on the zero point a dropped anchor would have had
        (shift / (set size + 1)).
    anchors, suggested : AnchorSet
        The anchor set checked, and the first suggested set (pass check.suggested, or check.apply_suggested().anchors,
        to calibrate(..., anchors=) or per_rater_model_selection(anchors=)).
    ties : list of AnchorSet
        The suggested set and the sets tied with it, ranked; pattern_ties: each one's pattern table;
        apply_suggested(choice=k).
    global_core : dict
        The global-effect search: sets (ranked), zero_shift (each set's mean global effect minus the mean of all the
        anchors given), near_tie, exhaustive, tol.
    notes : list of str
        What the check could not decide, and the warnings.
    profile, subsets : pandas.DataFrame or None
        With report='full': the suggestion at delta 0.1, 0.15, 0.2 and 0.3 widths (or 1, 1.5, 2 and 3 times delta in
        logits), and each subset tried by the global-effect search with each member's comparison.

    Methods cut(anchor, cell), drop(anchor), restore(anchor), free_pattern(anchor), hold_pattern(anchor) and apply_suggested(choice=0)
    return a new AnchorCheck on the adjusted set; this one is unchanged.
    """

    _DELTA_DEFAULT = 0.1  # median category widths, as per_rater_model_selection's delta

    def __init__(self, model, anchors, estimation, alpha=0.05, correction="bh", min_effect=0.0, min_anchors=2,
                 threshold_constant=None, previous=None, delta=None, delta_units="width", report="final"):
        from raschpy.mfrm import MFRM

        if not isinstance(anchors, AnchorSet):
            anchors = AnchorSet(list(anchors))
        facets = list(model.facet_names)
        unknown = [a for a in anchors.original + anchors.anchors if a not in facets]
        if unknown:
            raise ValueError(f"anchors: unknown {model.facet_label} {sorted(set(unknown))}.")
        if len(anchors.anchors) < 2:
            raise ValueError(f"anchors: at least 2 {model.facets_label} are needed to check their agreement.")
        if estimation not in ("cp-full-rcl", "cpat-rcl", "cml"):
            raise ValueError(
                f"check_anchor_homogeneity's joint-fit check needs estimation='cp-full-rcl', 'cpat-rcl' or 'cml', "
                f"got {estimation!r}; with 'pair-cpat' use rater_homogeneity_test='cochran'."
            )
        if delta_units not in ("width", "logits"):
            raise ValueError("delta_units must be 'width' or 'logits'.")
        if report not in ("final", "full"):
            raise ValueError(f"report must be 'final' or 'full', got {report!r}.")
        d = self._DELTA_DEFAULT if delta is None else delta
        if not np.isscalar(d) or d < 0:
            raise ValueError("delta must be a number >= 0.")
        self.model, self.anchors, self.estimation = model, anchors, estimation
        self.settings = dict(alpha=alpha, correction=correction, min_effect=float(min_effect),
                             min_anchors=int(min_anchors), threshold_constant=threshold_constant, delta=delta,
                             delta_units=delta_units, report=report)
        self._tol = {"delta": float(d)}
        self._tol_defaulted = {"delta"} if delta is None else set()
        self._unit = None
        self._widths_unusable = False
        self._panel, self._panel_ctx = {}, None
        self._sel = {}  # (subset, delta) -> per_rater_model_selection table (the pattern step)
        self.previous = previous
        self.items = list(model.responses.columns)
        self.K = int(model.max_score)
        # the anchors' responses only
        X = model.responses.loc[anchors.anchors]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._sub = MFRM(
                X, model.max_score, validate=False,
                facet=model.facet if model.facet_label != "facet element" else None,
                facet_plural=model.facets if model.facet_label != "facet element" else None,
            )
        # the sandwich's small-sample correction: check_anchor_homogeneity(se_correction=), set on the model for the call
        self._sub._se_correction_opt = getattr(model, "_se_correction_opt", "kauermann-carroll")
        self._names = list(self._sub.facet_names)
        # persons in the anchors' responses: the sandwich covariance sums one score per person, so the Wald tests
        # take Hotelling-type F / t references with persons - 1 df (a chi-squared reference is liberal with few)
        self.n_persons = int(self._sub.responses.index.get_level_values(1).nunique())
        self._check_connected()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ctx = self._sub._mixed_context(estimation, threshold_constant, {})
        self._fits = {}
        self._gfit = None
        self._grid = None
        self.notes = []
        self._warnings = []
        self._set_unit()
        self.overall = self._overall()
        self._run()
        self.profile = self.subsets = None
        if report == "full":
            self._full_report()
        self.notes += self._warnings
        if self._warnings:
            warnings.warn("check_anchor_homogeneity: " + " ".join(w[0].upper() + w[1:] for w in self._warnings),
                          UserWarning, stacklevel=3)

    # ---- Wald and close-fit references ------------------------------------------------------------------------
    def _wald_p(self, W, df):
        """p of a Wald statistic W on df, Hotelling-type: F = W (n - df + 1) / (df n) on (df, n - df + 1),
        n = persons - 1; NaN if n < df."""
        n = self.n_persons - 1
        if df < 1 or n - df + 1 < 1:
            return np.nan
        return float(fdist.sf(W * (n - df + 1) / (df * n), df, n - df + 1))

    def _z_p(self, z):
        """Two-sided p of a single-df Wald z, t reference on persons - 1 df."""
        return float(2 * tdist.sf(abs(z), max(self.n_persons - 1, 1)))

    def _close_p(self, shift, se, tol):
        """Close-fit p of a single contrast: P(|T| >= |shift| / se) when the true shift is tol (in delta_units), T
        noncentral t on persons - 1 df (the plain two-sided p when tol is 0)."""
        t = abs(shift) / se
        if tol <= 0:
            return self._z_p(t)
        nc = tol * self._unit / se
        with np.errstate(all="ignore"):
            p = float(nct.sf(t, max(self.n_persons - 1, 1), nc) + nct.cdf(-t, max(self.n_persons - 1, 1), nc))
        if not np.isfinite(p):  # scipy's noncentral t fails far in the tail; there df is large: the normal tail
            p = float(norm.sf(t - nc) + norm.cdf(-t - nc))
        return min(max(p, 0.0), 1.0)

    def _panel_widths(self, anchors):
        """The category widths (gaps between consecutive shared thresholds) of the joint fit of every facet element's
        responses anchored on anchors (on 'global', with their cut cells), the others on 'matrix':
        per_rater_model_selection's starting fit. Cached."""
        key = tuple(sorted(anchors))
        if key not in self._panel:
            m = self.model
            fac = list(m.facet_names)
            aidx = sorted(fac.index(a) for a in anchors)
            reps = ["global" if i in aidx else "matrix" for i in range(len(fac))]
            cuts = AnchorSet(list(anchors), {a: c for a, c in self.anchors.cuts.items() if a in anchors}).design_cuts(
                fac, self.items, self.K)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._panel_ctx is None:
                    self._panel_ctx = m._mixed_context(self.estimation, self.settings["threshold_constant"], {})
                design, fit, _ = m._mixed_fit_core(self._panel_ctx, reps, aidx, cuts or None)
                self._panel[key] = np.diff(m._mixed_quantities(design, fit["theta"])["thresholds"])
        return self._panel[key]

    def _pattern_unit(self, S):
        """Logits per unit of delta for the pattern test of subset S: the median category width of the joint fit
        anchored on S (the set a calibration would then use); the global-effect unit where that is not usable."""
        if self.settings["delta_units"] == "logits" or self._widths_unusable:
            return self._unit
        w = self._panel_widths(S)
        return float(np.median(w)) if len(w) and (w > 0).all() else self._unit

    def _set_unit(self):
        """Logits per unit of delta: 1 for 'logits'; for 'width', the median gap between consecutive shared thresholds
        of the joint fit of every facet element's responses with the anchors given on 'global' and the others on
        'matrix' (per_rater_model_selection's starting fit; also with delta=0, for the RMS (widths) column and the
        profile). The pattern test uses the same fit anchored on the subset tested (_pattern_unit). Unusable widths
        (maximum score 1, disordered thresholds): an error for a delta given, a warning and no tolerance for the
        default."""
        self._unit = 1.0
        if self.settings["delta_units"] == "logits":
            return
        widths = self._panel_widths(list(self.anchors.anchors)) if self.K >= 2 else np.array([])
        if len(widths) and (widths > 0).all():
            self._unit = float(np.median(widths))
            return
        why = ("the maximum score is 1, so there are no category widths" if not len(widths) else
               f"the anchors' thresholds are disordered (category widths {np.round(widths, 2).tolist()})")
        if not any(self._tol.values()):  # delta=0: no tolerance to measure, and no widths to report
            self._widths_unusable = True
            return
        given = [k for k in self._tol if k not in self._tol_defaulted and self._tol[k] > 0]
        if given:
            raise ValueError(f"check_anchor_homogeneity: delta_units='width', but {why}; give delta in logits "
                             "with delta_units='logits'.")
        warnings.warn(f"check_anchor_homogeneity: the default tolerance is in median category widths, but {why}, "
                      "so the check has no tolerance (plain tests). To use one, give delta in logits with "
                      "delta_units='logits'.", UserWarning, stacklevel=4)
        self._tol = {k: 0.0 for k in self._tol}
        self._widths_unusable = True


    # ---- fits ------------------------------------------------------------------------------------------------
    def _check_connected(self):
        """Each pair of anchors must share persons (their responses are compared through them)."""
        X = self._sub.responses
        rated = {a: set(X.loc[a].dropna(how="all").index) for a in self._names}
        thin = [(a, b) for i, a in enumerate(self._names) for b in self._names[i + 1:] if not rated[a] & rated[b]]
        if thin:
            self.notes.append(
                f"Anchor pairs with no persons in common: {thin}. They are compared only through the other anchors."
            )

    def _fit(self, reps, frame, aset, matrices=False):
        """The joint fit of the anchors' responses with representations reps {name: rep} (default 'global'), the
        anchors frame and the cut cells of aset (for anchors in frame). Cached."""
        models = [reps.get(n, "global") for n in self._names]
        cuts = AnchorSet(list(frame), {a: c for a, c in aset.cuts.items() if a in frame}).design_cuts(
            self._names, self.items, self.K)
        aidx = sorted(self._names.index(a) for a in frame)
        key = (tuple(models), tuple(aidx), tuple(sorted((r, tuple(c)) for r, c in cuts.items())), matrices)
        if key not in self._fits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # the check's tests are validated on the anchors' average frame (an anchor that moves off 'global'
                # is read against the average of the anchors, those held on 'global' included)
                design, fit, stored = self._sub._mixed_fit_core(self._ctx, models, aidx, cuts or None,
                                                                frame="average")
                crit = self._sub._mixed_criteria(self._ctx, design, fit,
                                                 matrices=matrices and self.estimation != "cml")
            self._fits[key] = (design, fit, stored, crit)
        return self._fits[key]

    def _covariance(self, design, fit, stored):
        """Sandwich covariance of the fit's coordinates (as calibrate(model='mixed')'s standard errors, with the
        Kauermann-Carroll correction when check_anchor_homogeneity(se_correction='kauermann-carroll'))."""
        from raschpy import _esf, _rcl

        R, I, K = design.R, design.I, design.K
        theta = fit["theta"]
        J, b = design.grid_jacobian(theta), design.grid(theta)
        kc = self._sub._kc_on()
        free = np.ones(len(theta), bool)
        free[list(stored.get("fixed", []))] = False
        ix = np.ix_(free, free)
        Binv = np.linalg.pinv(stored["info"][ix], hermitian=True)
        if self.estimation == "cml":
            Bfull = np.zeros((len(theta), len(theta)))
            Bfull[ix] = Binv
            _, meat = _esf.cml_sandwich(stored["X"], J, b.ravel(), [K] * (R * I), hessian=stored["hessian"],
                                        small_sample=kc, kc_scores=self._sub._kc_scores, bread_inv=Bfull)
        else:
            X3 = self._ctx["X3"]
            split = stored.get("split", [])
            empty = None if self.estimation == "cpat-rcl" else self._ctx["empty"]
            G = (_rcl.person_grid_scores(b, X3, K, split) if empty is None
                 else _rcl.full_person_grid_scores(b, X3, K, split, empty))
            U = G.reshape(G.shape[0], -1) @ J
            if kc:
                U = U.copy()
                U[:, free] = _rcl.kc_grid_scores(U[:, free], b, X3, K, J[:, free], Binv, split, empty)
            meat = U.T @ U
        V = np.zeros((len(theta), len(theta)))
        V[ix] = Binv @ meat[ix] @ Binv
        return V


    # ---- overall test ----------------------------------------------------------------------------------------
    def _overall(self):
        from scipy.linalg import null_space, orth
        from raschpy import _selection

        # anchors with cut cells stay on 'global' with their cuts in both fits (cuts need 'global'); the others move
        frame = list(self.anchors.anchors)
        aset = self.anchors
        free = [a for a in frame if a not in aset.cuts]
        small = self._fit({}, frame, aset, matrices=True)
        rows = {}
        for alt in ("items", "thresholds") if len(free) >= 2 or (free and len(frame) > len(free)) else ():
            large = self._fit({a: alt for a in free}, frame, aset, matrices=True)
            cS, cL = small[3], large[3]
            # nothing to test when the frame fixes the free anchors' effects (e.g. one free anchor)
            if self.estimation == "cml":
                stat = max(2.0 * (cL["ll"] - cS["ll"]), 0.0)
                df = int(cL["k"] - cS["k"])
                if df > 0:
                    rows[alt] = dict(Statistic=stat, df=df, p=float(chi2.sf(stat, df)))
            else:
                JL = cL["Jac"]
                Cb = orth(_selection.lstsq(JL, cS["Jac"]))
                B = np.hstack([Cb, null_space(Cb.T)])
                if B.shape[1] == Cb.shape[1]:
                    continue
                r = _selection.composite_lr_core(2.0 * (cL["ll"] - cS["ll"]), B.T @ cL["H"] @ B,
                                                 B.T @ cL["J"] @ B, Cb.shape[1])
                rows[alt] = dict(Statistic=r["statistic"], df=r["df"], p=r["p"])
        sev = self._severity(aset, {})
        rows["global effect"] = dict(Statistic=sev["wald"], df=sev["df"], p=sev["p"])
        t = pd.DataFrame(rows).T
        t.index.name = "Alternative"
        t["df"] = t["df"].astype(int)
        # Holm across the three: the overall decision is its first step
        self.overall_p = float(min(1.0, len(t) * t["p"].min()))
        return t

    def _severity(self, w, extra):
        """The anchors' global effects in the fit with every facet element on 'global' (extra: the representations of
        facet elements in the data but not in w), w's anchors as the frame: for each anchor, its global effect minus the
        mean of the other anchors' (shift, se, z, p, pull = shift / number of anchors), and the joint Wald test
        that all are equal (wald, df, p)."""
        frame = list(w.anchors)
        design, fit, stored, _ = self._fit(dict(extra), frame, w)
        V = self._covariance(design, fit, stored)
        th = fit["theta"]
        oL = design.off[2]
        L = {a: design.M[oL + self._names.index(a)] for a in frame}
        out = {}
        for a in frame:
            c = L[a] - np.mean([L[b] for b in frame if b != a], axis=0)
            shift, se = float(c @ th), float(np.sqrt(max(c @ V @ c, 1e-300)))
            out[a] = dict(shift=shift, se=se, z=shift / se, p=self._close_p(shift, se, self._tol["delta"]),
                          pull=shift / len(frame))
        C = np.array([L[a] - L[frame[0]] for a in frame[1:]])
        est, Vc = C @ th, C @ V @ C.T
        ev, Q = np.linalg.eigh((Vc + Vc.T) / 2)
        keep = ev > 1e-10 * max(ev.max(), 1e-300)
        z = Q[:, keep].T @ est
        W = float((z ** 2 / ev[keep]).sum())
        return dict(anchors=out, wald=W, df=int(keep.sum()), p=self._wald_p(W, int(keep.sum())))


    # ---- global effect ---------------------------------------------------------------------------------------
    def _global_fit(self):
        """The anchors' global effects with every anchor on 'matrix' (each keeps its own pattern over items and
        thresholds), no cuts: (names, global effects, their covariance, responses per anchor). A contrast of global
        effects does not depend on which anchors define the zero point, so one fit serves every subset."""
        if self._gfit is None:
            A = list(self.anchors.anchors)
            design, fit, stored, _ = self._fit({a: "matrix" for a in A}, A, AnchorSet(A))
            V = self._covariance(design, fit, stored)
            oL = design.off[2]
            L = np.array([design.M[oL + self._names.index(a)] for a in A])
            X = self._sub.responses
            marks = np.array([int(X.loc[a].notna().sum().sum()) for a in A])
            self._gfit = (A, L @ fit["theta"], L @ V @ L.T, marks)
        return self._gfit

    def _subset_test(self, S, tol):
        """Whether the anchors S (indices into _global_fit's names) agree in global effect: each one's global effect
        minus the mean of the others', close-fit test at tol, corrected across S, flagged with |shift| >= min_effect.
        (agrees, {index: (shift, se, p, flagged)})."""
        _, ge, Vg, _ = self._global_fit()
        st = self.settings
        out = {}
        for i in S:
            c = np.zeros(len(ge))
            c[list(S)] = -1.0 / (len(S) - 1)
            c[i] = 1.0
            shift, se = float(c @ ge), float(np.sqrt(max(c @ Vg @ c, 1e-300)))
            out[i] = [shift, se, self._close_p(shift, se, tol)]
        rej = corrected_reject({i: v[2] for i, v in out.items()}, st["alpha"], st["correction"])
        for i in out:
            out[i].append(bool(rej[i] and abs(out[i][0]) >= st["min_effect"]))
        return not any(v[3] for v in out.values()), out

    def _global_search(self, tol):
        """The largest sets of at least min_anchors anchors agreeing in global effect only, at tol: every subset,
        largest first, with up to 12 anchors (agreement is not monotone: dropping an anchor moves the others' mean);
        with more, dropping the worst anchor in turn from the full set and from each set without one anchor. Ties
        ranked by the shift of the zero point from the average of all the anchors given, then the range of their
        global effects, then their number of responses (most first)."""
        A, ge, Vg, marks = self._global_fit()
        n, m0 = len(A), max(self.settings["min_anchors"], 2)
        tested, best = {}, []

        def test(S):
            if S not in tested:
                tested[S] = self._subset_test(S, tol)
            return tested[S][0]

        exhaustive = n <= 12
        if exhaustive:
            for k in range(n, m0 - 1, -1):
                best = [S for S in itertools.combinations(range(n), k) if test(S)]
                if best:
                    break
        else:
            found = set()
            for start in [tuple(range(n))] + [tuple(j for j in range(n) if j != i) for i in range(n)]:
                S = start
                while len(S) >= m0 and not test(S):
                    res = tested[S][1]
                    worst = min((i for i in S if res[i][3]), key=lambda i: (res[i][2], -abs(res[i][0])))
                    S = tuple(j for j in S if j != worst)
                if len(S) >= m0 and test(S):
                    found.add(S)
            k = max((len(S) for S in found), default=0)
            best = sorted(S for S in found if len(S) == k)
        wall = np.full(n, 1.0 / n)

        def w(S):
            v = np.zeros(n)
            v[list(S)] = 1.0 / len(S)
            return v

        zero = {S: float((w(S) - wall) @ ge) for S in best}
        best.sort(key=lambda S: (round(abs(zero[S]), 12), float(np.ptp(ge[list(S)])), -int(marks[list(S)].sum())))
        near = False
        if len(best) >= 2:
            S1, S2 = best[:2]
            c = np.sign(zero[S1]) * (w(S1) - wall) - np.sign(zero[S2]) * (w(S2) - wall)
            near = abs(abs(zero[S1]) - abs(zero[S2])) < float(np.sqrt(max(c @ Vg @ c, 1e-300)))
        return dict(sets=[[A[i] for i in S] for S in best], zero_shift=[zero[S] for S in best], near_tie=near,
                    exhaustive=exhaustive, tol=tol,
                    tested={tuple(A[i] for i in S): (ok, {A[i]: v for i, v in res.items()})
                            for S, (ok, res) in tested.items()})

    # ---- pattern ------------------------------------------------------------------------------------------------
    def _grid_fit(self):
        """The anchors' locations at every cell (R x I*K) in the fit with every anchor on 'matrix' (_global_fit's), and
        their covariance."""
        if self._grid is None:
            A = list(self.anchors.anchors)
            design, fit, stored, _ = self._fit({a: "matrix" for a in A}, A, AnchorSet(A))
            th = fit["theta"]
            b = np.asarray(design.grid(th), float).reshape(design.R, -1)
            J = design.grid_jacobian(th)
            self._grid = (b, J @ self._covariance(design, fit, stored) @ J.T)
        return self._grid

    def _held(self, a):
        """Cells (flattened item x threshold) where anchor a is not cut."""
        held = np.ones((len(self.items), self.K), bool)
        for kind, lab in self.anchors.cuts.get(a, []):
            held[self._cells(kind, lab)] = False
        return held.ravel()

    def _cells(self, kind, lab):
        """Index into an (I, K) array for an item, threshold or (item, threshold) label."""
        th = lambda t: int(str(t).split()[-1]) - 1
        if kind == "item":
            return np.s_[self.items.index(lab), :]
        if kind == "threshold":
            return np.s_[:, th(lab)]
        return np.s_[self.items.index(lab[0]), th(lab[1])]

    def _pattern_test(self, S, tol):
        """Each anchor in S against the other members: its location at each cell minus their mean there (cells where
        it and at least one other member are not cut), centred on their weighted mean over the cells, each cell
        weighted by the inverse sampling variance of the anchor's location there (weights summing to 1, as
        per_rater_model_selection's departure size); Q = the weighted sum of squares, RMS = sqrt(Q). Close-fit p: Q's
        distribution (a weighted sum of noncentral chi-squares, weights the eigenvalues of the weighted departures'
        covariance) when the true departures have weighted RMS tol (logits) in the observed direction.
        {anchor: (RMS, bias-corrected RMS, p)}."""
        from raschpy._selection import nc_weighted_chi2_sf

        b, Vb = self._grid_fit()
        R, n = b.shape
        held = {a: self._held(a) for a in S}
        vdiag = np.diag(Vb)
        out = {}
        for a in S:
            ia, oth = self._names.index(a), [o for o in S if o != a]
            cells = [c for c in range(n) if held[a][c] and any(held[o][c] for o in oth)]
            m = len(cells)
            if m < 2:
                out[a] = (np.nan, np.nan, np.nan)
                continue
            C = np.zeros((m, R * n))
            for j, c in enumerate(cells):
                C[j, ia * n + c] += 1.0
                ref = [o for o in oth if held[o][c]]
                for o in ref:
                    C[j, self._names.index(o) * n + c] -= 1.0 / len(ref)
            v = vdiag[[ia * n + c for c in cells]]
            wt = np.where(v > 0, 1.0 / np.where(v > 0, v, 1.0), 0.0)
            wt = wt / wt.sum() if wt.sum() > 0 else np.full(m, 1.0 / m)
            C = (np.eye(m) - np.outer(np.ones(m), wt)) @ C  # the weighted mean departure (global effect) removed
            sw = np.sqrt(wt)
            z = sw * (C @ b.ravel())
            Sz = (sw[:, None] * (C @ Vb @ C.T)) * sw[None, :]
            lam, U = np.linalg.eigh((Sz + Sz.T) / 2)
            keep = lam > max(lam.max(), 1e-300) * 1e-9
            lam, U = lam[keep], U[:, keep]
            Q = float(z @ z)
            mu = z / max(np.linalg.norm(z), 1e-300) * tol
            p = nc_weighted_chi2_sf(Q, lam, (U.T @ mu) ** 2 / lam)
            out[a] = (float(np.sqrt(Q)), float(np.sqrt(max(Q - lam.sum(), 0.0))), float(p))
        return out

    def _selection(self, S, d):
        """per_rater_model_selection(scheme='one-pass') on the responses of subset S only, S as the anchors (with
        their cut cells): each member tested by the selection's steps against the other members on 'global'.
        Tolerances in logits from the width of the fit anchored on S: delta = d x width, delta_max = 2 d x width (the
        selection's default ratio), stretch_delta the selection's default (0 with d = 0); alpha and min_effect as the
        check's. Cached. Returns the selection table."""
        key = (tuple(S), float(d))
        if key not in self._sel:
            from raschpy.mfrm import MFRM

            st, m = self.settings, self.model
            unit = self._pattern_unit(S)
            tol = (dict(delta=d * unit, delta_max=2 * d * unit, delta_units="logits") if d > 0
                   else dict(delta=0.0, delta_max=0.0, stretch_delta=0.0))
            aset = AnchorSet(list(S), {a: c for a, c in self.anchors.cuts.items() if a in S})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sub = MFRM(m.responses.loc[list(S)], m.max_score, validate=False,
                           facet=m.facet if m.facet_label != "facet element" else None,
                           facet_plural=m.facets if m.facet_label != "facet element" else None)
                sub._se_correction_opt = getattr(m, "_se_correction_opt", "kauermann-carroll")
                self._sel[key] = sub.per_rater_model_selection(
                    anchors=aset, scheme="one-pass", estimation=self.estimation,
                    threshold_constant=st["threshold_constant"], alpha=st["alpha"], min_effect=st["min_effect"],
                    se_correction=sub._se_correction_opt, progress=False, **tol)
        return self._sel[key]

    def _pattern_table(self, S, d):
        """The pattern step for subset S at tolerance d (in delta_units). The RMS columns size each member's departure
        (information-weighted, _pattern_test) in the unit of the fit anchored on S. 'Representation': what
        per_rater_model_selection's steps choose for the member on S's responses (_selection); pattern free when
        that is not 'global'. A member with cells cut stays on 'global' in the selection (cuts need it), so it is
        tested instead by the close-fit RMS test over its cells not cut ('p', corrected across such members)."""
        st = self.settings
        unit = self._pattern_unit(S)
        res = self._pattern_test(S, d * unit)
        t = pd.DataFrame({a: {"RMS": r[0], "RMS (bias-corrected)": r[1]} for a, r in res.items()}).T.astype(float)
        if st["delta_units"] == "width" and not self._widths_unusable:
            t["RMS (widths)"] = t["RMS (bias-corrected)"] / unit
        cut = [a for a in S if a in self.anchors.cuts]
        sel = self._selection(S, d)
        t["Representation"] = [("global (cells cut)" if a in cut else str(sel.loc[a, "Representation"]))
                               for a in t.index]
        free = {a: t.loc[a, "Representation"] != "global" for a in t.index}
        if cut:
            pc = {a: res[a][2] for a in cut}
            rej = corrected_reject(pc, st["alpha"], st["correction"])
            t["p"] = [pc.get(a, np.nan) for a in t.index]
            for a in cut:
                free[a] = bool(rej[a] and res[a][1] >= st["min_effect"])
        t["Pattern free"] = [bool(free[a]) for a in t.index]
        t.attrs["width"] = unit
        t.index.name = self.model.facet
        return t

    # ---- the check ------------------------------------------------------------------------------------------
    def _run(self):
        st = self.settings
        A = list(self.anchors.anchors)
        f = self.model.facet
        gc = self._global_search(self._tol["delta"])
        self.global_core = gc
        _, ge, Vg, _ = self._global_fit()
        full = self._subset_test(tuple(range(len(A))), self._tol["delta"])[1]
        first = gc["sets"][0] if gc["sets"] else A
        self.global_effects = pd.DataFrame(
            {A[i]: {"Shift": v[0], "SE": v[1], "p": v[2], "Flagged": v[3], "In suggested set": A[i] in first}
             for i, v in full.items()}).T
        self.global_effects.index.name = f
        tolstr = (f"delta={self._tol['delta']:g} {'width' if st['delta_units'] == 'width' else 'logits'}"
                  if self._tol["delta"] > 0 else "no tolerance")
        sets = gc["sets"] or [A]
        if not gc["sets"]:
            self._warnings.append(
                f"no set of at least min_anchors={max(st['min_anchors'], 2)} anchors agrees in global effect "
                f"({tolstr})" + (": with two anchors the data can't say which one is off" if len(A) == 2 else "")
                + ". Choose a different anchor set; the pattern table is for the anchors as given.")
        self.ties, self.pattern_ties, rows = [], [], []
        for k, S in enumerate(sets):
            idx = [A.index(a) for a in S]
            aset = self.anchors
            for a in A:
                if a in S:
                    continue
                c = np.zeros(len(A))
                c[idx] = -1.0 / len(idx)
                c[A.index(a)] = 1.0
                shift, se = float(c @ ge), float(np.sqrt(max(c @ Vg @ c, 1e-300)))
                p = self._close_p(shift, se, self._tol["delta"])
                aset = aset.drop(a, source="suggested", reason="global effect", shift=shift, se=se, p=p)
                if k == 0:
                    rows.append({f: a, "Action": "drop", "Reason": "global effect", "Shift": shift, "SE": se,
                                 "p": p, "Pull on zero point": shift / (len(S) + 1)})
            pt = self._pattern_table(S, self._tol["delta"])
            for a in S:
                if pt.loc[a, "Pattern free"] and a not in aset.pattern_free:
                    ev = dict(rms=float(pt.loc[a, "RMS (bias-corrected)"]),
                              representation=pt.loc[a, "Representation"])
                    if "p" in pt and pd.notna(pt.loc[a, "p"]):
                        ev["p"] = float(pt.loc[a, "p"])
                    aset = aset.free_pattern(a, source="suggested", reason="pattern", **ev)
                    if k == 0:
                        rows.append({f: a, "Action": "free pattern", "Reason": "pattern", "RMS": ev["rms"],
                                     "p": ev.get("p", np.nan), "Representation": ev["representation"]})
            self.ties.append(aset)
            self.pattern_ties.append(pt)
        self.suggested, self.pattern = self.ties[0], self.pattern_ties[0]
        cols = [f, "Action", "Reason", "Shift", "SE", "RMS", "p", "Representation", "Pull on zero point"]
        self.suggestions = pd.DataFrame(rows, columns=cols).dropna(axis=1, how="all")
        free = list(self.suggested.pattern_free)
        S = first
        if len(sets) > 1:
            msg = (f"{len(sets)} sets tie as the largest agreeing in global effect ({tolstr}), ranked by the shift of "
                   "the zero point from the average of all the anchors given, then the range of their global "
                   "effects, then their number of responses: " + " or ".join("{" + ", ".join(s) + "}" for s in sets)
                   + "; apply_suggested(choice=k) applies the k-th (0 the first).")
            if gc["near_tie"]:
                msg += (" The first two are a near tie (their zero-point shifts differ by less than one standard "
                        "error): choose between them on substantive grounds.")
            self._warnings.append(msg)
        if free and len(free) == len(S):
            self.notes.append(
                "every anchor in the suggested set differs from the others in pattern: the zero point rests on their "
                "global effects alone, and the instrument's pattern on their average"
                + (" (with two anchors the data can't say which one is off)" if len(S) == 2 else "") + ".")
        if 2 * len(S) < len(A) and gc["sets"]:
            self._warnings.append(
                f"the suggested set keeps {len(S)} of the {len(A)} anchors given: check that it is a suitable "
                "reference for the scale.")

    # ---- report='full' ---------------------------------------------------------------------------------------
    def _full_report(self):
        """profile (the suggestion at four tolerances) and subsets (each subset tried by the global-effect search)."""
        st = self.settings
        f = self.model.facet
        rows = []
        for S, (ok, res) in self.global_core["tested"].items():
            for a, (shift, se, p, flag) in res.items():
                rows.append({"Subset": S, "Size": len(S), "Agrees": ok, f: a, "Shift": shift, "SE": se, "p": p,
                             "Flagged": flag})
        self.subsets = pd.DataFrame(rows, columns=["Subset", "Size", "Agrees", f, "Shift", "SE", "p", "Flagged"])
        if self._widths_unusable:  # no category widths to measure tolerances in (a warning said so)
            return
        d0 = self._tol["delta"] or self._DELTA_DEFAULT
        levels = ((0.1, 0.15, 0.2, 0.3) if st["delta_units"] == "width"
                  else tuple(round(d0 * x, 6) for x in (1, 1.5, 2, 3)))
        prof = []
        for d in levels:
            gc = self._global_search(d)
            S = gc["sets"][0] if gc["sets"] else list(self.anchors.anchors)
            pt = self._pattern_table(S, d)
            prof.append({"delta": d, "Anchors": list(S), "Pattern free": list(pt.index[pt["Pattern free"]]),
                         "Tied sets": len(gc["sets"]), "Agrees in global effect": bool(gc["sets"])})
        self.profile = pd.DataFrame(prof).set_index("delta")

    # ---- actions ---------------------------------------------------------------------------------------------
    def _new(self, aset):
        st = {k: v for k, v in self.settings.items()}
        chk = AnchorCheck(self.model, aset, self.estimation, previous=self, **st)
        self.model.anchor_check = chk
        return chk

    def cut(self, anchor, cell):
        """A new check with anchor no longer an anchor at cell (an item name, 'Threshold k' in any case, or an
        (item, threshold) pair): its departure there is left out of the pattern test."""
        return self._new(self.anchors.cut(anchor, cell, self.items, self.K))

    def drop(self, anchor):
        """A new check without anchor in the anchor set (it stays in the data as an ordinary facet element)."""
        return self._new(self.anchors.drop(anchor))

    def restore(self, anchor):
        """A new check with a dropped anchor put back."""
        return self._new(self.anchors.restore(anchor))

    def free_pattern(self, anchor):
        """A new check with anchor anchoring through its global effect only (its pattern estimated freely)."""
        return self._new(self.anchors.free_pattern(anchor))

    def hold_pattern(self, anchor):
        """A new check with anchor's pattern held with the other anchors' again."""
        return self._new(self.anchors.hold_pattern(anchor))

    def apply_suggested(self, choice=0):
        """A new check on the suggested anchor set; choice=k: the k-th of tied sets (ties, ranked; 0 the first)."""
        if not 0 <= int(choice) < len(self.ties):
            raise ValueError(f"choice must be 0 to {len(self.ties) - 1}: {len(self.ties)} suggested set(s).")
        return self._new(self.ties[int(choice)])

    # ---- report ----------------------------------------------------------------------------------------------
    def __repr__(self):
        fmt_p = lambda p: "      " if pd.isna(p) else ("< 0.001" if p < 0.001 else f"= {p:.3f}")
        st = self.settings
        corr = {"bh": "BH", "holm": "Holm", None: "no correction"}[st["correction"]]
        out = [f"Anchor check ({self.estimation}, the anchors' responses only)", f"  {self.anchors!r}"]
        if self.previous is not None:
            for e in self.anchors.log[len(self.previous.anchors.log):]:
                out.append(f"  changed: {e['action']} {e['anchor']}" + (f" at {e['cell']}" if e.get("cell") else ""))
        out += ["", "Overall agreement"]
        for alt, r in self.overall.iterrows():
            out.append(f"  {alt:<14} {'Wald' if alt == 'global effect' else 'LR'} p {fmt_p(r['p'])}")
        out.append(f"  overall p {fmt_p(self.overall_p)} (Holm)")
        tol = f"delta {self._tol['delta']:g} {'widths' if st['delta_units'] == 'width' else 'logits'}"
        wg = (f"; widths of the fit of all {self.model.facets_label} anchored on the anchors given: {self._unit:.3f} logits"
              if st["delta_units"] == "width" and not self._widths_unusable else "")
        out += ["", f"Global effect (each anchor minus the mean of the others, logits; close-fit at {tol}, {corr}{wg})"]
        for a, r in self.global_effects.iterrows():
            out.append(f"  {a:<12} shift {r['Shift']:+.3f} (SE {r['SE']:.3f})  p {fmt_p(r['p']):<8}"
                       + ("  flagged" if r["Flagged"] else ""))
        gc = self.global_core
        if gc["sets"]:
            out.append("Largest set(s) agreeing in global effect" + (f", {len(gc['sets'])} tied:" if len(gc["sets"]) > 1
                                                                     else ":"))
            for k, (S, z) in enumerate(zip(gc["sets"], gc["zero_shift"])):
                out.append(f"  {k}: {{{', '.join(S)}}}  zero point shift {z:+.3f}")
        else:
            out.append("No set of the anchors agrees in global effect.")
        wcol = "RMS (widths)" in self.pattern
        wtxt = (f"; widths of the fit anchored on the set: {self.pattern.attrs['width']:.3f} logits"
                if wcol and "width" in self.pattern.attrs else "")
        cuttxt = (f"; an anchor with cells cut: close-fit RMS test over its other cells, {corr}"
                  if "p" in self.pattern else "")
        out += ["", f"Pattern within set 0 (per_rater_model_selection's steps on the set's responses, the other "
                    f"members on 'global', at {tol}{cuttxt}; size: information-weighted RMS departure over the cells "
                    f"from the other members{wtxt})"]
        for a, r in self.pattern.iterrows():
            out.append(f"  {a:<12} RMS {r['RMS (bias-corrected)']:.3f} logits"
                       + (f" ({r['RMS (widths)']:.3f} widths)" if wcol else "")
                       + f"  {r['Representation']:<18}"
                       + (f"  p {fmt_p(r['p']):<8}" if "p" in self.pattern and pd.notna(r["p"]) else "")
                       + ("  pattern free" if r["Pattern free"] else ""))
        out.append("")
        if len(self.suggestions):
            out.append("Suggested adjustments")
            disp = self.suggestions.copy()
            for c in disp:
                if c in ("Shift", "SE", "RMS", "p", "Pull on zero point"):
                    disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
                elif c == "Representation":
                    disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else x)
            out.append(disp.to_string(index=False))
        else:
            out.append("No adjustments suggested.")
        for n in self.notes:
            out.append(f"Note: {n}")
        return "\n".join(out)
