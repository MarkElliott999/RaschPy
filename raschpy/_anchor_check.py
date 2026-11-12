"""
check_anchor_homogeneity()'s check of an MFRM anchor set on joint fits ('cp-full-rcl', 'cpat-rcl', 'cml'), using the
anchors' responses only (ANCHOR_AGREEMENT_DESIGN.md):

1. an overall test of the anchors' agreement: all anchors on 'global' against all on 'items' and all on 'thresholds'
   (composite likelihood ratio with Kent's weights, or the conditional likelihood ratio for 'cml'), and a Wald test
   of equal global effects, Holm across the three;
2. a screen of each anchor against the rest: its representation chosen by per-rater selection's AIC rule with the
   zero point at the other anchors' average, the gate p-values corrected across the anchors (Benjamini-Hochberg by
   default); the worst flagged anchor is resolved first and the rest screened again;
3. for the anchor being resolved, a Wald loop over the cells of its representation: the cell whose contrast with the
   mean of the anchor's remaining cells has the largest |z| (significant, with |shift| >= min_effect) is cut, until the
   remaining cells are equal; more than max_share of a family's cells cut means drop the anchor instead;
4. a global effect screen on the adjusted set (each anchor's global effect against the mean of the rest's, Wald; a
   flagged anchor is suggested for dropping), worst first;
5. refinement: each resolved anchor is re-estimated against the final set of anchors (pattern, then global effect).
With split_alpha the screens flag anchors at alpha / 2.

The suggestions are never applied by the check itself: the check object's cut / drop / restore / apply_suggested
methods return a new check on the adjusted anchor set.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as fdist, t as tdist

from raschpy._anchors import AnchorSet, threshold_label

CANDIDATES = ("global", "items", "thresholds", "bivector", "matrix")  # simplest first: ties go to the simpler


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
        The overall agreement test of the anchors (an anchor with cut cells held on 'global' with its cuts): for
        'items',
        'thresholds' (likelihood ratio) and 'global effect' (Wald), the statistic, its degrees of freedom and p; the
        overall p (Holm across the three) is overall_p.
    screen : list of (str, pandas.DataFrame)
        One table per screening round, first 'pattern' rounds (each anchor's representation against the rest, the
        AIC gain over 'global' scaled as per-rater selection's gate, its p), then 'global effect' rounds (each anchor's
        global effect minus the mean of the rest's, its SE and p), with whether it is flagged after the correction
        across anchors and which anchor was resolved.
    suggestions : pandas.DataFrame
        The suggested changes, after refinement: anchor, action ('cut', 'drop', 'restore' or 'keep cut' for a cut
        already in the set), reason ('pattern' or 'global effect'), cell, the shift (logits: the cell's contrast with
        the mean of the anchor's remaining cells, or the anchor's global effect minus the mean of the rest's), its
        standard error, z, p, and the pull on the zero point if it were kept (shift / number of anchors). A
        pattern drop lists the cells behind it.
    anchors, suggested : AnchorSet
        The anchor set checked, and that set with the suggestions applied (pass check.suggested, or
        check.apply_suggested().anchors, to calibrate(..., anchors=)).
    notes : list of str
        What the check could not decide (e.g. two anchors left that disagree). If the suggestions would drop more
        than half the anchors, the anchors have no majority core: nothing is suggested (suggestions is empty,
        suggested is the set checked) and a note says so; screen still shows each round.

    Methods cut(anchor, cell), drop(anchor), restore(anchor, cell=None) and apply_suggested() return a new
    AnchorCheck on the adjusted set; this one is unchanged.
    """

    def __init__(self, model, anchors, estimation, alpha=0.05, correction="bh", min_effect=0.0, max_share=0.5,
                 min_anchors=2, allow_cuts=True, threshold_constant=None, split_alpha=True, previous=None):
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
        self.model, self.anchors, self.estimation = model, anchors, estimation
        self.settings = dict(alpha=alpha, correction=correction, min_effect=float(min_effect), max_share=max_share,
                             min_anchors=int(min_anchors), allow_cuts=allow_cuts,
                             threshold_constant=threshold_constant, split_alpha=bool(split_alpha))
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
        self._names = list(self._sub.facet_names)
        # persons in the anchors' responses: the sandwich covariance sums one score per person, so the Wald tests
        # take Hotelling-type F / t references with persons - 1 df (a chi-squared reference is liberal with few)
        self.n_persons = int(self._sub.responses.index.get_level_values(1).nunique())
        self._check_connected()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ctx = self._sub._mixed_context(estimation, threshold_constant, {})
        self._fits = {}
        self.notes = []
        self.overall = self._overall()
        self.screen, self.suggestions, self.suggested = self._run()

    # ---- Wald references -------------------------------------------------------------------------------------
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
                design, fit, stored = self._sub._mixed_fit_core(self._ctx, models, aidx, cuts or None)
                crit = self._sub._mixed_criteria(self._ctx, design, fit,
                                                 matrices=matrices and self.estimation != "cml")
            self._fits[key] = (design, fit, stored, crit)
        return self._fits[key]

    def _covariance(self, design, fit, stored):
        """Sandwich covariance of the fit's coordinates (as calibrate(model='mixed')'s standard errors)."""
        from raschpy import _esf, _rcl

        R, I, K = design.R, design.I, design.K
        theta = fit["theta"]
        J, b = design.grid_jacobian(theta), design.grid(theta)
        if self.estimation == "cml":
            _, meat = _esf.cml_sandwich(stored["X"], J, b.ravel(), [K] * (R * I), hessian=stored["hessian"])
        else:
            X3 = self._ctx["X3"]
            split = stored.get("split", [])
            G = (_rcl.person_grid_scores(b, X3, K, split) if self.estimation == "cpat-rcl"
                 else _rcl.full_person_grid_scores(b, X3, K, split, self._ctx["empty"]))
            U = G.reshape(G.shape[0], -1) @ J
            meat = U.T @ U
        free = np.ones(len(theta), bool)
        free[list(stored.get("fixed", []))] = False
        ix = np.ix_(free, free)
        Binv = np.linalg.pinv(stored["info"][ix], hermitian=True)
        V = np.zeros((len(theta), len(theta)))
        V[ix] = Binv @ meat[ix] @ Binv
        return V

    # ---- 1. overall test -------------------------------------------------------------------------------------
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
                Cb = orth(np.linalg.lstsq(JL, cS["Jac"], rcond=None)[0])
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
            out[a] = dict(shift=shift, se=se, z=shift / se, p=self._z_p(shift / se),
                          pull=shift / len(frame))
        C = np.array([L[a] - L[frame[0]] for a in frame[1:]])
        est, Vc = C @ th, C @ V @ C.T
        ev, Q = np.linalg.eigh((Vc + Vc.T) / 2)
        keep = ev > 1e-10 * max(ev.max(), 1e-300)
        z = Q[:, keep].T @ est
        W = float((z ** 2 / ev[keep]).sum())
        return dict(anchors=out, wald=W, df=int(keep.sum()), p=self._wald_p(W, int(keep.sum())))

    # ---- 2. screen -------------------------------------------------------------------------------------------
    def _coverage(self, w, frame):
        """(I, K) number of the anchors in frame not cut at each item x threshold cell."""
        I, K = len(self.items), self.K
        cov = np.zeros((I, K), int)
        for a in frame:
            held = np.ones((I, K), bool)
            for kind, lab in w.cuts.get(a, []):
                held[self._cells(kind, lab)] = False
            cov += held
        return cov

    def _cells(self, kind, lab):
        """Index into an (I, K) array for an item, threshold or (item, threshold) label."""
        th = lambda t: int(str(t).split()[-1]) - 1
        if kind == "item":
            return np.s_[self.items.index(lab), :]
        if kind == "threshold":
            return np.s_[:, th(lab)]
        return np.s_[self.items.index(lab[0]), th(lab[1])]

    def _sole_cells(self, a, w):
        """Cells where every anchor but a is cut (a alone defines the zero point there), as labels."""
        frame = [x for x in w.anchors if x != a]
        bare = np.argwhere(self._coverage(w, frame) == 0)
        return [(self.items[i], threshold_label(k + 1)) for i, k in bare]

    def _select(self, a, w, extra):
        """Anchor a's representation against the rest of w (a out of the frame): (rep, scaled AIC gain, p)."""
        sole = self._sole_cells(a, w)
        if sole:
            note = (f"{a}: the only anchor not cut at {len(sole)} cell(s) (e.g. {sole[0]}), so it can't be compared "
                    "with the other anchors there; not screened.")
            if note not in self.notes:
                self.notes.append(note)
            return "global", 0.0, 1.0
        frame = [x for x in w.anchors if x != a]
        crit = {}
        for rep in CANDIDATES:
            reps = dict(extra)
            reps[a] = rep
            _, fit, _, c = self._fit(reps, frame, w)
            crit[rep] = (2 * c["pen"] - 2 * c["ll"], c["pen"], c["k"])
        best = min(CANDIDATES, key=lambda r: (round(crit[r][0], 9), CANDIDATES.index(r)))
        if best == "global":
            return "global", 0.0, 1.0
        delta = crit["global"][0] - crit[best][0]
        if self.estimation != "cml":
            cw = (crit[best][1] - crit["global"][1]) / max(crit[best][2] - crit["global"][2], 1)
            delta = delta / cw if cw > 0 else delta
        p = float(np.exp(-abs(delta) / 2)) if delta > 0 else 1.0
        return best, float(delta), p

    # ---- 3. Wald loop ----------------------------------------------------------------------------------------
    def _families(self, design, r, rep):
        """{family: (labels, rows)}: anchor r's cell effects as linear maps of the fit's coordinates."""
        oD, oT, oL, oA, oTr, oE, oI, oW = design.off
        M = design.M
        I, K = design.I, design.K
        thr = [threshold_label(k) for k in range(1, K + 1)]
        fam = {}
        if rep in ("items", "bivector"):
            fam["item"] = (list(self.items), M[oA + r * I: oA + (r + 1) * I])
        if rep in ("thresholds", "bivector"):
            fam["threshold"] = (thr, M[oTr + r * K: oTr + (r + 1) * K])
        if rep == "matrix":
            rows = [M[oA + r * I + i] + M[oTr + r * K + k] + M[oE + (r * I + i) * K + k]
                    for i in range(I) for k in range(K)]
            fam["cell"] = ([(it, t) for it in self.items for t in thr], np.vstack(rows))
        return fam

    def _cell_loop(self, a, rep, w, extra):
        """The Wald loop for anchor a on rep (a out of the frame, the rest of w defining the zero point)."""
        st = self.settings
        frame = [x for x in w.anchors if x != a]
        reps = dict(extra)
        reps[a] = rep
        design, fit, stored, _ = self._fit(reps, frame, w)
        V = self._covariance(design, fit, stored)
        th = fit["theta"]
        fam = self._families(design, self._names.index(a), rep)
        left = {f: list(range(len(v[0]))) for f, v in fam.items()}
        # a cell is cut only while at least two other anchors still define the zero point there, so that screening
        # any one of them later leaves one
        cov = self._coverage(w, frame)
        cuttable = lambda f, j: cov[self._cells(f, fam[f][0][j])].min() >= 2
        held_back = []
        cuts, p_joint = [], np.nan
        m_at_cell = len(frame) + 1
        while True:
            rows = [L[j] - L[idx[0]] for f, (lab, L) in fam.items() for idx in [left[f]] for j in idx[1:]]
            if not rows:
                break
            C = np.array(rows)
            est, Vc = C @ th, C @ V @ C.T
            w_, Q = np.linalg.eigh((Vc + Vc.T) / 2)
            keep = w_ > 1e-10 * max(w_.max(), 1e-300)
            z = Q[:, keep].T @ est
            p_joint = self._wald_p(float((z ** 2 / w_[keep]).sum()), int(keep.sum()))
            if not p_joint <= st["alpha"]:  # NaN (fewer persons than df): no test
                break
            best = None
            for f, (lab, L) in fam.items():
                idx = left[f]
                if len(idx) < 2:
                    continue
                for j in idx:
                    c = L[j] - np.mean([L[o] for o in idx if o != j], axis=0)
                    shift, se = float(c @ th), float(np.sqrt(max(c @ V @ c, 1e-300)))
                    zj = shift / se
                    pj = self._z_p(zj)
                    if not (pj <= st["alpha"] and abs(shift) >= st["min_effect"]):
                        continue
                    if not cuttable(f, j):
                        if fam[f][0][j] not in held_back:
                            held_back.append(fam[f][0][j])
                        continue
                    if best is None or abs(zj) > abs(best[3]):
                        best = (f, j, shift, zj, se, pj)
            if best is None:
                if held_back:
                    # it still differs, at cells it can't be cut at: as for too many cuts (max_share), drop it
                    self.notes.append(
                        f"{a}: differs at {len(held_back)} cell(s) where fewer than two other anchors are uncut, so it "
                        "can't be cut there; dropped instead."
                    )
                    return dict(outcome="drop", cuts=cuts, p_remaining=p_joint)
                break
            f, j, shift, zj, se, pj = best
            left[f].remove(j)
            cuts.append(dict(cell=fam[f][0][j], family=f, shift=shift, se=se, z=zj, p=pj,
                             pull=shift / m_at_cell))
            if len(fam[f][0]) - len(left[f]) > st["max_share"] * len(fam[f][0]):
                return dict(outcome="drop", cuts=cuts, p_remaining=p_joint)
        if not cuts:
            return dict(outcome="none", cuts=[], p_remaining=p_joint)
        if not st["allow_cuts"]:
            return dict(outcome="drop", cuts=cuts, p_remaining=p_joint)
        return dict(outcome="cells", cuts=cuts, p_remaining=p_joint)

    # ---- the sequence ----------------------------------------------------------------------------------------
    def _apply(self, w, a, res, extra, rep):
        """w with anchor a's loop result applied (as suggestions)."""
        st = self.settings
        if res["outcome"] == "drop":
            if len(w.anchors) - 1 < st["min_anchors"]:
                self.notes.append(f"{a}: the suggestion is to drop it, but that would leave fewer than "
                                  f"min_anchors={st['min_anchors']} anchors; kept.")
                return w
            if self._sole_cells(a, w):
                self.notes.append(f"{a}: the suggestion is to drop it, but it is the only anchor not cut at "
                                  f"{self._sole_cells(a, w)[0]}, so the zero point there would be undefined; kept.")
                return w
            extra[a] = rep
            ev = (dict(reason="global effect", shift=res["severity"]["shift"], se=res["severity"]["se"],
                       p=res["severity"]["p"]) if res.get("severity")
                  else dict(reason="pattern", representation=rep, cells=len(res["cuts"])))
            return w.drop(a, source="suggested", **ev)
        if res["outcome"] == "cells":
            for c in res["cuts"]:
                w = w.cut(a, c["cell"], self.items, self.K, source="suggested", shift=c["shift"], se=c["se"],
                          p=c["p"])
        return w

    def _run(self):
        """Pattern first (each anchor's global effect is free in those fits, so a difference in global effect can't
        affect them), then global effect on the adjusted set (a cut anchor's global effect is defined by its uncut
        cells, so a shift at one cell does not pass for a difference in global effect), then refinement."""
        st = self.settings
        w = self.anchors
        extra = {}
        resolved = [a for a in w.anchors if a in w.cuts]
        rounds = []
        sev_ok = lambda r: abs(r["shift"]) >= st["min_effect"]
        # the level of each stage's decisions to flag an anchor (pattern, global effect, refinement)
        a_stage = st["alpha"] / 2 if st["split_alpha"] else st["alpha"]
        # pattern
        while True:
            screened = [a for a in w.anchors if a not in resolved]
            if len(w.anchors) < 2 or not screened:
                break
            sel = {a: self._select(a, w, extra) for a in screened}
            rej = corrected_reject({a: v[2] for a, v in sel.items()}, a_stage, st["correction"])
            flag = {a: rej[a] and sel[a][0] != "global" for a in screened}
            tab = pd.DataFrame({a: {"Representation": sel[a][0], "AIC gain": sel[a][1], "p": sel[a][2],
                                    "Flagged": flag[a]} for a in screened}).T
            tab.index.name = self.model.facet
            rounds.append(("pattern", tab))
            flagged = [a for a in screened if flag[a]]
            if not flagged:
                break
            if len(w.anchors) == 2:
                self.notes.append(f"{w.anchors[0]} and {w.anchors[1]} differ in pattern, but with two anchors the "
                                  "data can't say which one is wrong.")
                break
            worst = min(flagged, key=lambda a: (sel[a][2], -sel[a][1]))
            tab["Resolved"] = [a == worst for a in tab.index]
            rep = sel[worst][0]
            res = self._cell_loop(worst, rep, w, extra)
            if res["outcome"] == "none":
                self.notes.append(f"{worst}: flagged on {rep!r}, but no single cell differs significantly with "
                                  f"|shift| >= min_effect={st['min_effect']}; kept as an anchor.")
            w = self._apply(w, worst, res, extra, rep)
            resolved.append(worst)
        # global effect, on the adjusted set
        while len(w.anchors) >= 2:
            sev = self._severity(w, extra)["anchors"]
            rej = corrected_reject({a: r["p"] for a, r in sev.items()}, a_stage, st["correction"])
            flag = {a: rej[a] and sev_ok(sev[a]) for a in sev}
            tab = pd.DataFrame({a: {"Shift": r["shift"], "SE": r["se"], "p": r["p"], "Flagged": flag[a]}
                                for a, r in sev.items()}).T
            tab.index.name = self.model.facet
            rounds.append(("global effect", tab))
            flagged = [a for a in sev if flag[a]]
            if not flagged:
                break
            if len(w.anchors) == 2:
                self.notes.append(f"{w.anchors[0]} and {w.anchors[1]} differ in global effect, but with two anchors "
                                  "the data can't say which one is wrong.")
                break
            worst = min(flagged, key=lambda a: (sev[a]["p"], -abs(sev[a]["shift"])))
            tab["Resolved"] = [a == worst for a in tab.index]
            before = len(w.anchors)
            w = self._apply(w, worst, dict(outcome="drop", cuts=[], severity=sev[worst]), extra, "global")
            if len(w.anchors) == before:  # kept by min_anchors
                break
            if worst not in resolved:
                resolved.append(worst)
        # refinement: each resolved anchor again, against the final anchors: its pattern, then its global effect
        state = {}
        for a in resolved:
            base = w
            if a in base.anchors:
                for _, lab in list(base.cuts.get(a, [])):
                    base = base.restore(a, lab, self.items, self.K, source="suggested")
            else:
                base = base.restore(a, source="suggested")
                extra.pop(a, None)
            res, rep = None, "global"
            rep_, delta, p = self._select(a, base, extra)
            if p <= a_stage and rep_ != "global":
                res = self._cell_loop(a, rep_, base, extra)
                rep = rep_
                if res["outcome"] == "none":
                    res, rep = None, "global"
            w = base if res is None else self._apply(base, a, res, extra, rep)
            if a in w.anchors:
                s_a = self._severity(w, extra)["anchors"][a]
                if s_a["p"] <= a_stage and sev_ok(s_a):
                    w2 = self._apply(w, a, dict(outcome="drop", cuts=[], severity=s_a), extra, "global")
                    if a not in w2.anchors:
                        w, res, rep = w2, dict(outcome="drop", cuts=[], severity=s_a), "global"
            state[a] = ("drop" if a not in w.anchors else "cuts", rep, res)
        # the suggestions: the net change from the given set, with the estimates behind each
        rows, final = [], self.anchors
        f = self.model.facet
        ev = lambda c: {"Shift": c["shift"], "SE": c["se"], "z": c["shift"] / c["se"], "p": c["p"],
                        "Pull on zero point": c["pull"]}
        for a, (kind, rep, res) in state.items():
            given = [lab for _, lab in self.anchors.cuts.get(a, [])]
            if kind == "drop":
                if res.get("severity"):
                    rows.append({f: a, "Action": "drop", "Reason": "global effect", "Cell": None,
                                 **ev(res["severity"])})
                    final = final.drop(a, source="suggested", reason="global effect", shift=res["severity"]["shift"],
                                       se=res["severity"]["se"], p=res["severity"]["p"])
                else:
                    rows += [{f: a, "Action": "drop", "Reason": "pattern", "Cell": c["cell"], "Representation": rep,
                              **ev(c)} for c in res["cuts"]]
                    final = final.drop(a, source="suggested", reason="pattern", representation=rep,
                                       cells=len(res["cuts"]))
                continue
            new = {c["cell"]: c for c in (res["cuts"] if res else [])}
            for lab in given:
                if lab not in new:
                    rows.append({f: a, "Action": "restore", "Reason": "pattern", "Cell": lab})
                    final = final.restore(a, lab, self.items, self.K, source="suggested")
            for lab, c in new.items():
                rows.append({f: a, "Action": "keep cut" if lab in given else "cut", "Reason": "pattern",
                             "Cell": lab, "Representation": rep, **ev(c)})
                if lab not in given:
                    final = final.cut(a, lab, self.items, self.K, source="suggested", shift=c["shift"], se=c["se"],
                                      p=c["p"])
        n0 = len(self.anchors.anchors)
        if 2 * len(final.anchors) < n0:
            return self._no_core(rounds, n0 - len(final.anchors), n0)
        return rounds, self._suggestion_table(rows), final

    def _suggestion_table(self, rows):
        sugg = pd.DataFrame(rows, columns=[self.model.facet, "Action", "Reason", "Cell", "Representation", "Shift",
                                           "SE", "z", "p", "Pull on zero point"])
        sugg = sugg.drop(columns=[c for c in ("Representation",) if sugg[c].isna().all()])
        # pandas 3 infers a string column, whose missing values are NaN: keep None for 'no cell'
        sugg["Cell"] = pd.Series([r.get("Cell") for r in rows], index=sugg.index, dtype=object)
        return sugg

    def _no_core(self, rounds, dropped, n0):
        """The suggestions would drop more than half the anchors: no agreeing majority to adjust the set towards, so
        nothing is suggested (the screen's rounds are kept; the notes on the discarded steps are not)."""
        self.notes[:] = []
        self.notes.append(
            f"Suggestions would drop {dropped} of the {n0} anchors, so the anchors have no majority core that agree to "
            "adjust the set towards; no changes are suggested. Recommend choosing a different anchor set."
        )
        return rounds, self._suggestion_table([]), self.anchors

    # ---- actions ---------------------------------------------------------------------------------------------
    def _new(self, aset):
        st = dict(self.settings)
        chk = AnchorCheck(self.model, aset, self.estimation, previous=self, **st)
        self.model.anchor_check = chk
        return chk

    def cut(self, anchor, cell):
        """A new check with anchor no longer an anchor at cell (an item name, 'Threshold k' in any case, or an
        (item, threshold) pair)."""
        return self._new(self.anchors.cut(anchor, cell, self.items, self.K))

    def drop(self, anchor):
        """A new check without anchor in the anchor set (it stays in the data as an ordinary facet element)."""
        return self._new(self.anchors.drop(anchor))

    def restore(self, anchor, cell=None):
        """A new check with a dropped anchor put back (cell=None) or one cut undone."""
        return self._new(self.anchors.restore(anchor, cell, self.items, self.K))

    def apply_suggested(self):
        """A new check on the suggested anchor set."""
        return self._new(self.suggested)

    # ---- report ----------------------------------------------------------------------------------------------
    def __repr__(self):
        fmt_p = lambda p: "< 0.001" if p < 0.001 else f"= {p:.3f}"
        out = [f"Anchor check ({self.estimation}, the anchors' responses only)", f"  {self.anchors!r}"]
        if self.previous is not None:
            new = self.anchors.log[len(self.previous.anchors.log):]
            for e in new:
                out.append(f"  changed: {e['action']} {e['anchor']}" + (f" at {e['cell']}" if e["cell"] else ""))
        out.append("")
        out.append("Overall agreement" + (" (anchors with cut cells held on 'global', with their cuts)"
                                          if self.anchors.cuts else ""))
        for alt, r in self.overall.iterrows():
            test = "Wald" if alt == "global effect" else "LR"
            out.append(f"  {alt:<14} {test} p {fmt_p(r['p'])}")
        rej = corrected_reject(dict(self.overall["p"]), self.settings["alpha"], "holm")
        dis = [alt for alt in self.overall.index if rej[alt]]
        out.append(f"  overall p {fmt_p(self.overall_p)} (Holm)"
                   + (f"  -> anchors disagree on {' and on '.join(dis)}" if dis else "  -> no disagreement found"))
        corr = {"bh": "BH", "holm": "Holm", None: "no correction"}[self.settings["correction"]]
        n = {"pattern": 0, "global effect": 0}
        for kind, tab in self.screen:
            n[kind] += 1
            out.append("")
            out.append(f"Screen, {kind}, round {n[kind]} (each anchor against the rest; {corr} across anchors)")
            for a, r in tab.iterrows():
                flag = ("flagged" + (", resolved" if r.get("Resolved", False) else "")) if r["Flagged"] else ""
                what = (f"{r['Representation']:<11}" if kind == "pattern"
                        else f"shift {r['Shift']:+.3f} (SE {r['SE']:.3f})")
                out.append(f"  {a:<12} {what:<26} p {fmt_p(r['p']):<9} {flag}")
        out.append("")
        if len(self.suggestions):
            out.append("Suggested adjustments (worst first, each re-estimated against the final anchors)")
            disp = self.suggestions.copy()
            for c in disp:
                num = c in ("Shift", "SE", "z", "p", "Pull on zero point")
                disp[c] = disp[c].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if num else str(x)))
            out.append(disp.to_string(index=False))
        else:
            out.append("No adjustments suggested.")
        for n in self.notes:
            out.append(f"Note: {n}")
        return "\n".join(out)
