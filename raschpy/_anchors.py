"""
Anchor sets with cells cut: which facet elements (e.g. raters) define the zero point of a 'mixed' fit, and at which
cells an anchor does not. Used by MFRM.calibrate(model='mixed', anchors=), per_rater_model_selection(anchors=) and
check_anchor_homogeneity().
"""

import re

import pandas as pd

_THRESHOLD = re.compile(r"^\s*threshold\s*(\d+)\s*$", re.IGNORECASE)


def threshold_label(k):
    """The label of threshold k (1-based), as the package's threshold tables print it."""
    return f"Threshold {int(k)}"


def parse_cell(cell, items, K):
    """
    A cell as given by the user -- an item name ('Item_5'), a threshold label ('Threshold 2', any case) or an
    (item, threshold) pair for one cell of the matrix layout -- as (kind, label): ('item', name),
    ('threshold', 'Threshold k') or ('cell', (name, 'Threshold k')).
    """
    items = list(items)

    def threshold(x):
        m = _THRESHOLD.match(str(x))
        if m is None or not 1 <= int(m.group(1)) <= K:
            return None
        return threshold_label(m.group(1))

    if isinstance(cell, tuple):
        if len(cell) != 2 or cell[0] not in items or threshold(cell[1]) is None:
            raise ValueError(
                f"cell {cell!r}: a pair must be (item name, threshold label), e.g. ({items[0]!r}, 'Threshold 1')."
            )
        return ("cell", (cell[0], threshold(cell[1])))
    is_item, thr = cell in items, threshold(cell)
    if is_item and thr is not None:
        raise ValueError(
            f"cell {cell!r} is both an item name and a threshold label; rename the item to tell them apart."
        )
    if is_item:
        return ("item", cell)
    if thr is not None:
        return ("threshold", thr)
    raise ValueError(
        f"cell {cell!r}: expected an item name, a threshold label ('Threshold 1' to 'Threshold {K}') or an "
        f"(item, threshold) pair."
    )


class AnchorSet:
    """
    An anchor set: the anchors (facet element names), the cells cut for each ({anchor: [(kind, label), ...]};
    kinds 'item', 'threshold', 'cell'), the anchors as first given (original) and the log of every change
    (a list of dicts: action, facet element, cell, source 'suggested' or 'user', and the estimates behind a
    suggested change). pattern_free: anchors that anchor through their global effect only, their pattern over items
    and thresholds estimated freely (check_anchor_homogeneity suggests this for an anchor agreeing with the others in
    global effect but not in pattern); in calibrate(model='mixed', anchors=) give them a representation with a
    pattern in rater_models (per_rater_model_selection(anchors=) selects one). Immutable: drop / cut / restore /
    free_pattern / hold_pattern return a new AnchorSet.
    """

    def __init__(self, anchors, cuts=None, original=None, log=None, pattern_free=None):
        self.anchors = list(anchors)
        self.cuts = {a: list(c) for a, c in (cuts or {}).items() if c}
        self.original = list(anchors) if original is None else list(original)
        self.log = list(log or [])
        self.pattern_free = [a for a in (pattern_free or []) if a in self.anchors]

    @property
    def adjusted(self):
        """Whether the set differs from the anchors as first given."""
        return set(self.anchors) != set(self.original) or bool(self.cuts) or bool(self.pattern_free)

    def _copy(self, anchors=None, cuts=None, entry=None, pattern_free=None):
        return AnchorSet(self.anchors if anchors is None else anchors,
                         self.cuts if cuts is None else cuts, self.original,
                         self.log + ([entry] if entry else []),
                         self.pattern_free if pattern_free is None else pattern_free)

    def free_pattern(self, anchor, source="user", **evidence):
        """The anchor anchors through its global effect only, its pattern over items and thresholds free."""
        if anchor not in self.anchors:
            raise ValueError(f"{anchor!r} is not in the anchor set {self.anchors}.")
        if anchor in self.pattern_free:
            raise ValueError(f"{anchor!r}'s pattern is already free.")
        return self._copy(pattern_free=self.pattern_free + [anchor],
                          entry=dict(action="free pattern", anchor=anchor, cell=None, source=source, **evidence))

    def hold_pattern(self, anchor, source="user"):
        """Undo free_pattern: the anchor's pattern is held with the other anchors' again."""
        if anchor not in self.pattern_free:
            raise ValueError(f"{anchor!r}'s pattern is not free.")
        return self._copy(pattern_free=[a for a in self.pattern_free if a != anchor],
                          entry=dict(action="hold pattern", anchor=anchor, cell=None, source=source))

    def drop(self, anchor, source="user", **evidence):
        if anchor not in self.anchors:
            raise ValueError(f"{anchor!r} is not in the anchor set {self.anchors}.")
        cuts = {a: c for a, c in self.cuts.items() if a != anchor}
        return self._copy([a for a in self.anchors if a != anchor], cuts,
                          dict(action="drop", anchor=anchor, cell=None, source=source, **evidence),
                          [a for a in self.pattern_free if a != anchor])

    def cut(self, anchor, cell, items, K, source="user", **evidence):
        if anchor not in self.anchors:
            raise ValueError(f"{anchor!r} is not in the anchor set {self.anchors}.")
        c = parse_cell(cell, items, K)
        if c in self.cuts.get(anchor, []):
            raise ValueError(f"{anchor!r} is already cut at {c[1]!r}.")
        cuts = {a: list(v) for a, v in self.cuts.items()}
        cuts.setdefault(anchor, []).append(c)
        return self._copy(cuts=cuts, entry=dict(action="cut", anchor=anchor, cell=c[1], source=source, **evidence))

    def restore(self, anchor, cell=None, items=None, K=None, source="user"):
        """Put a dropped anchor back (cell=None) or undo one cut."""
        if cell is None:
            if anchor in self.anchors:
                raise ValueError(f"{anchor!r} is already an anchor.")
            return self._copy(self.anchors + [anchor], entry=dict(action="restore", anchor=anchor, cell=None,
                                                                  source=source))
        c = parse_cell(cell, items, K)
        if c not in self.cuts.get(anchor, []):
            raise ValueError(f"{anchor!r} is not cut at {c[1]!r}.")
        cuts = {a: [x for x in v if not (a == anchor and x == c)] for a, v in self.cuts.items()}
        return self._copy(cuts=cuts, entry=dict(action="restore", anchor=anchor, cell=c[1], source=source))

    def design_cuts(self, facets, items, K):
        """The cuts in MixedDesign's form: {anchor index: [('item', i) / ('threshold', h) / ('cell', i, h)]}."""
        facets, items = list(facets), list(items)
        th = lambda lab: int(_THRESHOLD.match(lab).group(1)) - 1
        out = {}
        for a, cells in self.cuts.items():
            if a not in facets:
                continue
            out[facets.index(a)] = [
                ("item", items.index(lab)) if kind == "item" else ("threshold", th(lab)) if kind == "threshold"
                else ("cell", items.index(lab[0]), th(lab[1]))
                for kind, lab in cells
            ]
        return out

    def table(self, items, K):
        """True where the facet element is an anchor for that item / threshold (matrix-cell cuts are listed in
        cuts, not here): rows the anchors as first given and any restored, columns the items then thresholds."""
        cols = list(items) + [threshold_label(k) for k in range(1, K + 1)]
        rows = list(dict.fromkeys(self.original + self.anchors))
        t = pd.DataFrame(False, index=rows, columns=cols)
        for a in self.anchors:
            t.loc[a] = True
            for kind, lab in self.cuts.get(a, []):
                if kind in ("item", "threshold"):
                    t.loc[a, lab] = False
        return t

    def log_table(self):
        t = pd.DataFrame(self.log)
        if "cell" in t:  # pandas 3 infers a string column, whose missing values are NaN: keep None for 'no cell'
            t["cell"] = pd.Series([e.get("cell") for e in self.log], index=t.index, dtype=object)
        return t

    def __repr__(self):
        cut = "; ".join(f"{a}: {', '.join(str(l) for _, l in c)}" for a, c in self.cuts.items())
        dropped = [a for a in self.original if a not in self.anchors]
        return (f"AnchorSet(anchors={self.anchors}" + (f", cut: {cut}" if cut else "")
                + (f", pattern free: {self.pattern_free}" if self.pattern_free else "")
                + (f", dropped: {dropped}" if dropped else "") + ")")
