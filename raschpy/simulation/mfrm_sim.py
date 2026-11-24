import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from raschpy.simulation.base_sim import Rasch_Sim


_REPRESENTATIONS = (
    "global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch",
)
_STRETCH_KEYS = {
    "centrality": ("lambda", "omega"),
    "pseudo_halo": ("lambda", "omega"),
    "bistretch": ("lambda", "omega_items", "omega_thresholds"),
}


def _finite(values, what):
    """values as a float array, raising if any is NaN or infinite."""
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: values must be finite (no NaN or infinity).")
    return arr


def _check_labels(keys, labels, what, kind):
    """Raise, naming them, if keys has labels not in labels or misses some of labels."""
    keys = list(keys)
    unknown = [k for k in keys if k not in labels]
    missing = [k for k in labels if k not in keys]
    if unknown or missing:
        raise ValueError(
            f"{what}: "
            + "; ".join(
                ([f"unknown {kind} label(s) {unknown}"] if unknown else [])
                + ([f"missing {kind} label(s) {missing}"] if missing else [])
            )
            + f". Expected the {kind} labels {list(labels)}."
        )


def _labelled_vector(v, labels, what, kind):
    """
    A per-item or per-threshold vector: a dict or Series is aligned by label (a missing or unknown label raises,
    naming it); an array, list or tuple is read in the order of labels and must have exactly len(labels) values.
    """
    if isinstance(v, (dict, pd.Series)):
        _check_labels(v.keys(), labels, what, kind)
        return _finite([v[k] for k in labels], what)
    arr = _finite(v, what)
    if arr.shape != (len(labels),):
        raise ValueError(
            f"{what}: expected {len(labels)} values (one per {kind}, in order), got shape {arr.shape}."
        )
    return arr


def _coerce_rater_params(model, params, rater, item_names, K, source="rater_models"):
    """
    One facet element's (e.g. rater's) parameters for representation model, as passed by the user, checked and
    returned in a standard form:
      global       float (severity)
      items        Series over items (the facet element's effect on each item)
      thresholds   Series over thresholds 1..K
      bivector     {'item_effects': Series over items, 'threshold_effects': Series over thresholds, summing to 0}
                   (a threshold part not summing to 0 is moved into the item part, which leaves every location
                   unchanged)
      matrix       DataFrame, items x thresholds 1..K
      centrality / pseudo_halo   {'lambda': float, 'omega': float}
      bistretch    {'lambda': float, 'omega_items': float, 'omega_thresholds': float}
    Accepted input: items -- dict or Series by item name, or I values; thresholds -- dict or Series keyed 1..K, or
    K values; matrix -- DataFrame (items x 1..K), {item: K values} or an I x K array; bivector -- a pair (item
    values, threshold values) or {'item_effects', 'threshold_effects'}; stretch models -- a dict with the keys
    above or the values in that order. Labelled input is aligned by label, unlabelled input is read in order and
    must have exactly the right shape. Every value must be finite.
    """
    what = f"{source}[{rater!r}] ({model})"
    thr = list(range(1, K + 1))
    if model == "global":
        if isinstance(params, dict):
            if set(params) != {"lambda"}:
                raise ValueError(f"{what}: expected a single value or {{'lambda': value}}, got keys {list(params)}.")
            params = params["lambda"]
        arr = _finite(params, what)
        if arr.ndim != 0:
            raise ValueError(f"{what}: expected a single value, got shape {arr.shape}.")
        return float(arr)
    if model == "items":
        return pd.Series(_labelled_vector(params, item_names, what, "item"), index=item_names)
    if model == "thresholds":
        return pd.Series(_labelled_vector(params, thr, what, "threshold"), index=thr)
    if model == "bivector":
        if isinstance(params, dict):
            if set(params) != {"item_effects", "threshold_effects"}:
                raise ValueError(
                    f"{what}: expected keys 'item_effects' and 'threshold_effects', got {list(params)}."
                )
            ie, te = params["item_effects"], params["threshold_effects"]
        elif isinstance(params, (tuple, list)) and len(params) == 2:
            ie, te = params
        else:
            raise ValueError(f"{what}: expected a pair (item effects, threshold effects) or a dict with those keys.")
        a = _labelled_vector(ie, item_names, what + " item_effects", "item")
        t = _labelled_vector(te, thr, what + " threshold_effects", "threshold")
        a, t = a + t.mean(), t - t.mean()
        return {"item_effects": pd.Series(a, index=item_names), "threshold_effects": pd.Series(t, index=thr)}
    if model == "matrix":
        if isinstance(params, pd.DataFrame):
            _check_labels(params.index, item_names, what, "item")
            _check_labels(params.columns, thr, what, "threshold")
            arr = _finite(params.loc[item_names, thr].to_numpy(), what)
        elif isinstance(params, dict):
            _check_labels(params.keys(), item_names, what, "item")
            arr = np.vstack(
                [_labelled_vector(params[i], thr, f"{what} item {i!r}", "threshold") for i in item_names]
            )
        else:
            arr = _finite(params, what)
            if arr.shape != (len(item_names), K):
                raise ValueError(
                    f"{what}: expected an array of shape ({len(item_names)}, {K}) (items x thresholds), got {arr.shape}."
                )
        return pd.DataFrame(arr, index=item_names, columns=thr)
    if model in _STRETCH_KEYS:
        keys = _STRETCH_KEYS[model]
        if isinstance(params, dict):
            if set(params) != set(keys):
                raise ValueError(f"{what}: expected keys {list(keys)}, got {list(params)}.")
            vals = [params[k] for k in keys]
        else:
            vals = list(np.ravel(_finite(params, what))) if not isinstance(params, (tuple, list)) else list(params)
            if len(vals) != len(keys):
                raise ValueError(f"{what}: expected {len(keys)} values {list(keys)}, got {len(vals)}.")
        out = {}
        for k, v in zip(keys, vals):
            arr = _finite(v, f"{what} {k}")
            if arr.ndim != 0:
                raise ValueError(f"{what}: {k} must be a single value, got shape {arr.shape}.")
            out[k] = float(arr)
        return out
    raise ValueError(f"{what}: unknown representation {model!r}; expected one of {list(_REPRESENTATIONS)}.")


def _rater_grid(model, p, delta, tau):
    """A facet element's (I, K) location deviations from delta_i + tau_h, from its standard-form parameters p."""
    I, K = len(delta), len(tau)
    if model == "global":
        return np.full((I, K), p)
    if model == "items":
        return np.repeat(p.to_numpy()[:, None], K, axis=1)
    if model == "thresholds":
        return np.repeat(p.to_numpy()[None, :], I, axis=0)
    if model == "bivector":
        return p["item_effects"].to_numpy()[:, None] + p["threshold_effects"].to_numpy()[None, :]
    if model == "matrix":
        return p.to_numpy().copy()
    if model == "centrality":
        return p["lambda"] + (p["omega"] - 1.0) * np.broadcast_to(tau[None, :], (I, K))
    if model == "pseudo_halo":
        return p["lambda"] + (p["omega"] - 1.0) * np.broadcast_to(delta[:, None], (I, K))
    return (
        p["lambda"]
        + (p["omega_items"] - 1.0) * delta[:, None]
        + (p["omega_thresholds"] - 1.0) * tau[None, :]
    )


def _neutral_params(model, item_names, K):
    """Standard-form parameters of a facet element with no effect under representation model."""
    thr = list(range(1, K + 1))
    if model == "global":
        return 0.0
    if model == "items":
        return pd.Series(0.0, index=item_names)
    if model == "thresholds":
        return pd.Series(0.0, index=thr)
    if model == "bivector":
        return {"item_effects": pd.Series(0.0, index=item_names), "threshold_effects": pd.Series(0.0, index=thr)}
    if model == "matrix":
        return pd.DataFrame(0.0, index=item_names, columns=thr)
    return {k: (0.0 if k == "lambda" else 1.0) for k in _STRETCH_KEYS[model]}


def _stretch_bounds(stretch_range, what="stretch_range"):
    """(lowest, highest) from a pair, or from a single number s as (1 - s/2, 1 + s/2)."""
    if np.ndim(stretch_range) == 0:
        s = float(stretch_range)
        return 1.0 - s / 2, 1.0 + s / 2
    lo, hi = (float(x) for x in stretch_range)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError(f"{what} must be (lowest, highest) with lowest <= highest, got {stretch_range!r}.")
    return lo, hi


def _spread_factor(spread, x):
    """spread / (max(x) - min(x)), or 0 when x has no spread (one value), so a single facet element or item gets
    no spread instead of NaN."""
    width = x.max() - x.min()
    return spread / width if width > 0 else 0.0


def _draw_multipliers(rng, R, stretch_range, centre=True):
    """
    R stretch multipliers. A single number s gives 1 plus a zero-mean deviation scaled so its maximum minus its
    minimum is s (the original behaviour, so seeded results are unchanged); a pair (lowest, highest) gives each
    multiplier its own draw between the bounds (any real numbers, lowest <= highest).
    """
    x = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
    if np.ndim(stretch_range) == 0:
        x *= _spread_factor(stretch_range, x)
        if centre:
            x -= x.mean()
        return 1.0 + x
    lo, hi = _stretch_bounds(stretch_range)
    return lo + (x + 1.96) / 3.92 * (hi - lo)


def _draw_pattern(rng, shape, spread):
    """A zero-sum (zero-sum along both axes for a 2-D shape) truncated-normal pattern, scaled so its maximum
    minus its minimum is spread (all zeros when it has no spread)."""
    x = truncnorm.rvs(-1.96, 1.96, size=shape, random_state=rng)
    if x.ndim == 1:
        x = x - x.mean()
    else:
        x = x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()
    width = x.max() - x.min()
    return x * (spread / width) if width > 0 else np.zeros(shape)


def _draw_between(rng, lo, hi):
    """One value between lo and hi, from the truncated normal mapped onto [lo, hi]."""
    z = float(truncnorm.rvs(-1.96, 1.96, random_state=rng))
    return lo + (z + 1.96) / 3.92 * (hi - lo)


class MFRM_Sim(Rasch_Sim):
    """
    Simulate polytomous response data for the Many-Facet Rasch Model (MFRM).

    Generates item locations, shared Rasch-Andrich thresholds, person
    locations, and facet_element effect parameters under one of five parameterisations,
    then computes category probabilities and samples scores for every
    facet_element-person-item combination. Simulation runs automatically on instantiation;
    access results via self.responses.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    no_of_facet_elements : int
        Number of facet_elements to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    model : str, default 'global'
        Facet element (e.g. rater) effect parameterisation. One of:
        'global'     — single scalar facet_effect per facet_element;
        'items'      — separate facet_effect per (facet_element, item);
        'thresholds' — separate facet_effect per (facet_element, threshold);
        'matrix'     — full facet_effect per (facet_element, item, threshold);
        'bivector'   — additive (item effect + threshold effect) facet_effect
                       per (facet_element, item, threshold).
        'mixed'      — each facet element on its own representation, any of
                       the five above or 'centrality', 'pseudo_halo',
                       'bistretch' (see rater_models).
    item_range : float, default 2
        Total spread of item locations in logits.
    facet_range : float, default 2
        Total spread of facet_element effects in logits. Not used when
        model='bivector' — see item_facet_range/threshold_facet_range.
    item_facet_range : float, default 2
        model='bivector': total spread of per-(facet_element, item)
        facet_effect effects across the full facet_element x item matrix in logits.
        model='mixed': spread (maximum minus minimum) of each facet element's
        own item-effect pattern (items, bivector, matrix).
    threshold_facet_range : float, default 1
        model='bivector': total spread of per-(facet_element, threshold)
        facet_effect effects across the full facet_element x threshold matrix in logits.
        model='mixed': spread of each facet element's own threshold-effect
        pattern (thresholds, bivector, matrix).
    category_base : float, default 2
        Base width of each rating category. Larger values produce wider,
        more ordered categories.
    person_sd : float, default 1.5
        Standard deviation of the person location distribution (normal).
    max_disorder : float, default 0
        Maximum threshold disorder. 0 produces perfectly ordered thresholds.
    offset : float, default 0
        Mean shift applied to person locations after centring.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    shared_missing : bool, default True
        If True, the same persons are missing across all facet_elements (correlated
        missingness). If False, missingness is independent across facet_elements.
    manual_persons : array-like or None, default None
        Custom person measures. Length must equal no_of_persons.
    manual_items : array-like or None, default None
        Custom item locations. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector, length max_score. Must satisfy
        sum(thresholds) == 0.
    manual_raters : dict or array-like or None, default None
        Custom facet_element effect parameters. Structure must match the chosen model:
        global — array-like of length no_of_facet_elements;
        items  — {facet_element: values}, the values a dict or Series by item
                 name or no_of_items values in item order;
        thresholds — {facet_element: values}, a dict or Series keyed
                 1..max_score or max_score values in order;
        matrix — {facet_element: values}, the values {item: max_score values},
                 an items x thresholds DataFrame or an array.
        Labelled values are aligned by label (a missing or unknown label
        raises); unlabelled values are read in order.
        Not supported for model='bivector' — use manual_item_effects/
        manual_threshold_effects instead.
    manual_item_effects : dict or None, default None
        model='bivector' only. Custom per-(facet_element, item) facet_effect
        effects. Structure: {facet_element: {item: float}}.
    manual_threshold_effects : dict or None, default None
        model='bivector' only. Custom per-(facet_element, threshold) facet_effect
        effects. Structure: {facet_element: array of length max_score}.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.
    manual_facet_names : list of str or None, default None
        Custom facet_element labels. If None, labels are '{Facet}_1', '{Facet}_2', etc. where Facet is the capitalised facet name.
    facet : str or None, default None
        Name of the facet (e.g. 'rater', 'judge'), used in generated names
        ('{Facet}_1', ...), attribute aliases and messages. If None, names and
        aliases use 'rater' (Rater_1, ...) as before, and messages say
        "facet element(s)", with "(e.g. rater)" at first mention.
    facet_plural : str or None, default None
        Plural of facet, if not facet + 's'.
    seed : int or None, default None
        Seed for the random number generator. Pass an int for a fully
        reproducible simulation; None (default) draws fresh entropy each run.
        This seed only affects this simulation's own local random generator
        (numpy.random.default_rng(seed)), not the global numpy.random.seed()
        state -- a prior call to numpy.random.seed(N) has no effect on this
        simulation's output. Always pass seed= directly here for a
        reproducible run.
    missing_method : {'bernoulli', 'exact'}, default 'bernoulli'
        How `missing` is applied. 'bernoulli' makes each response missing
        independently with probability `missing`, so the realised proportion
        varies around it. 'exact' makes exactly round(missing * cells)
        responses missing, per facet_element when shared_missing=False, or
        in the one shared pattern when shared_missing=True. Both use the same
        random draws, so observed scores are identical and the missing
        patterns differ only in the few cells nearest the cut-off; seeded
        results are reproducible either way.

    rater_models : list, dict, Series or None, default None
        model='mixed' only: each facet element's representation. A list
        gives one representation per facet element, in order, with values
        drawn at random. A dict or Series {facet_element: model} or
        {facet_element: (model, params)} fixes the representation, and with
        params its values, in the form:
          global       a single value (severity);
          items        a dict or Series by item name, or no_of_items values;
          thresholds   a dict or Series keyed 1..max_score, or max_score values;
          bivector     a pair (item values, threshold values) or
                       {'item_effects': ..., 'threshold_effects': ...};
          matrix       a DataFrame (items x 1..max_score), {item: values}
                       or a no_of_items x max_score array;
          centrality / pseudo_halo   {'lambda', 'omega'} or (lambda, omega);
          bistretch    {'lambda', 'omega_items', 'omega_thresholds'} or a
                       triple.
        Labelled values are aligned by label; unlabelled values are read in
        order and must have exactly the right shape. Facet elements left out,
        and all of them when None, get a representation drawn from
        candidate_models and random values. Effects are relative to the shared
        item locations and thresholds (native_items, native_thresholds); the
        stretch models give lambda_r + (omega_r - 1) times the thresholds,
        item locations or both.
    candidate_models : list or None, default None
        model='mixed' only: the representations drawn from for facet elements
        whose representation is not given. None means all eight.
    proportions : list, dict, Series or None, default None
        model='mixed' only: weights for drawing from candidate_models (in its
        order, or keyed by representation). None means equal weights.
    interaction_range : float, default 1
        model='mixed' only: spread of a matrix facet element's own
        item-by-threshold interactions (summing to 0 over items and over
        thresholds).
    stretch_range : float or (lowest, highest), default 1
        model='mixed' only: bounds of the drawn stretch multipliers (any real
        numbers, 0 and negative values included, lowest <= highest); a single
        number s means (1 - s/2, 1 + s/2).
    item_stretch_range, threshold_stretch_range : float, pair or None, default None
        model='mixed' only: as stretch_range, for the item multipliers
        (pseudo_halo, bistretch) / threshold multipliers (centrality,
        bistretch) only.
    anchors : list or None, default None
        Facet element names. The reported true values are expressed in the
        frame calibrate_anchor(model, anchors) gives the estimates: the
        anchors' mean effect is taken into the item locations and thresholds
        (for the stretch models, the anchors' mean shift is removed and the
        multipliers are divided by the anchors' mean multiplier). Every
        location is unchanged apart from a common shift, which the reported
        person locations take up, so the data are unchanged. None reports
        the effects centred over all facet elements, as before.
    neutral_anchors : bool, default False
        With anchors, the anchors are generated with no effect at all (zero
        severity, item, threshold and interaction effects; for the stretch
        models shift 0 and multipliers 1), unless their values are passed,
        and the other facet elements' effects are drawn without being
        centred. The anchored frame is then the generating frame. Does
        nothing when anchors is None.
    Attributes set
    --------------
    responses : pandas.DataFrame
        Simulated response matrix with (Rater, Person) MultiIndex and items
        as columns. Values are integers in [0, max_score] or NaN (missing).
        This is the primary output — pass directly to MFRM(responses).
    persons : pandas.Series
        True person location parameters, indexed by person.
    items : pandas.Series
        True item location parameters, indexed by item.
    thresholds : numpy.ndarray
        True Rasch-Andrich threshold vector, length max_score,
        zero-sum.
    facet_effects : pandas.Series or dict
        True facet_element effect parameters. Structure depends on model.
    item_effects : pandas.DataFrame
        model='bivector' only. True per-(facet_element, item) facet_effect
        effects. Zero-mean per item across facet_elements; the per-facet_element
        mean is free and carries that facet_element's overall level (including
        any level folded in from ``threshold_effects`` during canonicalisation).
    threshold_effects : pandas.DataFrame
        model='bivector' only. True per-(facet_element, threshold) facet_effect
        effects. Zero-sum per facet_element (shape only) *and* zero-mean per
        threshold across facet_elements -- the exact representation the bivector
        estimator identifies, so recovered and generating tables are directly
        comparable. ``full[r,i,k] = item_effects[r,i] + threshold_effects[r,k]``
        is invariant to the per-facet_element gauge shift this pins down, so the
        sampled responses do not depend on it.
    cat_probs : dict
        {cat: DataFrame} of category probabilities used for simulation.
    person_names : list of str
        Person labels.
    item_names : list of str
        Item labels.
    rater_names : list of str
        Facet element labels.
    model : str
        Facet element parameterisation used for simulation.
    anchors : list or None
        The anchors the reported values are framed on.
    rater_models : pandas.Series
        model='mixed' only: each facet element's representation.
    rater_params : dict
        model='mixed' only: {facet_element: (model, params)}, each facet
        element's parameters in the form rater_models takes, relative to
        native_items and native_thresholds (pass it back as rater_models=,
        with manual_items=native_items, manual_thresholds=native_thresholds
        and manual_persons=native_persons, to simulate new data from the same
        parameters).
    native_items, native_thresholds, native_persons : pandas.Series
        model='mixed' only: the generating item locations, thresholds and
        person locations the facet elements' parameters are relative to.
    facet_effects_mixed_{model} : pandas.Series or DataFrame
        model='mixed' only, one per representation used: its facet elements'
        parameters in that representation's layout (bivector as
        facet_effects_mixed_bivector_items / _thresholds; the stretch models
        as columns lambda, omega or omega_items, omega_thresholds).
    facet_effects (model='mixed') : pandas.DataFrame
        Every facet element's effects in the matrix layout ((facet_element,
        item) x threshold), in the anchors' frame (all facet elements' when
        anchors is None): the anchors' mean item and threshold margins are in
        items and thresholds, their item-by-threshold remainder in the
        effects, as calibrate_anchor gives a matrix fit.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements,
        max_score,
        model="global",
        item_range=2,
        facet_range=2,
        item_facet_range=2,
        threshold_facet_range=1,
        category_base=2,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        shared_missing=True,
        manual_persons=None,
        manual_items=None,
        manual_thresholds=None,
        manual_facet_effects=None,
        manual_raters=None,
        manual_item_effects=None,
        manual_threshold_effects=None,
        manual_person_names=None,
        manual_item_names=None,
        manual_facet_names=None,
        facet=None,
        facet_plural=None,
        seed=None,
        missing_method="bernoulli",
        rater_models=None,
        candidate_models=None,
        proportions=None,
        interaction_range=1,
        stretch_range=1,
        item_stretch_range=None,
        threshold_stretch_range=None,
        anchors=None,
        neutral_anchors=False,
        _params_only=False,
    ):
        """
        Instantiate and run an MFRM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        self._rng = np.random.default_rng(seed)

        if model not in ("global", "items", "thresholds", "matrix", "bivector", "mixed"):
            raise ValueError(
                "model must be one of 'global', 'items', 'thresholds', 'matrix', 'bivector', 'mixed' "
                "(the stretch representations have their own classes: MFRM_Sim_Centrality, "
                "MFRM_Sim_PseudoHalo, MFRM_Sim_Bistretch)."
            )
        if model == "mixed":
            given = [
                n
                for n, v in (
                    ("manual_raters", manual_raters),
                    ("manual_facet_effects", manual_facet_effects),
                    ("manual_item_effects", manual_item_effects),
                    ("manual_threshold_effects", manual_threshold_effects),
                )
                if v is not None
            ]
            if given:
                raise ValueError(
                    f"model='mixed' takes each facet element's parameters through rater_models= "
                    f"({{rater: (model, params)}}), not {', '.join(given)}."
                )
            if isinstance(rater_models, (dict, pd.Series)) and manual_facet_names is None:
                keys = list(rater_models.keys())
                n = no_of_facet_elements if no_of_facet_elements is not None else len(keys)
                default = [f"{(facet or 'rater').capitalize()}_{r + 1}" for r in range(int(n))]
                if not set(keys) <= set(default):
                    if len(keys) != int(n):
                        raise ValueError(
                            f"rater_models names facet elements {keys} that are not the default names "
                            f"{default}: give all {int(n)} of them, or pass manual_facet_names."
                        )
                    manual_facet_names = keys
                no_of_facet_elements = n
        elif any(
            v is not None
            for v in (rater_models, candidate_models, proportions, item_stretch_range, threshold_stretch_range)
        ):
            raise ValueError(
                "rater_models, candidate_models, proportions, item_stretch_range and threshold_stretch_range "
                "apply to model='mixed' only."
            )
        if model == "bivector" and manual_raters is not None:
            raise ValueError(
                "manual_raters is not supported for model='bivector'; "
                "use manual_item_effects/manual_threshold_effects instead."
            )

        self.model = model
        self.item_facet_range = item_facet_range
        self.threshold_facet_range = threshold_facet_range
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        # manual_raters is a convenience alias: dict {name: facet_effect_array} → manual_facet_effects
        if manual_raters is not None:
            if manual_facet_effects is not None:
                raise ValueError(
                    "Pass manual_raters or manual_facet_effects, not both."
                )
            if manual_facet_names is None and hasattr(manual_raters, "keys"):
                manual_facet_names = list(manual_raters.keys())
            if no_of_facet_elements is None:
                no_of_facet_elements = len(manual_raters)
            # Conversion to manual_facet_effects is deferred until after item_names are set.

        self.no_of_facet_elements = int(no_of_facet_elements)
        # Facet name: messages say "facet element(s)" (glossed "(e.g. rater)" at
        # first mention) unless a name was passed;
        # generated names (Rater_1, ...) and attribute aliases keep 'rater'
        # when none was, so seeded output is unchanged.
        if facet is None:
            facet = "rater"
            self.facet_label, self.facets_label = "facet element", "facet elements"
            self.facet_label_eg = "facet element (e.g. rater)"
            self.facets_label_eg = "facet elements (e.g. raters)"
        else:
            self.facet_label = facet
            self.facets_label = (
                facet_plural if facet_plural is not None else facet + "s"
            )
            self.facet_label_eg, self.facets_label_eg = (
                self.facet_label,
                self.facets_label,
            )
        self.facet = facet
        self.facets = facet_plural if facet_plural is not None else facet + "s"
        setattr(self, f"no_of_{self.facets}", self.no_of_facet_elements)
        self.dropped_facet_elements = []
        setattr(self, f"dropped_{self.facets}", self.dropped_facet_elements)
        if self.facets != "facet_elements":
            setattr(self, f"drop_{self.facets}", self.drop_facet_elements)
        self.max_score = max_score
        self.item_range = item_range
        self.facet_range = facet_range
        self.category_base = category_base
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self._check_missing_method(missing_method)
        self.shared_missing = shared_missing

        # ------------------------------------------------------------------
        # Persons
        # ------------------------------------------------------------------
        if manual_person_names is not None:
            assert (
                len(manual_person_names) == self.no_of_persons
            ), "Length of person names must match number of persons."
            self.person_names = manual_person_names
        else:
            self.person_names = [f"Person_{p + 1}" for p in range(self.no_of_persons)]

        if manual_persons is not None:
            assert (
                len(manual_persons) == self.no_of_persons
            ), "Length of manual persons must match number of persons."
            locations = np.array(manual_persons)
        else:
            locations = self._rng.normal(0, self.person_sd, self.no_of_persons)
            locations -= locations.mean()
            locations += self.offset

        self.persons = pd.Series(
            {person: loc for person, loc in zip(self.person_names, locations)}
        )

        # ------------------------------------------------------------------
        # Items
        # ------------------------------------------------------------------
        if manual_item_names is not None:
            assert (
                len(manual_item_names) == self.no_of_items
            ), "Length of item names must match number of items."
            self.item_names = manual_item_names
        else:
            self.item_names = [f"Item_{i + 1}" for i in range(self.no_of_items)]

        if manual_items is not None:
            assert (
                len(manual_items) == self.no_of_items
            ), "Length of manual locations must match number of items."
            items = np.array(manual_items)
        else:
            items = self._rng.uniform(0, 1, self.no_of_items)
            items *= _spread_factor(self.item_range, items)
            items -= items.mean()

        self.items = pd.Series({item: d for item, d in zip(self.item_names, items)})

        # ------------------------------------------------------------------
        # Thresholds (shared RSM structure across all four models)
        # ------------------------------------------------------------------
        if manual_thresholds is not None:
            assert (
                len(manual_thresholds) == self.max_score
            ), "Number of manual thresholds must equal max_score."
            assert np.isclose(
                sum(manual_thresholds), 0
            ), "Manual thresholds must sum to zero."
            self.thresholds = pd.Series(
                np.array(manual_thresholds), index=range(1, self.max_score + 1)
            )
        else:
            cat_widths = self._rng.uniform(
                self.max_disorder,
                2 * self.category_base - self.max_disorder,
                self.max_score,
            )
            thresholds = np.array([cat_widths[:k].sum() for k in range(self.max_score)])
            thresholds -= thresholds.mean()
            self.thresholds = pd.Series(thresholds, index=range(1, self.max_score + 1))

        # ------------------------------------------------------------------
        # Raters
        # ------------------------------------------------------------------
        if manual_facet_names is not None:
            assert (
                len(manual_facet_names) == self.no_of_facet_elements
            ), "Length of facet_element names must match number of facet_elements."
            self.facet_names = manual_facet_names
        else:
            self.facet_names = [
                f"{self.facet.capitalize()}_{r + 1}"
                for r in range(self.no_of_facet_elements)
            ]
        self.rater_names = self.facet_names  # alias for default facet
        self.facet_ids = self.facet_names
        self.rater_ids = self.facet_names  # alias for default facet

        # ------------------------------------------------------------------
        # Facet effects (model-specific)
        # ------------------------------------------------------------------
        self.anchors = self._check_anchors(self.facet_names, anchors)
        self.neutral_anchors = bool(neutral_anchors)
        self._neutral = self._neutral_mask(self.facet_names, self.anchors, neutral_anchors)
        if self.model == "mixed":
            self._simulate_mixed(
                rater_models,
                candidate_models,
                proportions,
                interaction_range,
                stretch_range,
                item_stretch_range,
                threshold_stretch_range,
                anchors,
                neutral_anchors,
            )
        else:
            if manual_raters is not None:
                if self.model == "global":
                    if hasattr(manual_raters, "items"):
                        manual_facet_effects = pd.Series(
                            {r: float(v) for r, v in manual_raters.items()}
                        )
                    else:
                        manual_facet_effects = pd.Series(
                            np.asarray(manual_raters, dtype=float)
                        )
                elif self.model in ("items", "thresholds"):
                    # labelled values (dict / Series) are aligned by label, others read in order
                    manual_facet_effects = pd.DataFrame(
                        {
                            r: _coerce_rater_params(
                                self.model, v, r, self.item_names, self.max_score, "manual_raters"
                            )
                            for r, v in manual_raters.items()
                        }
                    ).T
                elif self.model == "matrix":
                    I, K = self.no_of_items, self.max_score
                    blocks = []
                    for r, v in manual_raters.items():
                        if isinstance(v, (dict, pd.DataFrame)):
                            # {item_name: K values} or an items x thresholds DataFrame, aligned by label
                            mat = _coerce_rater_params(
                                "matrix", v, r, self.item_names, K, "manual_raters"
                            ).to_numpy()
                        else:
                            arr = np.asarray(v, dtype=float)
                            if arr.ndim == 0:
                                # scalar → global: uniform (I, K)
                                mat = np.full((I, K), float(arr))
                            elif arr.ndim == 1 and len(arr) == I:
                                # (I,) → items: tile over thresholds
                                mat = np.tile(arr[:, None], (1, K))
                            elif arr.ndim == 1 and len(arr) == K:
                                # (K,) → thresholds: tile over items
                                mat = np.tile(arr[None, :], (I, 1))
                            elif arr.ndim == 2 and arr.shape == (I, K):
                                # (I, K) → matrix: use as-is
                                mat = arr
                            elif isinstance(v, (tuple, list)) and len(v) == 2:
                                # (item_vec, thresh_vec) → bivector
                                iv = np.asarray(v[0], dtype=float)
                                tv = np.asarray(v[1], dtype=float)
                                mat = iv[:, None] + tv[None, :]
                            else:
                                raise ValueError(
                                    f"manual_raters['{r}']: cannot infer model from shape {arr.shape}. "
                                    f"Pass a scalar, dict, (I,) array, (K,) array, (I,K) array, "
                                    f"or (item_vec, thresh_vec) tuple."
                                )
                        blocks.append(
                            pd.DataFrame(
                                mat, index=self.item_names, columns=range(1, K + 1)
                            )
                        )
                    mi = pd.MultiIndex.from_product(
                        [list(manual_raters.keys()), self.item_names],
                        names=["facet_element", "item"],
                    )
                    manual_facet_effects = pd.DataFrame(
                        np.vstack([b.values for b in blocks]),
                        index=mi,
                        columns=range(1, K + 1),
                    )
            self.facet_effects = self._generate_facet_effects(
                manual_facet_effects, manual_item_effects, manual_threshold_effects
            )
        setattr(self, f"{self.facets}_effects", self.facet_effects)

        if _params_only:
            # Parameters only (the stretch-model wrappers' preliminary sims):
            # no category probabilities or responses.
            return

        # ------------------------------------------------------------------
        # Scores + missing data. Rows are stacked (facet_element, person),
        # facet_element-major.
        # ------------------------------------------------------------------
        self._prepare_cat_probs()
        R = self.no_of_facet_elements
        scores = self._draw_scores(R * self.no_of_persons, self._cat_prob_rows)
        mask = self._missing_mask(n_blocks=R, shared=shared_missing)
        self.responses = self._responses_frame(scores, mask, self._row_index)
        if self.model != "mixed":
            self._report_in_anchor_frame()

    # ------------------------------------------------------------------
    # Anchors: the frame of the reported true values
    # ------------------------------------------------------------------

    @staticmethod
    def _check_anchors(names, anchors):
        """anchors as a list of facet element names (None stays None), raising on unknown or empty."""
        if anchors is None:
            return None
        anchors = list(anchors)
        unknown = [a for a in anchors if a not in list(names)]
        if unknown or not anchors:
            raise ValueError(
                f"anchors must be a non-empty list of facet element names; unknown: {unknown}. "
                f"The names are {list(names)}."
            )
        return anchors

    @staticmethod
    def _neutral_mask(names, anchors, neutral_anchors):
        """Boolean mask over names: the anchors when neutral_anchors is set (no-op without anchors)."""
        anchors = MFRM_Sim._check_anchors(names, anchors)
        return np.array([bool(neutral_anchors) and anchors is not None and r in anchors for r in names])

    def _report_in_anchor_frame(self):
        """
        With anchors, re-express the reported true values in the frame calibrate_anchor(model, anchors) gives the
        estimates: global -- the anchors' mean effect is subtracted from every facet element; items / thresholds --
        the anchors' mean effect on each item / threshold moves into that item location / threshold, which are then
        centred; bivector -- both parts as items and thresholds; matrix -- the anchors' mean item and threshold margins
        move into the item locations and thresholds (their item-by-threshold remainder stays in the effects). Every
        location is unchanged apart from a common shift, which the reported person locations take up, so the data
        are unchanged.
        """
        if self.anchors is None:
            return
        a, names, K = self.anchors, list(self.facet_names), self.max_score
        m = self.model
        if m == "global":
            adj = float(self.facet_effects.loc[a].mean())
            self.facet_effects = self.facet_effects - adj
            shift = adj
        elif m in ("items", "thresholds"):
            adj = self.facet_effects.loc[a].mean(axis=0)
            self.facet_effects = self.facet_effects - adj
            base = (self.items if m == "items" else self.thresholds) + adj.to_numpy()
            shift = float(base.mean())
            base = base - shift
            if m == "items":
                self.items = base
            else:
                self.thresholds = base
        elif m == "bivector":
            ia = self.item_effects.loc[a].mean(axis=0)
            ta = self.threshold_effects.loc[a].mean(axis=0)
            self.item_effects = self.item_effects - ia
            self.threshold_effects = self.threshold_effects - ta
            items, thresholds = self.items + ia.to_numpy(), self.thresholds + ta.to_numpy()
            shift = float(items.mean() + thresholds.mean())
            self.items, self.thresholds = items - items.mean(), thresholds - thresholds.mean()
            mi = pd.MultiIndex.from_product([names, self.item_names], names=["facet_element", "item"])
            full = self.item_effects.to_numpy()[:, :, None] + self.threshold_effects.to_numpy()[:, None, :]
            self.facet_effects = pd.DataFrame(full.reshape(-1, K), index=mi, columns=range(1, K + 1))
        else:  # matrix
            arr = self.facet_effects.to_numpy().reshape(len(names), self.no_of_items, K)
            c = arr[[names.index(x) for x in a]].mean(axis=0)
            da, dt = c.mean(axis=1), c.mean(axis=0)
            arr = arr - (da[:, None] + dt[None, :] - c.mean())[None]
            items, thresholds = self.items + da, self.thresholds + dt
            shift = float(items.mean() + thresholds.mean() - c.mean())
            self.items, self.thresholds = items - items.mean(), thresholds - thresholds.mean()
            self.facet_effects = pd.DataFrame(
                arr.reshape(-1, K), index=self.facet_effects.index, columns=self.facet_effects.columns
            )
        self.persons = self.persons - shift
        setattr(self, f"{self.facets}_effects", self.facet_effects)

    def _report_stretch_anchor_frame(self, anchors, neutral_anchors):
        """
        The stretch simulations' version of _report_in_anchor_frame, in the frame calibrate_anchor gives a
        stretch model (estimation='cp-full-rcl' / 'cpat-rcl'): the anchors' mean shift is subtracted from every
        lambda_r, every multiplier is divided by the anchors' mean multiplier on its axis, and the item locations /
        thresholds are multiplied by it. Every location is unchanged apart from the anchors' mean shift, which the
        reported person locations take up.
        """
        self.anchors = self._check_anchors(self.facet_names, anchors)
        self.neutral_anchors = bool(neutral_anchors)
        if self.anchors is None:
            return
        a, K = self.anchors, self.max_score
        lam = self.lambda_
        om_i = getattr(self, "omega_items", self.omega if self.model == "pseudo_halo" else None)
        om_t = getattr(self, "omega_thresholds", self.omega if self.model == "centrality" else None)
        lam_a = float(lam.loc[a].mean())
        wi = 1.0 if om_i is None else float(om_i.loc[a].mean())
        wt = 1.0 if om_t is None else float(om_t.loc[a].mean())
        if wi == 0 or wt == 0:
            raise ValueError(
                "anchors: the anchors' mean stretch multiplier is 0, so the frame calibrate_anchor uses (each "
                "multiplier divided by the anchors' mean) does not exist. Choose other anchors."
            )
        self.lambda_ = lam - lam_a
        delta, tau = self.items * wi, self.thresholds * wt
        self.items, self.thresholds = delta, tau
        if om_i is not None:
            om_i = om_i / wi
        if om_t is not None:
            om_t = om_t / wt
        if self.model == "centrality":
            self.omega = om_t
            eff = self.lambda_.to_numpy()[:, None] + (om_t.to_numpy()[:, None] - 1.0) * tau.to_numpy()[None, :]
            self.facet_effects = pd.DataFrame(eff, index=self.facet_names, columns=range(1, K + 1))
        elif self.model == "pseudo_halo":
            self.omega = om_i
            eff = self.lambda_.to_numpy()[:, None] + (om_i.to_numpy()[:, None] - 1.0) * delta.to_numpy()[None, :]
            self.facet_effects = pd.DataFrame(eff, index=self.facet_names, columns=self.item_names)
        else:
            self.omega_items, self.omega_thresholds = om_i, om_t
            eff = (
                self.lambda_.to_numpy()[:, None, None]
                + (om_i.to_numpy()[:, None, None] - 1.0) * delta.to_numpy()[None, :, None]
                + (om_t.to_numpy()[:, None, None] - 1.0) * tau.to_numpy()[None, None, :]
            )
            self.facet_effects = pd.DataFrame(
                eff.reshape(-1, K), index=self.facet_effects.index, columns=range(1, K + 1)
            )
        self.persons = self.persons - lam_a
        setattr(self, f"{self.facets}_effects", self.facet_effects)

    # ------------------------------------------------------------------
    # Mixed representations
    # ------------------------------------------------------------------

    def _simulate_mixed(
        self,
        rater_models,
        candidate_models,
        proportions,
        interaction_range,
        stretch_range,
        item_stretch_range,
        threshold_stretch_range,
        anchors,
        neutral_anchors,
    ):
        """
        model='mixed': each facet element's representation and parameters (manual or drawn), its location
        deviations from the shared item locations and thresholds, and the reported tables. The data come from the
        native parameters (relative to native_items / native_thresholds, with native_persons); the matrix layout
        (facet_effects) and items / thresholds / persons are that grid re-expressed with the anchors' mean item and
        threshold margins taken into the item locations and thresholds, as calibrate_anchor does.
        """
        rng = self._rng
        R, I, K = self.no_of_facet_elements, self.no_of_items, self.max_score
        names = list(self.facet_names)
        pool = list(_REPRESENTATIONS if candidate_models is None else candidate_models)
        bad = [m for m in pool if m not in _REPRESENTATIONS]
        if bad or not pool:
            raise ValueError(
                f"candidate_models must be a non-empty list of {list(_REPRESENTATIONS)}, got {candidate_models!r}."
            )
        if proportions is None:
            weights = np.full(len(pool), 1.0 / len(pool))
        else:
            if isinstance(proportions, (dict, pd.Series)):
                _check_labels(proportions.keys(), pool, "proportions", "representation")
                weights = _finite([proportions[m] for m in pool], "proportions")
            else:
                weights = _finite(proportions, "proportions")
                if weights.shape != (len(pool),):
                    raise ValueError(
                        f"proportions: expected {len(pool)} values, one per representation in {pool}, got "
                        f"shape {weights.shape}."
                    )
            if np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("proportions must be non-negative and not all zero.")
            weights = weights / weights.sum()
        anchors = self._check_anchors(names, anchors)
        neutral = set(anchors) if (neutral_anchors and anchors is not None) else set()

        # representations: given (list or dict), the rest drawn from the pool
        spec = {}
        if rater_models is None:
            pass
        elif isinstance(rater_models, (dict, pd.Series)):
            unknown = [r for r in rater_models.keys() if r not in names]
            if unknown:
                raise ValueError(f"rater_models: unknown facet element(s) {unknown}; the names are {names}.")
            for r, v in rater_models.items():
                if isinstance(v, str):
                    spec[r] = (v, None)
                elif isinstance(v, (tuple, list)) and len(v) == 2 and isinstance(v[0], str):
                    spec[r] = (v[0], v[1])
                else:
                    raise ValueError(
                        f"rater_models[{r!r}]: expected a representation name or (representation, parameters), "
                        f"got {v!r}."
                    )
        else:
            given = list(rater_models)
            if len(given) != R:
                raise ValueError(
                    f"rater_models has {len(given)} representations but there are {R} facet elements: give one "
                    f"per facet element, in order."
                )
            spec = {r: (m, None) for r, m in zip(names, given)}
        for r, (m, _) in spec.items():
            if m not in _REPRESENTATIONS:
                raise ValueError(f"rater_models[{r!r}]: unknown representation {m!r}; expected one of {list(_REPRESENTATIONS)}.")
        drawn = rng.choice(len(pool), size=R, p=weights)  # drawn for every facet element so seeds stay aligned
        models = {r: spec[r][0] if r in spec else pool[drawn[k]] for k, r in enumerate(names)}

        # random values: severity across all facet elements (as the global simulation), the rest per facet element
        free = [r for r in names if r not in neutral]
        sev = dict.fromkeys(names, 0.0)
        if len(free) > 1:
            x = truncnorm.rvs(-1.96, 1.96, size=len(free), random_state=rng)
            x *= _spread_factor(self.facet_range, x)
            if not neutral:
                x -= x.mean()
            sev.update(zip(free, x))
        lo_i, hi_i = _stretch_bounds(stretch_range if item_stretch_range is None else item_stretch_range,
                                     "item_stretch_range" if item_stretch_range is not None else "stretch_range")
        lo_t, hi_t = _stretch_bounds(stretch_range if threshold_stretch_range is None else threshold_stretch_range,
                                     "threshold_stretch_range" if threshold_stretch_range is not None else "stretch_range")
        thr = list(range(1, K + 1))
        params = {}
        for r in names:
            m, given = spec.get(r, (models[r], None))
            if given is not None:
                params[r] = _coerce_rater_params(m, given, r, self.item_names, K)
                continue
            if r in neutral:
                params[r] = _neutral_params(m, self.item_names, K)
                continue
            lam = sev[r]
            if m == "global":
                params[r] = float(lam)
            elif m == "items":
                params[r] = pd.Series(lam + _draw_pattern(rng, I, self.item_facet_range), index=self.item_names)
            elif m == "thresholds":
                params[r] = pd.Series(lam + _draw_pattern(rng, K, self.threshold_facet_range), index=thr)
            elif m == "bivector":
                params[r] = {
                    "item_effects": pd.Series(lam + _draw_pattern(rng, I, self.item_facet_range), index=self.item_names),
                    "threshold_effects": pd.Series(_draw_pattern(rng, K, self.threshold_facet_range), index=thr),
                }
            elif m == "matrix":
                g = (
                    lam
                    + _draw_pattern(rng, I, self.item_facet_range)[:, None]
                    + _draw_pattern(rng, K, self.threshold_facet_range)[None, :]
                    + _draw_pattern(rng, (I, K), interaction_range)
                )
                params[r] = pd.DataFrame(g, index=self.item_names, columns=thr)
            elif m == "centrality":
                params[r] = {"lambda": float(lam), "omega": _draw_between(rng, lo_t, hi_t)}
            elif m == "pseudo_halo":
                params[r] = {"lambda": float(lam), "omega": _draw_between(rng, lo_i, hi_i)}
            else:
                params[r] = {
                    "lambda": float(lam),
                    "omega_items": _draw_between(rng, lo_i, hi_i),
                    "omega_thresholds": _draw_between(rng, lo_t, hi_t),
                }

        delta, tau, theta = self.items.to_numpy(), self.thresholds.to_numpy(), self.persons.to_numpy()
        grid = np.stack([_rater_grid(models[r], params[r], delta, tau) for r in names])  # (R, I, K)
        self.native_items, self.native_thresholds, self.native_persons = (
            self.items.copy(), self.thresholds.copy(), self.persons.copy(),
        )
        self._native_grid = grid
        self.rater_models = pd.Series(models, name="Representation")[names]
        self.rater_params = {r: (models[r], params[r]) for r in names}
        self.anchors = anchors
        self.neutral_anchors = bool(neutral_anchors)

        # one table per representation, in its single-representation layout (only the facet elements on it)
        self._mixed_tables = []
        for m in _REPRESENTATIONS:
            on = [r for r in names if models[r] == m]
            if not on:
                continue
            if m == "global":
                tabs = {"": pd.Series({r: params[r] for r in on}, name="Severity")}
            elif m in ("items", "thresholds"):
                tabs = {"": pd.DataFrame({r: params[r] for r in on}).T}
            elif m == "bivector":
                tabs = {
                    "_items": pd.DataFrame({r: params[r]["item_effects"] for r in on}).T,
                    "_thresholds": pd.DataFrame({r: params[r]["threshold_effects"] for r in on}).T,
                }
            elif m == "matrix":
                tabs = {"": pd.concat({r: params[r] for r in on}, names=["facet_element", "item"])}
            else:
                tabs = {"": pd.DataFrame({r: params[r] for r in on}).T[list(_STRETCH_KEYS[m])]}
            for suffix, tab in tabs.items():
                name = f"facet_effects_mixed_{m}{suffix}"
                setattr(self, name, tab)
                self._mixed_tables.append(name)

        # reported frame: the anchors' (all facet elements') mean item and threshold margins move into the item
        # locations and thresholds; their item-by-threshold remainder stays in the facet element effects
        ref = [names.index(a) for a in (anchors if anchors is not None else names)]
        c = grid[ref].mean(axis=0)
        da, dt = c.mean(axis=1), c.mean(axis=0)
        rep = grid - (da[:, None] + dt[None, :] - c.mean())[None]
        items, thresholds = delta + da, tau + dt
        shift = items.mean() + thresholds.mean() - c.mean()  # common shift of every location, carried by the persons
        self.items = pd.Series(items - items.mean(), index=self.item_names)
        self.thresholds = pd.Series(thresholds - thresholds.mean(), index=thr)
        self.persons = pd.Series(theta - shift, index=self.persons.index)
        mi = pd.MultiIndex.from_product([names, self.item_names], names=["facet_element", "item"])
        self.facet_effects = pd.DataFrame(rep.reshape(-1, K), index=mi, columns=thr)

    # ------------------------------------------------------------------
    # Facet effect generation
    # ------------------------------------------------------------------

    def _generate_facet_effects(
        self,
        manual_facet_effects,
        manual_item_effects=None,
        manual_threshold_effects=None,
    ):
        """Generate or validate facet_element effect parameters for the given model."""

        if self.model == "global":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                sev = np.array(manual_facet_effects)
            else:
                sev = truncnorm.rvs(
                    -1.96, 1.96, size=self.no_of_facet_elements, random_state=self._rng
                )
                sev *= _spread_factor(self.facet_range, sev)
                if not self._neutral.any():
                    sev -= sev.mean()
                sev[self._neutral] = 0.0
            return pd.Series(
                {facet_element: s for facet_element, s in zip(self.facet_names, sev)}
            )

        elif self.model == "items":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(
                            -1.96, 1.96, size=self.no_of_items, random_state=self._rng
                        )
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I)
                sev *= _spread_factor(self.facet_range, sev)
                # Centre per item (column)
                if not self._neutral.any():
                    for i in range(self.no_of_items):
                        sev[:, i] -= sev[:, i].mean()
                sev[self._neutral] = 0.0
                return pd.DataFrame(
                    sev, index=self.facet_names, columns=self.item_names
                )

        elif self.model == "thresholds":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(
                            -1.96, 1.96, size=self.max_score, random_state=self._rng
                        )
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, K)
                sev *= _spread_factor(self.facet_range, sev)
                # Centre per threshold (column), matching _estimate_raters_thresholds'
                # per-threshold independent zero-centring across facet_elements —
                # otherwise a nonzero true per-column mean shows up as a constant
                # per-threshold bias against the (correctly zero-centred) estimates.
                if not self._neutral.any():
                    for k in range(self.max_score):
                        sev[:, k] -= sev[:, k].mean()
                sev[self._neutral] = 0.0
                return pd.DataFrame(
                    sev, index=self.facet_names, columns=range(1, self.max_score + 1)
                )

        elif self.model == "matrix":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects.index.get_level_values(0).unique())
                    == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        [
                            truncnorm.rvs(
                                -1.96, 1.96, size=self.max_score, random_state=self._rng
                            )
                            for _ in range(self.no_of_items)
                        ]
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I, K)
                sev *= _spread_factor(self.facet_range, sev)
                # Centre per (item, threshold) cell across facet_elements, matching
                # _estimate_raters_matrix's per-cell independent zero-centring across
                # facet_elements — otherwise a nonzero true per-cell mean shows up as a
                # constant per-(item, threshold) bias against the (correctly
                # zero-centred) estimates.
                if not self._neutral.any():
                    for i in range(self.no_of_items):
                        for k in range(self.max_score):
                            sev[:, i, k] -= sev[:, i, k].mean()
                sev[self._neutral] = 0.0
                mi = pd.MultiIndex.from_product(
                    [self.facet_names, self.item_names], names=["facet_element", "item"]
                )
                return pd.DataFrame(
                    sev.reshape(-1, self.max_score),
                    index=mi,
                    columns=range(1, self.max_score + 1),
                )

        elif self.model == "bivector":
            # Item effects — (R, I): free mean per facet_element, centred per item.
            if manual_item_effects is not None:
                assert (
                    len(manual_item_effects) == self.no_of_facet_elements
                ), "Length of manual item effects must match number of facet_elements."
                item_effects = manual_item_effects
            else:
                raw = np.array(
                    [
                        truncnorm.rvs(
                            -1.96, 1.96, size=self.no_of_items, random_state=self._rng
                        )
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I)
                raw *= _spread_factor(self.item_facet_range, raw)
                if not self._neutral.any():
                    raw -= raw.mean(axis=0, keepdims=True)
                raw[self._neutral] = 0.0
                item_effects = pd.DataFrame(
                    raw, index=self.facet_names, columns=self.item_names
                )

            # Threshold effects — (R, K). Generated with per-threshold (axis 0)
            # centring, then canonicalised below to the exact representation the
            # bivector estimator identifies.
            if manual_threshold_effects is not None:
                assert (
                    len(manual_threshold_effects) == self.no_of_facet_elements
                ), "Length of manual threshold effects must match number of facet_elements."
                threshold_effects = manual_threshold_effects
            else:
                raw = np.array(
                    [
                        truncnorm.rvs(
                            -1.96, 1.96, size=self.max_score, random_state=self._rng
                        )
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, K)
                raw *= _spread_factor(self.threshold_facet_range, raw)
                if not self._neutral.any():
                    raw -= raw.mean(axis=0, keepdims=True)
                raw[self._neutral] = 0.0
                threshold_effects = pd.DataFrame(
                    raw, index=self.facet_names, columns=range(1, self.max_score + 1)
                )

            # Canonicalise to the identified bivector representation: the split
            # full[r,i,k] = item_effect[r,i] + threshold_effect[r,k] has a
            # per-facet_element gauge freedom (add c_r to a rater's item-effect
            # row, subtract it from its threshold-effect row -> same full), which
            # the estimator pins by forcing threshold_effect to sum to zero
            # within each facet_element and letting the item-effect row carry the
            # per-facet_element level. Match that here, so recovered and
            # generating parameters are directly comparable, table for table.
            # full is unchanged, so the sampled responses are identical.
            if isinstance(item_effects, pd.DataFrame) and isinstance(
                threshold_effects, pd.DataFrame
            ):
                per_element_level = threshold_effects.mean(axis=1)
                threshold_effects = threshold_effects.sub(per_element_level, axis=0)
                item_effects = item_effects.add(per_element_level, axis=0)

            self.item_effects = item_effects
            self.threshold_effects = threshold_effects

            # Reconstruct full facet_effect matrix, same (facet_element, item) x
            # threshold format as the "matrix" model, so cat_prob/exp_score/etc.
            # can treat bivector and matrix identically downstream.
            mi = pd.MultiIndex.from_product(
                [self.facet_names, self.item_names], names=["facet_element", "item"]
            )
            rows = []
            for facet_element in self.facet_names:
                for item in self.item_names:
                    ie = (
                        item_effects.loc[facet_element, item]
                        if isinstance(item_effects, pd.DataFrame)
                        else item_effects[facet_element][item]
                    )
                    te = (
                        threshold_effects.loc[facet_element].values
                        if isinstance(threshold_effects, pd.DataFrame)
                        else threshold_effects[facet_element]
                    )
                    rows.append(np.asarray(te, dtype=float) + ie)
            return pd.DataFrame(rows, index=mi, columns=range(1, self.max_score + 1))

    # ------------------------------------------------------------------
    # Category probability computation
    # ------------------------------------------------------------------

    def _prepare_cat_probs(self):
        """Cache the arrays and labels behind the category probabilities."""
        R, I, K = self.no_of_facet_elements, self.no_of_items, self.max_score
        mixed = self.model == "mixed"  # the data come from the native parameters
        self._theta = (self.native_persons if mixed else self.persons).to_numpy()
        self._delta = (self.native_items if mixed else self.items).to_numpy()
        thresholds = self.native_thresholds if mixed else self.thresholds
        self._tau_sums = [thresholds.iloc[:cat].sum() for cat in range(K + 1)]
        # the stretch-model wrappers relabel self.model afterwards
        self._prob_model = "matrix" if mixed else self.model

        effects = self.facet_effects
        if self.model == "global":
            self._effects = effects.loc[self.facet_names].to_numpy()  # (R,)
        elif self.model == "items":
            self._effects = effects.loc[
                self.facet_names, self.item_names
            ].to_numpy()  # (R, I)
        else:
            if self.model == "thresholds":
                arr = effects.loc[self.facet_names].to_numpy()  # (R, K)
            elif mixed:
                arr = self._native_grid
            else:  # matrix, bivector
                mi = pd.MultiIndex.from_product([self.facet_names, self.item_names])
                arr = effects.loc[mi].to_numpy().reshape(R, I, K)
            # sums of each facet_element's first `cat` threshold effects; row-major
            # so each sum adds in the same order as a 1-D (pandas) row sum
            arr = np.ascontiguousarray(arr)
            self._effects = np.stack(
                [arr[..., :cat].sum(axis=-1) for cat in range(K + 1)], axis=-1
            )

        self._row_index = pd.concat(
            [pd.DataFrame(index=self.persons.index) for _ in self.facet_names],
            keys=self.facet_names,
        ).index
        self._prob_labels = (self._row_index, list(self.item_names))
        self._cat_probs = None

    def _logits(self, r, c_p, cat):
        """Category-`cat` numerator logits for facet_element r; c_p = person - item."""
        tau_sum = self._tau_sums[cat]
        if self._prob_model == "global":
            return cat * (c_p - self._effects[r]) - tau_sum
        if self._prob_model == "items":
            return cat * (c_p - self._effects[r][None, :]) - tau_sum
        if self._prob_model == "thresholds":
            return cat * c_p - tau_sum - self._effects[r, cat]
        return (cat * c_p - tau_sum) - self._effects[r, :, cat][None, :]

    def _cat_prob_rows(self, start, stop):
        """Category probabilities [P(X=0), ..., P(X=K)] for stacked rows start:stop."""
        N = self.no_of_persons
        num = [
            np.empty((stop - start, self.no_of_items))
            for _ in range(self.max_score + 1)
        ]
        for r in range(start // N, (stop - 1) // N + 1):
            a, b = max(start, r * N), min(stop, (r + 1) * N)
            c_p = self._theta[a - r * N : b - r * N, None] - self._delta[None, :]
            for cat, x in enumerate(num):
                x[a - start : b - start] = self._logits(r, c_p, cat)
        for x in num:
            np.exp(x, out=x)
        den = num[0].copy()
        for x in num[1:]:
            den += x
        return [x / den for x in num]

    @property
    def cat_probs(self):
        """{cat: DataFrame} of category probabilities (computed on first access)."""
        if self._cat_probs is None:
            index, columns = self._prob_labels
            probs = self._cat_prob_rows(
                0, self.no_of_facet_elements * self.no_of_persons
            )
            self._cat_probs = {
                cat: pd.DataFrame(p, index=index, columns=columns)
                for cat, p in enumerate(probs)
            }
        return self._cat_probs

    @cat_probs.setter
    def cat_probs(self, value):
        self._cat_probs = value

    # ------------------------------------------------------------------
    # Dropping facet elements
    # ------------------------------------------------------------------

    def drop_facet_elements(self, which, min_persons=3):
        """
        Remove facet elements (e.g. raters) from the simulation: their
        responses and their generating parameters (facet_effects, plus
        item_effects/threshold_effects for bivector and lambda_/omega
        parameters for the stretch models). Aliased as
        drop_{facet_plural}() (e.g. drop_raters()). Responses of the other
        facet elements, and all item, threshold and person parameters, are
        unchanged.

        Parameters
        ----------
        which : 'thin', 'extreme' or list
            'thin' removes facet elements that rated fewer than min_persons
            persons (including those with no responses); 'extreme' removes
            those that used only the lowest or only the highest score
            category; a list removes the named ones.
        min_persons : int, default 3
            Threshold for 'thin'.

        Attributes set
        --------------
        dropped_facet_elements : list
            All facet elements removed so far.
        """
        if isinstance(which, str):
            if which not in ("thin", "extreme"):
                raise ValueError(
                    f"which must be 'thin', 'extreme' or a list of {self.facet_label_eg} names."
                )
            if which == "thin":
                rated = (
                    self.responses.notna()
                    .any(axis=1)
                    .groupby(level=0, sort=False)
                    .sum()
                )
                which = [
                    fe for fe in self.facet_names if rated.get(fe, 0) < min_persons
                ]
            else:
                lo = self.responses.min(axis=1).groupby(level=0, sort=False).min()
                hi = self.responses.max(axis=1).groupby(level=0, sort=False).max()
                which = [
                    fe
                    for fe in self.facet_names
                    if hi.get(fe) == 0 or lo.get(fe) == self.max_score
                ]
        else:
            which = list(which)
            unknown = [fe for fe in which if fe not in self.facet_names]
            if unknown:
                raise ValueError(f"Names not in the data: {unknown}.")
        if not which:
            return
        keep_pos = [i for i, fe in enumerate(self.facet_names) if fe not in which]
        keep = [fe for fe in self.facet_names if fe not in which]
        for name in self._FACET_PARAMS:
            obj = getattr(self, name, None)
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                setattr(self, name, obj.loc[keep])
        if self.model == "mixed":
            self.rater_models = self.rater_models.loc[keep]
            self.rater_params = {r: v for r, v in self.rater_params.items() if r in keep}
            self._native_grid = self._native_grid[keep_pos]
            for name in list(self._mixed_tables):
                tab = getattr(self, name)
                tab = tab[tab.index.get_level_values(0).isin(keep)]
                if len(tab):
                    setattr(self, name, tab)
                else:
                    delattr(self, name)
                    self._mixed_tables.remove(name)
            if self.anchors is not None:
                self.anchors = [a for a in self.anchors if a in keep] or None
        setattr(self, f"{self.facets}_effects", self.facet_effects)
        self.responses = self.responses[
            ~self.responses.index.get_level_values(0).isin(which)
        ]
        self.facet_names = [fe for fe in self.facet_names if fe not in which]
        self.rater_names = self.facet_names  # alias for default facet
        self.facet_ids = self.facet_names
        self.rater_ids = self.facet_names  # alias for default facet
        self.no_of_facet_elements = len(self.facet_names)
        setattr(self, f"no_of_{self.facets}", self.no_of_facet_elements)
        if hasattr(self, "_effects"):
            self._effects = self._effects[keep_pos]
            self._row_index = pd.concat(
                [pd.DataFrame(index=self.persons.index) for _ in self.facet_names],
                keys=self.facet_names,
            ).index
            self._prob_labels = (self._row_index, list(self.item_names))
            self._cat_probs = None
        self.dropped_facet_elements = self.dropped_facet_elements + which
        setattr(self, f"dropped_{self.facets}", self.dropped_facet_elements)
        warnings.warn(
            f"{len(which)} {self.facet_label_eg if len(which) == 1 else self.facets_label_eg} "
            f"removed: {which}.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def rename_rater(self, old, new):
        """
        Rename a single facet element (e.g. rater) in the simulation: in the
        responses and in the generating parameters, so recovery comparisons
        stay aligned.

        Parameters
        ----------
        old : str
            Current facet element name.
        new : str
            Desired new facet element name.
        """
        if self._check_rename(self.facet_label_eg, list(self.facet_names), old, new):
            self._relabel("facet", {old: new})

    def rename_raters_all(self, new_names):
        """
        Rename all facet elements (e.g. raters) at once, in the responses and
        in the generating parameters.

        Parameters
        ----------
        new_names : list of str
            New facet element names in the same order as self.facet_names.
        """
        if self._check_rename_all(
            self.facet_label_eg, self.facets_label, list(self.facet_names), new_names
        ):
            self._relabel("facet", dict(zip(self.facet_names, new_names)))

    _FACET_PARAMS = (
        "facet_effects",
        "item_effects",
        "threshold_effects",
        "lambda_",
        "omega",
        "omega_items",
        "omega_thresholds",
    )


# ------------------------------------------------------------------
# Backwards-compatible subclass aliases
# ------------------------------------------------------------------


class MFRM_Sim_Global(MFRM_Sim):
    """
    MFRM simulation — global (scalar) facet_element effect parameterisation.

    Convenience subclass of MFRM_Sim with model='global' fixed.
    Each facet_element has a single scalar facet_effect estimate applied equally
    across all items and thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='global' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="global",
            **kw,
        )


class MFRM_Sim_Items(MFRM_Sim):
    """
    MFRM simulation — items (per facet_element×item) facet_effect parameterisation.

    Convenience subclass of MFRM_Sim with model='items' fixed.
    Each facet_element has a separate facet_effect for each item, constant across
    thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='items' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="items",
            **kw,
        )


class MFRM_Sim_Thresholds(MFRM_Sim):
    """
    MFRM simulation — thresholds (per facet_element×threshold) facet_effect parameterisation.

    Convenience subclass of MFRM_Sim with model='thresholds' fixed.
    Each facet_element has a separate facet_effect for each threshold, constant across
    items. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='thresholds' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="thresholds",
            **kw,
        )


class MFRM_Sim_Matrix(MFRM_Sim):
    """
    MFRM simulation — matrix (full facet_element×item×threshold tensor) parameterisation.

    Convenience subclass of MFRM_Sim with model='matrix' fixed.
    Each facet_element has a separate facet_effect for every (item, threshold) combination.
    See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='matrix' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="matrix",
            **kw,
        )


class MFRM_Sim_Bivector(MFRM_Sim):
    """
    MFRM simulation — bivector (additive item + threshold facet_element effects)
    parameterisation.

    Convenience subclass of MFRM_Sim with model='bivector' fixed. The bivector
    model treats facet_element effect as the sum of two additive components: a
    per-(facet_element, item) item effect and a per-(facet_element, threshold)
    threshold effect. This is analogous to treating the facet_element as an RSM
    (rather than a PCM as in the matrix model) — each facet_element has a location
    profile across items and a shape profile across thresholds, but the two
    are independent and additive.

    True facet_effect for facet_element r, item i, threshold k is:

        facet_effect[r, i, k] = item_effect[r, i] + threshold_effect[r, k]

    Identification constraints:
    - item_effect: free mean per facet_element (overall facet_element effect lives here).
    - threshold_effect: zero-sum per facet_element across thresholds (shape only,
      no net location contribution).

    See MFRM_Sim for full parameter docs, including item_facet_range/
    threshold_facet_range and manual_item_effects/manual_threshold_effects
    (the bivector-specific parameters in place of MFRM_Sim's generic
    facet_range/manual_raters).
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='bivector' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="bivector",
            **kw,
        )
        self.model = "bivector"


class MFRM_Sim_Centrality(MFRM_Sim):
    """
    MFRM simulation -- Jin & Wang (2018)-style facet element (e.g. rater) centrality/extremity
    parameterisation (2 true parameters per facet element: location shift lambda_r,
    threshold-stretch omega_r), restricting the fully-free 'thresholds'
    parameterisation. omega_r is Jin & Wang's own notation for this
    threshold-stretch coefficient.

    Thin wrapper: generates a preliminary MFRM_Sim(model='thresholds') to
    get realistic item locations/thresholds/persons, draws true
    lambda_r/omega_r per facet element, computes the implied full (facet element x threshold)
    location table via lambda_rk = lambda_r + (omega_r - 1) * tau_k, then
    delegates to the full MFRM_Sim(model='thresholds', manual_raters=...)
    machinery (fixed to the same item/threshold/person values as the
    preliminary sim) for everything else -- category probabilities and
    response sampling. All the actual generative machinery is reused
    unchanged; only lambda_r/omega_r and their implied location table are new.

    omega_r > 1 spreads reference thresholds apart for that facet element
    ("central"); 0 < omega_r < 1 compresses them ("extreme"); omega_r == 1
    is neutral. See MFRM._derive_stretch_model / MFRM.calibrate_centrality
    for the corresponding closed-form estimation side.

    Parameters
    ----------
    global_range : float, default 2
        Total spread (maximum - minimum) of true lambda_r (location
        shift) values across facet elements. Renamed from MFRM_Sim's own
        facet_range for this and the other two stretch models -- lambda_r
        is the parameter that collapses this representation to 'global'
        when every facet element's omega_r == 1, and 'global_range' pairs
        directly with 'stretch_range' below, matching the global_/stretch
        attribute alias naming (see MFRM._PARAM_ALIASES).
    stretch_range : float or (lowest, highest), default 1
        Total spread of true omega_r values around 1 (analogous to
        global_range for lambda_r). E.g. stretch_range=1 typically gives
        omega_r roughly in [0.5, 1.5].
        A pair (lowest, highest) instead draws each facet element's
        omega_r between the bounds, which may be any real numbers (0 and
        negative values included), lowest <= highest.
    manual_lambda : array-like or None, default None
        Custom true lambda_r values, length no_of_facet_elements. If None,
        drawn the same way MFRM_Sim's own facet_range scales locations.
    manual_omega : array-like or None, default None
        Custom true omega_r values, length no_of_facet_elements. If None,
        drawn as 1 + a stretch_range-scaled, zero-mean deviation.
    seed : int or None, default None
        Seed for both lambda_r/omega_r generation and everything else
        (persons/items/thresholds/response sampling, via the preliminary
        and final MFRM_Sim calls). Pass an int for full reproducibility.
        This seed only affects this simulation's own local random generator,
        not the global numpy.random.seed() state -- a prior call to
        numpy.random.seed(N) has no effect on this simulation's output.

    See MFRM_Sim for all other parameter docs (item_range, person_sd,
    missing, manual_items, manual_thresholds, manual_persons, etc. --
    all forwarded unchanged to both the preliminary and final sims, so
    supplying e.g. manual_items yourself works exactly as it would for
    MFRM_Sim_Thresholds).

    Attributes set (in addition to the usual MFRM_Sim attributes)
    ---------------------------------------------------------------
    lambda_ : pandas.Series
        True generating lambda_r (location shift) per facet element.
    omega : pandas.Series
        True generating omega_r (threshold stretch) per facet element.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        global_range=2,
        stretch_range=1,
        manual_lambda=None,
        manual_omega=None,
        seed=None,
        **kw,
    ):
        if "facet_range" in kw:
            raise TypeError(
                "MFRM_Sim_Centrality.__init__() got an unexpected keyword "
                "argument 'facet_range' -- renamed to 'global_range' (it "
                "controls the spread of true lambda_r values, not "
                "MFRM_Sim's own facet_range, which would otherwise silently "
                "absorb this into the discarded preliminary simulation)."
            )
        anchors = kw.pop("anchors", None)
        neutral_anchors = kw.pop("neutral_anchors", False)
        prelim = MFRM_Sim(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="thresholds",
            seed=seed,
            _params_only=True,
            **kw,
        )

        rng = np.random.default_rng(seed)
        R = prelim.no_of_facet_elements
        neutral = MFRM_Sim._neutral_mask(prelim.facet_names, anchors, neutral_anchors)

        if manual_lambda is not None:
            assert (
                len(manual_lambda) == R
            ), "Length of manual_lambda must match number of facet_elements."
            lam = np.array(manual_lambda, dtype=float)
        else:
            lam = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            lam *= _spread_factor(global_range, lam)
            if not neutral.any():
                lam -= lam.mean()
            lam[neutral] = 0.0

        if manual_omega is not None:
            assert (
                len(manual_omega) == R
            ), "Length of manual_omega must match number of facet_elements."
            omega = np.array(manual_omega, dtype=float)
        else:
            omega = _draw_multipliers(rng, R, stretch_range, centre=not neutral.any())
            omega[neutral] = 1.0

        tau = prelim.thresholds.values
        implied_raters = {
            r: lam[i] + (omega[i] - 1.0) * tau for i, r in enumerate(prelim.facet_names)
        }

        # Already captured into prelim (and re-supplied explicitly below) --
        # drop from kw so a caller who passed these directly doesn't hit a
        # "got multiple values for keyword argument" TypeError.
        for key in (
            "manual_items",
            "manual_thresholds",
            "manual_persons",
            "manual_item_names",
            "manual_person_names",
            "manual_facet_names",
            "manual_raters",
        ):
            kw.pop(key, None)

        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=prelim.no_of_facet_elements,
            max_score=prelim.max_score,
            model="thresholds",
            manual_items=prelim.items.values,
            manual_thresholds=prelim.thresholds.values,
            manual_persons=prelim.persons.values,
            manual_person_names=list(prelim.person_names),
            manual_item_names=list(prelim.item_names),
            manual_facet_names=list(prelim.facet_names),
            manual_raters=implied_raters,
            seed=seed,
            **kw,
        )
        self.model = "centrality"
        self.lambda_ = pd.Series(lam, index=self.facet_names)
        self.omega = pd.Series(omega, index=self.facet_names)
        self._report_stretch_anchor_frame(anchors, neutral_anchors)

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega, matching MFRM's own
        global_{model}/stretch_{model} attribute aliases (see
        MFRM._PARAM_ALIASES) -- 'global_' for 'lambda_' (no model suffix
        needed here, since this sim class is already model-specific;
        trailing underscore because 'global', like 'lambda', is also a
        Python keyword), 'stretch' for 'omega'. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        if name == "global_":
            return self.lambda_
        if name == "stretch":
            return self.omega
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class MFRM_Sim_PseudoHalo(MFRM_Sim):
    """
    MFRM simulation -- (pseudo-)halo-effect parameterisation, the
    item-facet analogue of MFRM_Sim_Centrality (2 true parameters per
    facet element (e.g. rater): location shift lambda_r, item-difficulty stretch omega_r),
    restricting the fully-free 'items' parameterisation.

    Thin wrapper: generates a preliminary MFRM_Sim(model='items') to get
    realistic item locations/thresholds/persons, draws true
    lambda_r/omega_r per facet element, computes the implied full (facet element x item)
    location table via mu_ri = lambda_r + (omega_r - 1) * delta_i, then
    delegates to the full MFRM_Sim(model='items', manual_raters=...)
    machinery (fixed to the same item/threshold/person values as the
    preliminary sim) for everything else. All the actual generative
    machinery is reused unchanged; only lambda_r/omega_r and their implied
    location table are new.

    omega_r < 1 compresses the spread of reference item locations toward
    zero for that facet element (the pseudo-halo signature -- items look more
    similar in difficulty to that facet element than they really are); omega_r > 1
    exaggerates real item-difficulty differences; omega_r == 1 is neutral.
    See MFRM._derive_stretch_model / MFRM.calibrate_pseudo_halo for the
    corresponding closed-form estimation side.

    Parameters
    ----------
    global_range : float, default 2
        Total spread (maximum - minimum) of true lambda_r (location
        shift) values across facet elements. Renamed from MFRM_Sim's own
        facet_range for this and the other two stretch models -- lambda_r
        is the parameter that collapses this representation to 'global'
        when every facet element's omega_r == 1, and 'global_range' pairs
        directly with 'stretch_range' below, matching the global_/stretch
        attribute alias naming (see MFRM._PARAM_ALIASES).
    stretch_range : float or (lowest, highest), default 1
        Total spread of true omega_r values around 1 (analogous to
        global_range for lambda_r).
        A pair (lowest, highest) instead draws each facet element's
        omega_r between the bounds, which may be any real numbers (0 and
        negative values included), lowest <= highest.
    manual_lambda : array-like or None, default None
        Custom true lambda_r values, length no_of_facet_elements. If None,
        drawn the same way MFRM_Sim's own facet_range scales locations.
    manual_omega : array-like or None, default None
        Custom true omega_r values, length no_of_facet_elements. If None,
        drawn as 1 + a stretch_range-scaled, zero-mean deviation.
    seed : int or None, default None
        Seed for both lambda_r/omega_r generation and everything else, via
        the preliminary and final MFRM_Sim calls. Pass an int for full
        reproducibility. This seed only affects this simulation's own local
        random generator, not the global numpy.random.seed() state -- a
        prior call to numpy.random.seed(N) has no effect on this
        simulation's output.

    See MFRM_Sim for all other parameter docs -- all forwarded unchanged
    to both the preliminary and final sims.

    Attributes set (in addition to the usual MFRM_Sim attributes)
    ---------------------------------------------------------------
    lambda_ : pandas.Series
        True generating lambda_r (location shift) per facet element.
    omega : pandas.Series
        True generating omega_r (item-difficulty stretch) per facet element.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        global_range=2,
        stretch_range=1,
        manual_lambda=None,
        manual_omega=None,
        seed=None,
        **kw,
    ):
        if "facet_range" in kw:
            raise TypeError(
                "MFRM_Sim_PseudoHalo.__init__() got an unexpected keyword "
                "argument 'facet_range' -- renamed to 'global_range' (it "
                "controls the spread of true lambda_r values, not "
                "MFRM_Sim's own facet_range, which would otherwise silently "
                "absorb this into the discarded preliminary simulation)."
            )
        anchors = kw.pop("anchors", None)
        neutral_anchors = kw.pop("neutral_anchors", False)
        prelim = MFRM_Sim(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="items",
            seed=seed,
            _params_only=True,
            **kw,
        )

        rng = np.random.default_rng(seed)
        R = prelim.no_of_facet_elements
        neutral = MFRM_Sim._neutral_mask(prelim.facet_names, anchors, neutral_anchors)

        if manual_lambda is not None:
            assert (
                len(manual_lambda) == R
            ), "Length of manual_lambda must match number of facet_elements."
            lam = np.array(manual_lambda, dtype=float)
        else:
            lam = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            lam *= _spread_factor(global_range, lam)
            if not neutral.any():
                lam -= lam.mean()
            lam[neutral] = 0.0

        if manual_omega is not None:
            assert (
                len(manual_omega) == R
            ), "Length of manual_omega must match number of facet_elements."
            omega = np.array(manual_omega, dtype=float)
        else:
            omega = _draw_multipliers(rng, R, stretch_range, centre=not neutral.any())
            omega[neutral] = 1.0

        delta = prelim.items.values
        implied_raters = {
            r: {
                item: float(lam[i] + (omega[i] - 1.0) * delta[j])
                for j, item in enumerate(prelim.item_names)
            }
            for i, r in enumerate(prelim.facet_names)
        }

        # Already captured into prelim (and re-supplied explicitly below) --
        # drop from kw so a caller who passed these directly doesn't hit a
        # "got multiple values for keyword argument" TypeError.
        for key in (
            "manual_items",
            "manual_thresholds",
            "manual_persons",
            "manual_item_names",
            "manual_person_names",
            "manual_facet_names",
            "manual_raters",
        ):
            kw.pop(key, None)

        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=prelim.no_of_facet_elements,
            max_score=prelim.max_score,
            model="items",
            manual_items=prelim.items.values,
            manual_thresholds=prelim.thresholds.values,
            manual_persons=prelim.persons.values,
            manual_person_names=list(prelim.person_names),
            manual_item_names=list(prelim.item_names),
            manual_facet_names=list(prelim.facet_names),
            manual_raters=implied_raters,
            seed=seed,
            **kw,
        )
        self.model = "pseudo_halo"
        self.lambda_ = pd.Series(lam, index=self.facet_names)
        self.omega = pd.Series(omega, index=self.facet_names)
        self._report_stretch_anchor_frame(anchors, neutral_anchors)

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega, matching MFRM's own
        global_{model}/stretch_{model} attribute aliases (see
        MFRM._PARAM_ALIASES) -- 'global_' for 'lambda_' (no model suffix
        needed here, since this sim class is already model-specific;
        trailing underscore because 'global', like 'lambda', is also a
        Python keyword), 'stretch' for 'omega'. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        if name == "global_":
            return self.lambda_
        if name == "stretch":
            return self.omega
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class MFRM_Sim_Bistretch(MFRM_Sim):
    """
    MFRM simulation -- "bistretch" parameterisation, a 3-parameter
    (location shift lambda_r, item-stretch omega_items_r, threshold-stretch
    omega_thresholds_r) restriction of the fully-free 'bivector'
    parameterisation -- combines MFRM_Sim_PseudoHalo's item-stretch axis
    and MFRM_Sim_Centrality's threshold-stretch axis into a single
    facet_element effect.

    Thin wrapper: generates a preliminary MFRM_Sim(model='bivector') to
    get realistic item locations/thresholds/persons, draws true
    lambda_r/omega_items_r/omega_thresholds_r per facet element (e.g. rater), computes the
    implied item_effects table via mu_ri = lambda_r + (omega_items_r - 1)
    * delta_i (bivector's item axis carries the facet_element's overall
    level, exactly as in MFRM_Sim_PseudoHalo) and the implied
    threshold_effects table via lambda_rk = (omega_thresholds_r - 1) *
    tau_k (bivector's threshold axis is already zero-sum shape-only, so
    no lambda_r term here -- unlike MFRM_Sim_Centrality, where lambda_r
    carries the level because 'thresholds' has no separate item axis to
    carry it instead), then delegates to the full
    MFRM_Sim(model='bivector', manual_item_effects=...,
    manual_threshold_effects=...) machinery (fixed to the same
    item/threshold/person values as the preliminary sim) for everything
    else. All the actual generative machinery is reused unchanged; only
    lambda_r/omega_items_r/omega_thresholds_r and their implied effect
    tables are new.

    See MFRM._derive_stretch_model / MFRM.calibrate_bistretch for the
    corresponding closed-form estimation side.

    Parameters
    ----------
    global_range : float, default 2
        Total spread (maximum - minimum) of true lambda_r (location
        shift) values across facet elements. Renamed from MFRM_Sim's own
        facet_range for this and the other two stretch models -- lambda_r
        is the parameter that collapses this representation to 'global'
        when every facet element's omega_items_r == omega_thresholds_r == 1, and
        'global_range' pairs directly with 'stretch_range' below,
        matching the global_/stretch_items/stretch_thresholds attribute
        alias naming (see MFRM._PARAM_ALIASES).
    stretch_range : float or (lowest, highest), default 1
        Total spread of true omega_items_r and omega_thresholds_r values
        around 1 (analogous to global_range for lambda_r); used for both
        axes unless overridden by item_stretch_range /
        threshold_stretch_range. A pair (lowest, highest) instead draws each
        facet element's multipliers between the bounds, which may be any
        real numbers (0 and negative values included), lowest <= highest.
    item_stretch_range, threshold_stretch_range : float, pair or None, default None
        As stretch_range, for omega_items_r / omega_thresholds_r only.
    manual_lambda : array-like or None, default None
        Custom true lambda_r values, length no_of_facet_elements. If None,
        drawn the same way MFRM_Sim's own facet_range scales locations.
    manual_omega_items : array-like or None, default None
        Custom true omega_items_r values, length no_of_facet_elements. If
        None, drawn as 1 + a stretch_range-scaled, zero-mean deviation.
    manual_omega_thresholds : array-like or None, default None
        Custom true omega_thresholds_r values, length no_of_facet_elements.
        If None, drawn as 1 + a stretch_range-scaled, zero-mean deviation
        (independently of omega_items_r).
    seed : int or None, default None
        Seed for lambda_r/omega_items_r/omega_thresholds_r generation and
        everything else, via the preliminary and final MFRM_Sim calls.
        Pass an int for full reproducibility. This seed only affects this
        simulation's own local random generator, not the global
        numpy.random.seed() state -- a prior call to numpy.random.seed(N)
        has no effect on this simulation's output.

    See MFRM_Sim for all other parameter docs (item_range, person_sd,
    missing, manual_items, manual_thresholds, manual_persons, etc. --
    all forwarded unchanged to both the preliminary and final sims).

    Attributes set (in addition to the usual MFRM_Sim attributes)
    ---------------------------------------------------------------
    lambda_ : pandas.Series
        True generating lambda_r (location shift) per facet element.
    omega_items : pandas.Series
        True generating omega_items_r (item-difficulty stretch) per facet element.
    omega_thresholds : pandas.Series
        True generating omega_thresholds_r (threshold stretch) per facet element.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        global_range=2,
        stretch_range=1,
        item_stretch_range=None,
        threshold_stretch_range=None,
        manual_lambda=None,
        manual_omega_items=None,
        manual_omega_thresholds=None,
        seed=None,
        **kw,
    ):
        if "facet_range" in kw:
            raise TypeError(
                "MFRM_Sim_Bistretch.__init__() got an unexpected keyword "
                "argument 'facet_range' -- renamed to 'global_range' (it "
                "controls the spread of true lambda_r values, not "
                "MFRM_Sim's own facet_range, which would otherwise silently "
                "absorb this into the discarded preliminary simulation)."
            )
        anchors = kw.pop("anchors", None)
        neutral_anchors = kw.pop("neutral_anchors", False)
        prelim = MFRM_Sim(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="bivector",
            seed=seed,
            _params_only=True,
            **kw,
        )

        rng = np.random.default_rng(seed)
        R = prelim.no_of_facet_elements
        neutral = MFRM_Sim._neutral_mask(prelim.facet_names, anchors, neutral_anchors)

        if manual_lambda is not None:
            assert (
                len(manual_lambda) == R
            ), "Length of manual_lambda must match number of facet_elements."
            lam = np.array(manual_lambda, dtype=float)
        else:
            lam = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            lam *= _spread_factor(global_range, lam)
            if not neutral.any():
                lam -= lam.mean()
            lam[neutral] = 0.0

        if manual_omega_items is not None:
            assert (
                len(manual_omega_items) == R
            ), "Length of manual_omega_items must match number of facet_elements."
            omega_items = np.array(manual_omega_items, dtype=float)
        else:
            omega_items = _draw_multipliers(
                rng,
                R,
                stretch_range if item_stretch_range is None else item_stretch_range,
                centre=not neutral.any(),
            )
            omega_items[neutral] = 1.0

        if manual_omega_thresholds is not None:
            assert (
                len(manual_omega_thresholds) == R
            ), "Length of manual_omega_thresholds must match number of facet_elements."
            omega_thresholds = np.array(manual_omega_thresholds, dtype=float)
        else:
            omega_thresholds = _draw_multipliers(
                rng,
                R,
                stretch_range if threshold_stretch_range is None else threshold_stretch_range,
                centre=not neutral.any(),
            )
            omega_thresholds[neutral] = 1.0

        delta = prelim.items.values
        tau = prelim.thresholds.values

        implied_item_effects = pd.DataFrame(
            [lam[i] + (omega_items[i] - 1.0) * delta for i in range(R)],
            index=prelim.facet_names,
            columns=prelim.item_names,
        )
        implied_threshold_effects = pd.DataFrame(
            [(omega_thresholds[i] - 1.0) * tau for i in range(R)],
            index=prelim.facet_names,
            columns=range(1, prelim.max_score + 1),
        )

        # Already captured into prelim (and re-supplied explicitly below) --
        # drop from kw so a caller who passed these directly doesn't hit a
        # "got multiple values for keyword argument" TypeError.
        for key in (
            "manual_items",
            "manual_thresholds",
            "manual_persons",
            "manual_item_names",
            "manual_person_names",
            "manual_facet_names",
            "manual_item_effects",
            "manual_threshold_effects",
        ):
            kw.pop(key, None)

        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=prelim.no_of_facet_elements,
            max_score=prelim.max_score,
            model="bivector",
            manual_items=prelim.items.values,
            manual_thresholds=prelim.thresholds.values,
            manual_persons=prelim.persons.values,
            manual_person_names=list(prelim.person_names),
            manual_item_names=list(prelim.item_names),
            manual_facet_names=list(prelim.facet_names),
            manual_item_effects=implied_item_effects,
            manual_threshold_effects=implied_threshold_effects,
            seed=seed,
            **kw,
        )
        self.model = "bistretch"
        self.lambda_ = pd.Series(lam, index=self.facet_names)
        self.omega_items = pd.Series(omega_items, index=self.facet_names)
        self.omega_thresholds = pd.Series(omega_thresholds, index=self.facet_names)
        self._report_stretch_anchor_frame(anchors, neutral_anchors)

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega_items/omega_thresholds,
        matching MFRM's own global_{model}/stretch_items_{model}/
        stretch_thresholds_{model} attribute aliases (see
        MFRM._PARAM_ALIASES) -- 'global_' for 'lambda_' (no model suffix
        needed here, since this sim class is already model-specific;
        trailing underscore because 'global', like 'lambda', is also a
        Python keyword), 'stretch_items'/'stretch_thresholds' for
        'omega_items'/'omega_thresholds'. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        if name == "global_":
            return self.lambda_
        if name == "stretch_items":
            return self.omega_items
        if name == "stretch_thresholds":
            return self.omega_thresholds
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class MFRM_Sim_Mixed(MFRM_Sim):
    """
    MFRM simulation -- each facet element (e.g. rater) on its own representation.

    Convenience subclass of MFRM_Sim with model='mixed' fixed: every facet
    element is generated under one of the eight representations ('global',
    'items', 'thresholds', 'bivector', 'matrix', 'centrality', 'pseudo_halo',
    'bistretch'), sharing the item locations and thresholds. global_range is
    MFRM_Sim's facet_range under the name the stretch classes use: the spread
    of the facet elements' global effects. See MFRM_Sim for rater_models,
    candidate_models, proportions, the other ranges, anchors and the
    attributes set.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        rater_models=None,
        candidate_models=None,
        proportions=None,
        global_range=2,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='mixed' fixed. See MFRM_Sim for full documentation."""
        if "facet_range" in kw:
            raise ValueError("MFRM_Sim_Mixed takes global_range=, not facet_range=.")
        if "model" in kw:
            raise ValueError("MFRM_Sim_Mixed fixes model='mixed'; use MFRM_Sim for other representations.")
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="mixed",
            facet_range=global_range,
            rater_models=rater_models,
            candidate_models=candidate_models,
            proportions=proportions,
            **kw,
        )
        self.global_range = global_range
