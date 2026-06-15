import itertools
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

from raschpy.simulation.base_sim import Rasch_Sim
from raschpy.mfrm import MFRM


class MFRM_Sim(Rasch_Sim):
    '''
    Generates simulated data for the Many-Facet Rasch Model (MFRM)
    under four rater severity parameterisations:
      'global'     — scalar severity per rater
      'items'      — severity vector per (rater, item)
      'thresholds' — severity vector per (rater, threshold)
      'matrix'     — full severity tensor per (rater, item, threshold)

    Parameters
    ----------
    no_of_items     : int
    no_of_persons   : int
    no_of_raters    : int
    max_score       : int
    model           : str, one of 'global', 'items', 'thresholds', 'matrix'
    item_range      : float, spread of item difficulties
    rater_range     : float, spread of rater severities
    category_base   : float, base width of rating categories
    person_sd       : float, SD of person ability distribution
    max_disorder    : float, maximum threshold disorder (0 = ordered)
    offset          : float, shift of person distribution relative to items
    missing         : float, proportion of missing responses [0, 1)
    shared_missing  : bool, if True the same persons are missing across raters
    manual_abilities    : array-like, optional
    manual_diffs        : array-like, optional
    manual_thresholds   : array-like, optional
    manual_severities   : dict or array-like, optional (structure depends on model)
    manual_person_names : list of str, optional
    manual_item_names   : list of str, optional
    manual_rater_names  : list of str, optional
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 no_of_raters,
                 max_score,
                 model='global',
                 item_range=2,
                 rater_range=2,
                 category_base=1,
                 person_sd=1.5,
                 max_disorder=0,
                 offset=0,
                 missing=0,
                 shared_missing=True,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None,
                 manual_severities=None,
                 manual_person_names=None,
                 manual_item_names=None,
                 manual_rater_names=None):

        if model not in ('global', 'items', 'thresholds', 'matrix'):
            raise ValueError(f"model must be one of 'global', 'items', 'thresholds', 'matrix'")

        self.model          = model
        self.no_of_items    = int(no_of_items)
        self.no_of_persons  = int(no_of_persons)
        self.no_of_raters   = int(no_of_raters)
        self.max_score      = max_score
        self.item_range     = item_range
        self.rater_range    = rater_range
        self.category_base  = category_base
        self.person_sd      = person_sd
        self.max_disorder   = max_disorder
        self.offset         = offset
        self.missing        = missing
        self.shared_missing = shared_missing

        # ------------------------------------------------------------------
        # Persons
        # ------------------------------------------------------------------
        if manual_person_names is not None:
            assert len(manual_person_names) == self.no_of_persons, \
                'Length of person names must match number of persons.'
            self.persons = manual_person_names
        else:
            self.persons = [f'Person_{p + 1}' for p in range(self.no_of_persons)]

        if manual_abilities is not None:
            assert len(manual_abilities) == self.no_of_persons, \
                'Length of manual abilities must match number of persons.'
            abilities = np.array(manual_abilities)
        else:
            abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            abilities -= abilities.mean()
            abilities += self.offset

        self.abilities = pd.Series(
            {person: ab for person, ab in zip(self.persons, abilities)}
        )

        # ------------------------------------------------------------------
        # Items
        # ------------------------------------------------------------------
        if manual_item_names is not None:
            assert len(manual_item_names) == self.no_of_items, \
                'Length of item names must match number of items.'
            self.items = manual_item_names
        else:
            self.items = [f'Item_{i + 1}' for i in range(self.no_of_items)]

        if manual_diffs is not None:
            assert len(manual_diffs) == self.no_of_items, \
                'Length of manual difficulties must match number of items.'
            diffs = np.array(manual_diffs)
        else:
            diffs = np.random.uniform(0, 1, self.no_of_items)
            diffs *= self.item_range / (diffs.max() - diffs.min())
            diffs -= diffs.mean()

        self.diffs = pd.Series(
            {item: d for item, d in zip(self.items, diffs)}
        )

        # ------------------------------------------------------------------
        # Thresholds (shared RSM structure across all four models)
        # ------------------------------------------------------------------
        if manual_thresholds is not None:
            assert len(manual_thresholds) == self.max_score + 1, \
                'Number of manual thresholds must be max score plus 1.'
            assert manual_thresholds[0] == 0, \
                'First threshold must be zero.'
            assert sum(manual_thresholds) == 0, \
                'Manual thresholds must sum to zero.'
            self.thresholds = np.array(manual_thresholds)
        else:
            cat_widths = np.random.uniform(
                self.max_disorder,
                2 * self.category_base - self.max_disorder,
                self.max_score
            )
            thresholds = np.array([cat_widths[:k].sum() for k in range(self.max_score)])
            thresholds -= thresholds.mean()
            self.thresholds = np.insert(thresholds, 0, 0.0)

        # ------------------------------------------------------------------
        # Raters
        # ------------------------------------------------------------------
        if manual_rater_names is not None:
            assert len(manual_rater_names) == self.no_of_raters, \
                'Length of rater names must match number of raters.'
            self.raters = manual_rater_names
        else:
            self.raters = [f'Rater_{r + 1}' for r in range(self.no_of_raters)]

        # ------------------------------------------------------------------
        # Severities (model-specific)
        # ------------------------------------------------------------------
        self.severities = self._generate_severities(manual_severities)

        # ------------------------------------------------------------------
        # Category probabilities
        # ------------------------------------------------------------------
        self.cat_probs = self._compute_cat_probs()

        # ------------------------------------------------------------------
        # Scores + missing data
        # ------------------------------------------------------------------
        scoring_randoms = {
            rater: pd.DataFrame(
                self.randoms(), columns=self.items, index=self.persons
            )
            for rater in self.raters
        }
        scoring_randoms = pd.concat(
            scoring_randoms.values(), keys=scoring_randoms.keys()
        )

        self.scores = sum(
            scoring_randoms < sum(
                self.cat_probs[cat] for cat in range(c, self.max_score + 1)
            )
            for c in range(1, self.max_score + 1)
        )

        if shared_missing:
            missing_randoms = pd.DataFrame(
                self.randoms(), columns=self.items, index=self.persons
            )
            missing_randoms = pd.concat(
                {rater: missing_randoms for rater in self.raters},
                keys=self.raters
            )
        else:
            missing_randoms = pd.concat(
                {rater: pd.DataFrame(
                    self.randoms(), columns=self.items, index=self.persons
                ) for rater in self.raters},
                keys=self.raters
            )

        self.scores[missing_randoms < self.missing] = np.nan

    # ------------------------------------------------------------------
    # Severity generation
    # ------------------------------------------------------------------

    def _generate_severities(self, manual_severities):
        '''Generate or validate rater severity parameters for the given model.'''

        if self.model == 'global':
            if manual_severities is not None:
                assert len(manual_severities) == self.no_of_raters, \
                    'Length of manual severities must match number of raters.'
                sev = np.array(manual_severities)
            else:
                sev = truncnorm.rvs(-1.96, 1.96, size=self.no_of_raters)
                sev *= self.rater_range / (sev.max() - sev.min())
                sev -= sev.mean()
            return pd.Series(
                {rater: s for rater, s in zip(self.raters, sev)}
            )

        elif self.model == 'items':
            if manual_severities is not None:
                assert len(manual_severities) == self.no_of_raters, \
                    'Length of manual severities must match number of raters.'
                return manual_severities
            else:
                sev = np.array([
                    truncnorm.rvs(-1.96, 1.96, size=self.no_of_items)
                    for _ in range(self.no_of_raters)
                ])  # (R, I)
                sev *= self.rater_range / (sev.max() - sev.min())
                # Centre per item (column)
                for i in range(self.no_of_items):
                    sev[:, i] -= sev[:, i].mean()
                return {
                    rater: {item: sev[r, i]
                            for i, item in enumerate(self.items)}
                    for r, rater in enumerate(self.raters)
                }

        elif self.model == 'thresholds':
            if manual_severities is not None:
                assert len(manual_severities) == self.no_of_raters, \
                    'Length of manual severities must match number of raters.'
                return manual_severities
            else:
                sev = np.array([
                    truncnorm.rvs(-1.96, 1.96, size=self.max_score)
                    for _ in range(self.no_of_raters)
                ])  # (R, K)
                sev *= self.rater_range / (sev.max() - sev.min())
                sev -= sev.mean()
                sev = np.insert(sev, 0, 0.0, axis=1)  # prepend zero slot
                return {
                    rater: sev[r, :]
                    for r, rater in enumerate(self.raters)
                }

        elif self.model == 'matrix':
            if manual_severities is not None:
                assert len(manual_severities) == self.no_of_raters, \
                    'Length of manual severities must match number of raters.'
                return manual_severities
            else:
                sev = np.array([
                    [truncnorm.rvs(-1.96, 1.96, size=self.max_score)
                     for _ in range(self.no_of_items)]
                    for _ in range(self.no_of_raters)
                ])  # (R, I, K)
                sev *= self.rater_range / (sev.max() - sev.min())
                # Centre per (rater, item) cell
                for r in range(self.no_of_raters):
                    for i in range(self.no_of_items):
                        sev[r, i, :] -= sev[r, i, :].mean()
                sev = np.insert(sev, 0, 0.0, axis=2)  # prepend zero slot
                return {
                    rater: {item: sev[r, i, :]
                            for i, item in enumerate(self.items)}
                    for r, rater in enumerate(self.raters)
                }

    # ------------------------------------------------------------------
    # Category probability computation
    # ------------------------------------------------------------------

    def _compute_cat_probs(self):
        '''Compute category probability DataFrames for all raters and categories.'''

        if self.model == 'global':
            c_p_df = pd.DataFrame(
                {item: self.abilities - self.diffs[item] for item in self.items}
            )
            cat_probs = {
                cat: {
                    rater: (cat * (c_p_df - self.severities[rater])
                            - self.thresholds[:cat + 1].sum())
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == 'items':
            c_p_df = {
                rater: pd.DataFrame({
                    item: self.abilities - self.diffs[item] - self.severities[rater][item]
                    for item in self.items
                })
                for rater in self.raters
            }
            cat_probs = {
                cat: {
                    rater: (cat * c_p_df[rater] - self.thresholds[:cat + 1].sum())
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == 'thresholds':
            c_p_df = pd.DataFrame(
                {item: self.abilities - self.diffs[item] for item in self.items}
            )
            cat_probs = {
                cat: {
                    rater: (cat * c_p_df
                            - self.thresholds[:cat + 1].sum()
                            - self.severities[rater][:cat + 1].sum())
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == 'matrix':
            c_p_df = pd.DataFrame(
                {item: self.abilities - self.diffs[item] for item in self.items}
            )
            cat_probs = {
                cat: {
                    rater: (cat * c_p_df - self.thresholds[:cat + 1].sum())
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }
            # Apply per-(rater, item, threshold) severity
            for cat in range(self.max_score + 1):
                for rater in self.raters:
                    for item in self.items:
                        cat_probs[cat][rater][item] -= (
                            self.severities[rater][item][:cat + 1].sum()
                        )

        # Concatenate across raters, exponentiate, normalise
        for cat in range(self.max_score + 1):
            cat_probs[cat] = pd.concat(
                cat_probs[cat].values(), keys=cat_probs[cat].keys()
            )
            cat_probs[cat] = np.exp(cat_probs[cat])

        den = sum(cat_probs[cat] for cat in range(self.max_score + 1))
        for cat in range(self.max_score + 1):
            cat_probs[cat] /= den

        return cat_probs

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def rename_rater(self, old, new):
        if old == new:
            print('New rater name is the same as old rater name.')
        elif new in self.raters:
            print('New rater name is a duplicate of an existing rater name')
        if old not in self.raters:
            print(f'Old rater name "{old}" not found in data. Please check')
        elif not isinstance(new, str):
            print('Rater names must be strings')
        else:
            new_names = [new if r == old else r for r in self.raters]
            self.rename_raters_all(new_names)

    def rename_raters_all(self, new_names):
        if len(new_names) != len(set(new_names)):
            print('List of new rater names contains duplicates.')
        elif len(new_names) != self.no_of_raters:
            print(f'Incorrect number of rater names. {len(new_names)} in list, '
                  f'{self.no_of_raters} raters in data.')
        elif not all(isinstance(n, str) for n in new_names):
            print('Rater names must be strings')
        else:
            df_dict = {new: self.scores.xs(old)
                       for old, new in zip(self.raters, new_names)}
            self.scores = pd.concat(df_dict.values(), keys=df_dict.keys())
        self.raters = self.scores.index.get_level_values(0).unique().tolist()

    def rename_person(self, old, new):
        if old == new:
            print('New person name is the same as old person name.')
        elif new in self.scores.index.get_level_values(1):
            print('New person name is a duplicate of an existing person name')
        if old not in self.scores.index.get_level_values(1):
            print(f'Old person name "{old}" not found in data. Please check')
        elif not isinstance(new, str):
            print('Person names must be strings')
        else:
            self.scores.rename(index={old: new}, inplace=True)
        self.persons = self.scores.index.get_level_values(1).unique().tolist()

    def rename_persons_all(self, new_names):
        old_names = self.scores.index.get_level_values(1)
        if len(new_names) != len(set(new_names)):
            print('List of new person names contains duplicates.')
        elif len(new_names) != self.no_of_persons:
            print(f'Incorrect number of person names.')
        elif not all(isinstance(n, str) for n in new_names):
            print('Person names must be strings')
        else:
            self.scores.rename(
                index={old: new for old, new in zip(old_names, new_names)},
                inplace=True
            )
        self.persons = self.scores.index.get_level_values(1).unique().tolist()


# ------------------------------------------------------------------
# Backwards-compatible subclass aliases
# ------------------------------------------------------------------

class MFRM_Sim_Global(MFRM_Sim):
    '''MFRM simulation — global (scalar) rater severity parameterisation.'''
    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(no_of_items, no_of_persons, no_of_raters, max_score,
                         model='global', **kw)


class MFRM_Sim_Items(MFRM_Sim):
    '''MFRM simulation — items (vector per rater×item) severity parameterisation.'''
    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(no_of_items, no_of_persons, no_of_raters, max_score,
                         model='items', **kw)


class MFRM_Sim_Thresholds(MFRM_Sim):
    '''MFRM simulation — thresholds (vector per rater×threshold) severity parameterisation.'''
    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(no_of_items, no_of_persons, no_of_raters, max_score,
                         model='thresholds', **kw)


class MFRM_Sim_Matrix(MFRM_Sim):
    '''MFRM simulation — matrix (full rater×item×threshold tensor) severity parameterisation.'''
    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(no_of_items, no_of_persons, no_of_raters, max_score,
                         model='matrix', **kw)