"""
RaschPy — Rasch measurement in Python.

Public API surface:

Models
------
    SLM   — Simple Logistic Model (dichotomous Rasch)
    PCM   — Partial Credit Model
    RSM   — Rating Scale Model
    MFRM  — Many-Facet Rasch Model

Loaders
-------
    loadup_slm              — load SLM data from CSV/Excel/JSON
    loadup_pcm              — load PCM data
    loadup_rsm              — load RSM data
    loadup_mfrm_single      — load MFRM data from a single (Rater, Person) file
    loadup_mfrm_xlsx_tabs   — load MFRM data from an Excel workbook (one sheet per rater)
    loadup_mfrm_multiple    — load MFRM data from separate files per rater
    loadup_exogenous        — load person-level DIF covariates (Gender, L1, etc.)

Simulation
----------
    SLM_Sim              — simulate SLM data
    PCM_Sim              — simulate PCM data
    RSM_Sim              — simulate RSM data
    MFRM_Sim             — simulate MFRM data (any parameterisation)
    MFRM_Sim_Global      — MFRM simulation, global facet effects
    MFRM_Sim_Items       — MFRM simulation, per-item facet effects
    MFRM_Sim_Thresholds  — MFRM simulation, per-threshold facet effects
    MFRM_Sim_Bivector    — MFRM simulation, additive (item + threshold) facet effects
    MFRM_Sim_Matrix      — MFRM simulation, full rater × item × facet effects
    MFRM_Sim_Centrality  — MFRM simulation, Jin & Wang (2018) rater centrality/extremity
    MFRM_Sim_PseudoHalo  — MFRM simulation, (pseudo-)halo-effect item-difficulty stretch
    MFRM_Sim_Bistretch   — MFRM simulation, combined threshold + item stretch (bivector restriction)
"""

from raschpy.slm import SLM
from raschpy.pcm import PCM
from raschpy.rsm import RSM
from raschpy.mfrm import MFRM

from raschpy.loaders import (
    loadup_exogenous,
    loadup_slm,
    loadup_pcm,
    loadup_rsm,
    loadup_mfrm_single,
    loadup_mfrm_xlsx_tabs,
    loadup_mfrm_multiple,
)

from raschpy.simulation.slm_sim import SLM_Sim
from raschpy.simulation.pcm_sim import PCM_Sim
from raschpy.simulation.rsm_sim import RSM_Sim
from raschpy.simulation.mfrm_sim import (
    MFRM_Sim,
    MFRM_Sim_Global,
    MFRM_Sim_Items,
    MFRM_Sim_Thresholds,
    MFRM_Sim_Bivector,
    MFRM_Sim_Matrix,
    MFRM_Sim_Centrality,
    MFRM_Sim_PseudoHalo,
    MFRM_Sim_Bistretch,
)

__version__ = "1.2.1"
__author__ = "Mark Elliott"

__all__ = [
    # Models
    "SLM",
    "PCM",
    "RSM",
    "MFRM",
    # Loaders
    "loadup_exogenous",
    "loadup_slm",
    "loadup_pcm",
    "loadup_rsm",
    "loadup_mfrm_single",
    "loadup_mfrm_xlsx_tabs",
    "loadup_mfrm_multiple",
    # Simulation
    "SLM_Sim",
    "PCM_Sim",
    "RSM_Sim",
    "MFRM_Sim",
    "MFRM_Sim_Global",
    "MFRM_Sim_Items",
    "MFRM_Sim_Thresholds",
    "MFRM_Sim_Bivector",
    "MFRM_Sim_Matrix",
    "MFRM_Sim_Centrality",
    "MFRM_Sim_PseudoHalo",
    "MFRM_Sim_Bistretch",
]
