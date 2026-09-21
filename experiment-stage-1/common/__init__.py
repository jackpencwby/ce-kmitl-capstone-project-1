"""Shared library for PM2.5 Stage 1 experiments.

This package holds everything the individual experiment scripts
(E1_1.py, E1_2.py, ..., E4_4.py) share so that every run uses the same
data, splits, features, metrics and artifact layout. Only the one factor
under test changes between scripts (see Experimental_Plan.md section 10-11).
"""

from . import config, data, features, splits, metrics, artifacts, cli  # noqa: F401
