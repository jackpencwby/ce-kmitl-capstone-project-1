"""Time-based splitting: main holdout + expanding walk-forward folds.

All splits are by calendar date only, shared across every station and every
experiment (Experimental_Plan.md sections 3, 4). No shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config


@dataclass
class Fold:
    """One expanding-window fold: train dates < train_end, validate in window."""

    index: int
    train_end: pd.Timestamp        # exclusive upper bound of training dates
    valid_start: pd.Timestamp      # inclusive
    valid_end: pd.Timestamp        # exclusive


def date_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the main train / validation / test holdout."""
    d = df[config.DATE_COL]
    vstart = pd.Timestamp(config.VALIDATION_START)
    tstart = pd.Timestamp(config.TEST_START)
    return {
        "train": d < vstart,
        "validation": (d >= vstart) & (d < tstart),
        "test": d >= tstart,
    }


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = config.N_WALK_FORWARD_FOLDS,
    embargo_days: int = config.EMBARGO_DAYS,
) -> list[Fold]:
    """Build expanding-window folds inside the Train+Validation span.

    The validation region [VALIDATION_START, TEST_START) is divided into
    ``n_folds`` equal, contiguous windows. Each fold trains on everything up
    to ``embargo_days`` before its validation window start, so target label
    windows do not overlap the validation origin (plan section 15).
    """
    vstart = pd.Timestamp(config.VALIDATION_START)
    tstart = pd.Timestamp(config.TEST_START)
    total_days = (tstart - vstart).days
    if total_days < n_folds:
        n_folds = max(1, total_days)
    edges = [vstart + pd.Timedelta(days=int(round(i * total_days / n_folds)))
             for i in range(n_folds + 1)]

    folds: list[Fold] = []
    for i in range(n_folds):
        v0, v1 = edges[i], edges[i + 1]
        train_end = v0 - pd.Timedelta(days=embargo_days)
        folds.append(Fold(index=i + 1, train_end=train_end, valid_start=v0, valid_end=v1))
    return folds


def fold_masks(df: pd.DataFrame, fold: Fold) -> tuple[pd.Series, pd.Series]:
    """(train_mask, valid_mask) for a given fold."""
    d = df[config.DATE_COL]
    train_mask = d < fold.train_end
    valid_mask = (d >= fold.valid_start) & (d < fold.valid_end)
    return train_mask, valid_mask


def fold_manifest(folds: list[Fold]) -> pd.DataFrame:
    """Tabular description of folds for artifacts/fold_manifest.csv."""
    return pd.DataFrame([
        {
            "fold": f.index,
            "train_end_exclusive": f.train_end.date().isoformat(),
            "valid_start": f.valid_start.date().isoformat(),
            "valid_end_exclusive": f.valid_end.date().isoformat(),
        }
        for f in folds
    ])


def residual_oof_folds(df: pd.DataFrame, train_end: pd.Timestamp,
                       n_folds: int = 4) -> list[Fold]:
    """Expanding OOF folds inside the training window for local correction."""
    start = pd.Timestamp(df[config.DATE_COL].min()).normalize()
    end = pd.Timestamp(train_end).normalize()
    warmup = start + pd.Timedelta(days=180)
    if warmup >= end:
        return []
    span = (end - warmup).days
    edges = [warmup + pd.Timedelta(days=round(i * span / n_folds))
             for i in range(n_folds + 1)]
    return [Fold(i + 1, edges[i] - pd.Timedelta(days=config.EMBARGO_DAYS),
                 edges[i], edges[i + 1]) for i in range(n_folds)]
