"""Tree-count selection on training history only, followed by a fresh refit."""
from __future__ import annotations

import pandas as pd


def fit_temporal_selection(factory, prefer_gpu, algorithm, X, y, dates,
                           horizon: int, patience: int):
    """No outer validation/test labels enter this interface.

    Use the final 15% of training dates for stopping, purge overlapping
    training labels, then refit a fresh estimator on every supplied row.
    Short histories retain the configured budget and record that decision.
    """
    candidate = factory(prefer_gpu=prefer_gpu)
    budget = int(candidate.get_params()['n_estimators'])
    dates = pd.to_datetime(dates)
    cutoff = dates.quantile(.85)
    fit_mask = dates + pd.Timedelta(days=horizon) < cutoff
    tail_mask = dates >= cutoff
    metadata = {'policy':'training_tail_85_15_refit', 'patience':patience,
                'requested_n_estimators':budget, 'selected_n_estimators':budget,
                'training_rows':len(X), 'status':'insufficient_history'}
    if fit_mask.sum() >= 50 and tail_mask.sum() >= 10:
        if algorithm == 'xgboost':
            candidate.set_params(early_stopping_rounds=patience, eval_metric='rmse')
            candidate.fit(X.loc[fit_mask], y.loc[fit_mask],
                          eval_set=[(X.loc[tail_mask], y.loc[tail_mask])], verbose=False)
            selected = int(candidate.best_iteration) + 1
        elif algorithm == 'lightgbm':
            from lightgbm import early_stopping, log_evaluation
            candidate.fit(X.loc[fit_mask], y.loc[fit_mask],
                          eval_set=[(X.loc[tail_mask], y.loc[tail_mask])], eval_metric='rmse',
                          callbacks=[early_stopping(patience, first_metric_only=True, verbose=False),
                                     log_evaluation(0)])
            selected = int(candidate.best_iteration_ or budget)
        else:
            raise ValueError(f'Unsupported temporal selection algorithm: {algorithm}')
        metadata.update(status='selected_and_refit', selected_n_estimators=selected,
                        fit_end=str(dates.loc[fit_mask].max()),
                        selection_start=str(dates.loc[tail_mask].min()),
                        selection_end=str(dates.loc[tail_mask].max()))
    fitted = factory(prefer_gpu=prefer_gpu)
    fitted.set_params(n_estimators=metadata['selected_n_estimators'])
    if algorithm == 'xgboost':
        fitted.set_params(early_stopping_rounds=None)
    fitted.fit(X, y)
    fitted.selection_metadata_ = metadata
    return fitted
