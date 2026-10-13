"""Experiments 13/14: Tawan XGBoost + Stage 1 station OOF tree residuals."""
from __future__ import annotations

import logging
import signal
from contextlib import ExitStack
from pathlib import Path
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import Tawan_run_stage2 as base
from xgb_checkpoint import RunCheckpoint, run_identity, atomic_json


def make_context(df, stations, spec, evaluation):
    context = base.make_context(df, stations, spec, evaluation)
    context['device'] = 'cuda:0' if spec.prefer_gpu else 'cpu'
    context['candidate_evaluator'] = fit_evaluate
    return context


def _fit(rows, targets, xcols, params, seed, budget, patience, multi, counter):
    """Select rounds on a purged training tail, then refit all supplied rows.

    The OOF scoring rows and outer validation/test labels never enter fitting
    or early stopping. The same policy applies to global and local trees.
    """
    rounds = budget
    if patience is not None:
        dates = rows[base.config.DATE_COL]
        cutoff = dates.quantile(.85)
        fit_mask = (dates + np.timedelta64(max(base.config.FORECAST_HORIZONS), 'D') < cutoff).to_numpy()
        stop_mask = (dates >= cutoff).to_numpy()
        if fit_mask.sum() >= 50 and stop_mask.sum() >= 10:
            probe = base._xgb(params, seed, budget, patience, multi)
            counter['total'] += 1
            probe.fit(rows.loc[fit_mask, xcols], targets[fit_mask],
                      eval_set=[(rows.loc[stop_mask, xcols], targets[stop_mask])], verbose=False)
            rounds = int(getattr(probe, 'best_iteration', budget - 1)) + 1
            del probe
    estimator = base._xgb(params, seed, rounds, None, multi)
    counter['total'] += 1
    estimator.fit(rows[xcols], targets, verbose=False)
    return estimator, rounds


def fit_evaluate(context: dict[str, Any], params: dict[str, Any], budget: int,
                 early_stopping_rounds: int | None, seed: int,
                 fit_counter: dict[str, int], keep_models: bool = False,
                 selected_rounds: dict[str, int] | None = None) -> dict[str, Any]:
    if context['evaluation'] == 'test' and selected_rounds is None:
        raise ValueError('Test refit requires validation-selected boosting rounds')
    spec = context['spec']
    params = {**params, 'device': context['device']}
    multi = spec.forecast_strategy == 'multi'
    train, score_rows = context['train_rows'], context['validation_rows']
    xcols = context['features']
    cutoff = context['fit_boundary']
    folds = base.splits.residual_oof_folds(train, cutoff)
    checkpoint = context.get('checkpoint')
    scope = checkpoint.scope({'evaluation': context['evaluation'], 'params': params,
                              'budget': budget, 'patience': early_stopping_rounds,
                              'seed': seed, 'keep_models': keep_models,
                              'selected_rounds': selected_rounds}) if checkpoint else None
    if scope and not keep_models:
        cached = scope.load_summary()
        if cached is not None:
            return cached

    def fit_unit(key, *args):
        if scope:
            return scope.fit(key, lambda: _fit(*args), context['device'])
        return _fit(*args)

    horizons = [0] if multi else list(base.config.FORECAST_HORIZONS)
    records, importances, blocks = [], [], []
    round_counts = {}
    for horizon in horizons:
        targets = base.TARGET_COLS if multi else f'target_t{horizon}'
        global_key = f'global:{horizon}'
        global_budget = selected_rounds[global_key] if selected_rounds is not None else budget
        patience = None if selected_rounds is not None else early_stopping_rounds
        global_model, used_rounds = fit_unit(global_key,
            train, train[targets].to_numpy(), xcols, params, seed,
            global_budget, patience, multi, fit_counter)
        round_counts[global_key] = used_rounds
        prediction = np.asarray(global_model.predict(score_rows[xcols])).copy()
        oof_rows, oof_targets = [], []
        for fold in folds:
            fit_mask, valid_mask = base.splits.fold_masks(train, fold)
            fit_mask &= train[base.config.DATE_COL] + np.timedelta64(max(base.config.FORECAST_HORIZONS), 'D') < fold.valid_start
            fold_train, fold_score = train.loc[fit_mask], train.loc[valid_mask]
            if fold_train.empty or fold_score.empty:
                continue
            fold_key = f'oof:{fold.index}:{horizon}'
            fold_budget = budget
            if selected_rounds is not None:
                fold_budget = selected_rounds.get(fold_key, global_budget)
                if fold_key not in selected_rounds:
                    base.LOGGER.warning('%s has no saved rounds; using selected global rounds %d',
                                        fold_key, global_budget)
            fold_model, fold_rounds = fit_unit(fold_key, fold_train, fold_train[targets].to_numpy(), xcols,
                                           params, seed, fold_budget, patience, multi, fit_counter)
            round_counts[fold_key] = fold_rounds
            oof_rows.append(fold_score)
            oof_targets.append(fold_score[targets].to_numpy() - fold_model.predict(fold_score[xcols]))
            del fold_model

        def record_model(model, group, role):
            if not keep_models:
                return
            records.append({'estimator': model, 'group': str(group), 'horizon': horizon,
                            'role': role, 'feature_columns': xcols,
                            'target_columns': base.TARGET_COLS if multi else [targets]})
            # Native vector-leaf importance is not supported by every build.
            try:
                importance = base.models._importance_frame(model, xcols, f'{role}_h{horizon}')
            except (AttributeError, ValueError, NotImplementedError):
                return
            if not importance.empty:
                importance['group'] = str(group)
                importances.append(importance)

        record_model(global_model, 'global', 'multi_output' if multi else 'direct')
        if oof_rows:
            local_rows = pd.concat(oof_rows, ignore_index=True)
            residual_targets = np.concatenate(oof_targets, axis=0)
            for station, station_rows in local_rows.groupby(base.config.STATION_ID_COL):
                score_mask = (score_rows[base.config.STATION_ID_COL] == station).to_numpy()
                if len(station_rows) < base.config.MLP_PARAMS.min_station_rows or not score_mask.any():
                    continue
                key = f'local:{station}:{horizon}'
                local_budget = selected_rounds.get(key, global_budget) if selected_rounds is not None else budget
                local_model, used_rounds = fit_unit(key,
                    station_rows, residual_targets[station_rows.index.to_numpy()], xcols,
                    params, seed, local_budget, patience, multi, fit_counter)
                round_counts[key] = used_rounds
                prediction[score_mask] += local_model.predict(score_rows.loc[score_mask, xcols])
                record_model(local_model, station, 'local_residual_multi_output' if multi else 'local_residual')
        if multi:
            blocks.extend(base._prediction_blocks(score_rows, prediction))
        else:
            block = score_rows[[base.config.STATION_ID_COL, base.config.DATE_COL]].copy()
            block['horizon'], block['y_true'], block['y_pred'] = horizon, score_rows[targets].to_numpy(), prediction
            blocks.append(block)
    predictions = pd.concat(blocks, ignore_index=True)
    reports = base.metrics.build_reports(predictions)
    score = reports['overall']['macro'].get('primary_macro_rmse')
    if score is None or not np.isfinite(score):
        raise ValueError('No finite validation primary_macro_rmse was produced')
    by_horizon = reports['by_horizon'].set_index('horizon')['rmse'].to_dict()
    result = {'score': float(score), 'predictions': predictions, 'reports': reports,
            'importance': pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
            'model_records': records, 'selected_rounds': round_counts,
            'best_iterations': {k: v - 1 for k, v in round_counts.items()}, 'best_scores': {},
            'rmse_by_horizon': {f'rmse_t{h}': float(by_horizon[h]) for h in base.config.FORECAST_HORIZONS},
            'folds_manifest': base.splits.fold_manifest(folds)}
    if scope and not keep_models:
        scope.save_summary({k: result[k] for k in ('score', 'selected_rounds', 'best_iterations',
                                                  'best_scores', 'rmse_by_horizon')})
    return result


def cpu_preflight(multi):
    """Probe the explicitly selected CPU architecture without changing devices."""
    import xgboost as xgb
    X = np.arange(128, dtype=float).reshape(32, 4)
    y = np.column_stack([X[:, 0] + h for h in range(7)]) if multi else X[:, 0]
    model = base._xgb({'device': 'cpu', 'n_jobs': 1, 'max_depth': 2}, 42, 3, 2, multi)
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    if np.asarray(model.predict(X)).shape != y.shape:
        raise RuntimeError('XGBoost CPU preflight prediction shape mismatch')
    return {'xgboost_version': xgb.__version__, 'requested_device': 'cpu'}


def _stop_for_resume(signum, frame):
    base.LOGGER.warning('Received signal %s; exiting with completed checkpoints intact', signum)
    raise SystemExit(75)


def main(argv=None):
    with ExitStack() as stack:
        return _main(argv, stack)


def _main(argv, stack):
    args = base.parse_args(argv)
    if args.experiment not in (13, 14):
        raise ValueError('This runner supports experiments 13 and 14 only')
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s %(levelname)s %(message)s')
    spec = base.make_spec(args.experiment, args.seed)
    device = args.device or ('cpu' if args.experiment == 14 else 'cuda:0')
    if device == 'cpu':
        spec = replace(spec, prefer_gpu=False, run_id=spec.run_id.replace('XGB_CUDA', 'XGB_CPU'))
    gpu_info = (cpu_preflight(spec.forecast_strategy == 'multi') if device == 'cpu'
                else base.cuda_preflight(require_multi=spec.forecast_strategy == 'multi'))
    if not args.train:
        print(f'[dry run] XGBoost {device} architecture probe passed. Add --train to tune.')
        return 0
    started = perf_counter()
    df = base.runner.prepare_dataframe(spec, source=args.source)
    stations = base.data.resolve_stations(df, args.stations)
    if args.max_stations is not None:
        stations = stations[:args.max_stations]
    if not stations:
        raise ValueError('No eligible stations remain after selection')
    context = make_context(df, stations, spec, 'validation')
    run_id = ('DEV_' if args.development else '') + spec.run_id
    run_dir = args.resume.resolve() if args.resume else base._artifact_dir(run_id)
    checkpoint = None
    if args.checkpoint or args.resume:
        checkpoint = stack.enter_context(RunCheckpoint(
            run_dir, run_identity(df, stations, args, device), resume=bool(args.resume)))
        context['checkpoint'] = checkpoint
        for sig in (signal.SIGTERM, getattr(signal, 'SIGUSR1', signal.SIGTERM)):
            previous = signal.signal(sig, _stop_for_resume)
            stack.callback(signal.signal, sig, previous)
        if (checkpoint.root / 'completed.json').exists():
            base.LOGGER.info('Run already completed: %s', run_dir)
            return 0
    base.LOGGER.info('Output: %s; tuning global + station residual XGBoost', run_dir)
    base.LOGGER.info('Early stopping: patience=%d, purged 85/15 training split for global/OOF/local; '
                     'test reuses each validation-selected tree count', args.early_stopping_rounds)
    best_params, summary, _, _, fit_counter = base.hierarchical_search(context, args, run_dir)
    validation = fit_evaluate(context, best_params, base.MAX_BOOSTING_ROUNDS,
                              args.early_stopping_rounds, args.seed, {'total': 0}, True)
    selected = validation['selected_rounds']
    run_config = {
        'description': 'Tawan XGBoost with per-station OOF tree residual correction (E1.3 method).',
        'experiment_id': args.experiment, 'algorithm': 'xgboost',
        'training_strategy': spec.training_strategy, 'forecast_strategy': spec.forecast_strategy,
        'global_algorithm': 'xgboost', 'local_algorithm': 'xgboost', 'device': device,
        'multi_strategy': 'multi_output_tree' if spec.forecast_strategy == 'multi' else None,
        'seed': args.seed, 'random_seed': args.seed, 'source': args.source, 'gpu_info': gpu_info,
        'evaluation_splits': ['validation'] if args.skip_test else ['validation', 'test'],
        'best_params': best_params, 'selected_boosting_rounds': selected,
        'tuning_mode': args.tuning_mode, 'tuning_summary': summary,
        'development_run': args.development, 'selection_split': 'validation',
        'test_locked_during_tuning': True, 'early_stopping_policy': 'purged_training_tail_85_15_then_refit',
        'early_stopping_rounds': args.early_stopping_rounds,
        'oof_rounds_policy': 'per_fold_validation_selected',
        'local_min_rows': base.config.MLP_PARAMS.min_station_rows,
        'local_fallback': 'global_prediction', 'new_local_rounds_fallback': 'selected_global_rounds',
        'residual_source': 'expanding_training_only_out_of_fold',
    }

    def save(result, split):
        base.artifacts.save_run(
            run_dir=run_dir, run_id=run_id, run_config=run_config, df=df, stations=stations,
            folds_manifest=result['folds_manifest'], reports=result['reports'],
            predictions=result['predictions'], feature_list=context['features'],
            feature_importance=result['importance'], training_log=run_config['description'],
            split=split, model_records=result['model_records'], prefer_gpu=spec.prefer_gpu)
        result['folds_manifest'].to_csv(run_dir / f'residual_oof_folds_{split}.csv', index=False)
        # The shared writer supplies Stage 1's default seed. Preserve this
        # runner's explicit --seed in the final reproducibility manifest.
        config_path = run_dir / 'config.json'
        saved_config = base.json.loads(config_path.read_text(encoding='utf-8'))
        saved_config['seed'] = args.seed
        base.artifacts.write_json(config_path, saved_config)

    save(validation, 'validation')
    base.artifacts.write_json(run_dir / 'best_params.json',
                              {'params': best_params, 'selected_boosting_rounds': selected,
                               'primary_macro_rmse': validation['score']})
    if not args.skip_test:
        test_context = make_context(df, stations, spec, 'test')
        if checkpoint:
            test_context['checkpoint'] = checkpoint
        test = fit_evaluate(test_context, best_params, base.MAX_BOOSTING_ROUNDS,
                            None, args.seed, {'total': 0}, True, selected_rounds=selected)
        save(test, 'test')
        base.LOGGER.info('Test primary macro RMSE: %.6f', test['score'])
    base.artifacts.write_json(run_dir / 'runtime.json',
                              {'total_seconds': perf_counter() - started,
                               'model_fits_during_tuning': fit_counter['total']})
    base.LOGGER.info('Completed experiment %d; artifacts: %s', args.experiment, run_dir)
    if checkpoint:
        atomic_json(checkpoint.root / 'completed.json', {'experiment': args.experiment, 'run_id': run_id})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
