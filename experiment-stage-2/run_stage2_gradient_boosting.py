#!/usr/bin/env python
"""Run Stage 2 Gradient Boosting experiments 5, 6, 11, 12, 17 and 18."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "experiment-stage-1"
if str(S1) not in sys.path:
    sys.path.insert(0, str(S1))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import artifacts, config, data, metrics, models, runner, splits  # noqa: E402
from config.gradient_boosting import (FIXED, GBR_PARAM_DOMAIN, PHASE1_GRID,
    PHASE2_GRID, phase1_candidates, phase2_candidates, phase3_candidates)  # noqa: E402
from models.joint_gradient_boosting import JointMultiOutputGradientBoostingRegressor  # noqa: E402

LOG = logging.getLogger("stage2_gradient_boosting")
TARGET_COLS = [f"target_t{h}" for h in config.FORECAST_HORIZONS]
TARGET_LABELS = [f"t+{h}" for h in config.FORECAST_HORIZONS]
EXPERIMENTS = {
    5: ("global", "direct", False, "GLOBAL_SINGLE_HEAD"),
    6: ("global", "multi", False, "GLOBAL_JOINT_MULTI_HEAD"),
    11: ("regional", "direct", False, "REGIONAL_SINGLE_HEAD"),
    12: ("regional", "multi", False, "REGIONAL_JOINT_MULTI_HEAD"),
    17: ("global_local_tree", "direct", True, "GLOBAL_LOCAL_RESIDUAL_SINGLE_HEAD"),
    18: ("global_local_tree", "multi", True, "GLOBAL_LOCAL_RESIDUAL_JOINT_MULTI_HEAD"),
}

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS), type=int)
    p.add_argument("--train", action="store_true")
    p.add_argument("--source", choices=("auto", "local", "gcs"), default="auto")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--stations", nargs="+", default=["all"])
    p.add_argument("--max-stations", type=int)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--tuning-mode", choices=("hierarchical-grid",), default="hierarchical-grid")
    p.add_argument("--stage-patience", type=int, default=100)
    p.add_argument("--phase1-pruning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-estimators", type=int, default=2000)
    p.add_argument("--max-phase1-candidates", type=int)
    p.add_argument("--max-phase2-candidates", type=int)
    p.add_argument("--max-phase3-candidates", type=int)
    p.add_argument("--development", action="store_true")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    for name in ("max_stations", "max_estimators", "stage_patience", "max_phase1_candidates",
                 "max_phase2_candidates", "max_phase3_candidates"):
        v = getattr(a, name)
        if v is not None and v < 1: p.error(f"--{name.replace('_','-')} must be positive")
    if a.development:
        a.max_phase1_candidates = min(a.max_phase1_candidates or 3, 3)
        a.max_phase2_candidates = min(a.max_phase2_candidates or 3, 3)
        a.max_phase3_candidates = min(a.max_phase3_candidates or 3, 3)
        a.max_estimators = min(a.max_estimators, 20)
        a.skip_test = True
    return a

def make_spec(exp, seed):
    scope, forecast, _, label = EXPERIMENTS[exp]
    return models.RunSpec(run_id=f"S2_{exp}__{label}__GBR_CPU__SEED{seed}",
        algorithm="gbr", training_strategy=scope, forecast_strategy=forecast,
        prefer_gpu=False, include_station_id=True, include_region_id=scope == "regional",
        complete_target_evaluation=True)

def cpu_preflight():
    import sklearn
    X = np.arange(32, dtype=float).reshape(16, 2)
    y = np.arange(16, dtype=float)
    scalar = GradientBoostingRegressor(n_estimators=2, random_state=42).fit(X, y)
    tree = __import__("sklearn.tree", fromlist=["DecisionTreeRegressor"]).DecisionTreeRegressor(max_depth=2).fit(X, np.tile(y[:, None], (1, 7)))
    if tree.predict(X).shape != (16, 7): raise RuntimeError("DecisionTreeRegressor multi-output preflight failed")
    joint = JointMultiOutputGradientBoostingRegressor(n_estimators=2, random_state=42).fit(X, np.tile(y[:, None], (1, 7)))
    if joint.predict(X).shape != (16, 7): raise RuntimeError("Joint GBR preflight failed")
    return {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
            "backend": "sklearn", "device": "cpu", "scalar": "PASS",
            "multi_output_tree": "PASS", "joint_shape": list(joint.predict(X).shape)}

def build_context(spec, df, stations):
    frame = df[df[config.STATION_ID_COL].isin(stations)].copy()
    masks = splits.date_masks(frame)
    base = models.assemble_feature_columns(spec)
    frame, cols = models._add_categorical_codes(frame, spec, base, masks["train"])
    groups = [("global", frame)] if spec.training_strategy != "regional" else list(frame.groupby("region_id", sort=True))
    parts = []
    for group, sub in groups:
        tr = sub.loc[masks["train"].reindex(sub.index)].copy()
        va = sub.loc[masks["validation"].reindex(sub.index)].copy()
        parts.append((str(group), tr, va))
    return {"frame": frame, "masks": masks, "groups": parts, "columns": cols}

def _rows(tr, va, joint, evaluation_start, evaluation_end):
    maxh = max(config.FORECAST_HORIZONS)
    if joint:
        tr = tr.dropna(subset=TARGET_COLS)
        tr = tr[tr[config.DATE_COL] + pd.to_timedelta(int(maxh), unit="D") < evaluation_start]
        va = va.dropna(subset=TARGET_COLS)
        va = va[va[config.DATE_COL] + pd.to_timedelta(int(maxh), unit="D") < evaluation_end]
    else:
        # Stage 1 uses the complete-seven-target cohort for comparable pooled
        # evaluation, while each scalar head may use every available training
        # label up to its own horizon boundary.
        tr = tr.copy()
        va = va.dropna(subset=TARGET_COLS)
        va = va[va[config.DATE_COL] + pd.to_timedelta(int(maxh), unit="D") < evaluation_end]
    return tr, va

def _model(params, joint, seed, n_estimators):
    p = dict(params)
    p.pop("n_estimators", None)
    p["n_estimators"] = int(n_estimators)
    p["random_state"] = int(seed)
    return (JointMultiOutputGradientBoostingRegressor(**p) if joint
            else GradientBoostingRegressor(loss="squared_error", **p))

def _stage_predictions(context, params, joint, seed, budget, evaluation="validation", patience=100):
    """Fit group systems once and select the best common boosting stage on validation."""
    start = pd.Timestamp(config.VALIDATION_START)
    end = pd.Timestamp(config.TEST_START)
    fitted, staged_blocks, importance = [], [], []
    eval_groups = context["groups"]
    for group, tr, va in eval_groups:
        if evaluation == "test":
            tr = context["frame"].loc[context["masks"]["train"] | context["masks"]["validation"]]
            va = context["frame"].loc[context["masks"]["test"]]
            if group != "global" and "region_id" in va:
                va = va[va["region_id"].astype(str) == group]
            if group != "global" and "region_id" in tr:
                tr = tr[tr["region_id"].astype(str) == group]
            start = pd.Timestamp(config.TEST_START); end = pd.Timestamp.max.normalize()
        tr, va = _rows(tr, va, joint, start, end)
        if tr.empty or va.empty: continue
        cols = context["columns"]
        imputer = SimpleImputer(strategy="constant", fill_value=-999.0, keep_empty_features=True)
        Xtr = imputer.fit_transform(tr[cols]); Xva = imputer.transform(va[cols])
        if not np.isfinite(Xtr).all() or not np.isfinite(Xva).all():
            raise ValueError("Stage 1 GBR preprocessing left non-finite feature values")
        if joint:
            est = _model(params, True, seed, budget).fit(Xtr, tr[TARGET_COLS].to_numpy(float))
            iterator = est.staged_predict(Xva)
            targets = [0]
        else:
            ests, iterators = [], []
            for h in config.FORECAST_HORIZONS:
                htr = tr.dropna(subset=[f"target_t{h}"])
                htr = htr[htr[config.DATE_COL] + pd.to_timedelta(int(h), unit="D") < start]
                if htr.empty: raise ValueError(f"No training target rows for t+{h}")
                # Refit imputer on each horizon's actual training rows only.
                hi = SimpleImputer(strategy="constant", fill_value=-999.0, keep_empty_features=True)
                hx = hi.fit_transform(htr[cols]); hv = hi.transform(va[cols])
                e = _model(params, False, seed + h, budget).fit(hx, htr[f"target_t{h}"].to_numpy(float))
                ests.append((e, hi, hv)); iterators.append(e.staged_predict(hv))
            def direct_stages():
                for stage_values in zip(*iterators): yield np.column_stack(stage_values)
            iterator = direct_stages(); est = ests
        def group_predictions(iterator=iterator, va=va, group=group):
          if evaluation == "test":
            matrix = None
            for pred in iterator: matrix = np.asarray(pred)
            stages = [matrix] if matrix is not None else []
          else:
            stages = iterator
          for pred in stages:
            matrix = np.asarray(pred)
            if joint and matrix.shape != (len(va), 7): raise RuntimeError("Joint model prediction shape mismatch")
            if not joint and matrix.shape != (len(va), 7): raise RuntimeError("Seven scalar models did not produce seven targets")
            blocks = []
            for j, h in enumerate(config.FORECAST_HORIZONS):
                b = va[[config.STATION_ID_COL, config.DATE_COL]].copy()
                b["horizon"] = h; b["y_true"] = va[f"target_t{h}"].to_numpy(); b["y_pred"] = matrix[:, j]
                blocks.append(b)
            yield pd.concat(blocks, ignore_index=True)
        staged_blocks.append(group_predictions())
        fitted.append((group, est, imputer if joint else None))
        if joint:
            importance.append(pd.DataFrame({"feature": cols, "importance": est.feature_importances_, "model": "joint"}).assign(group=group))
        else:
            for h, (e, _, _) in zip(config.FORECAST_HORIZONS, est):
                importance.append(pd.DataFrame({"feature": cols, "importance": e.feature_importances_, "model": f"t{h}"}).assign(group=group))
    best_score, best_iter, best_pred, bad = float("inf"), 1, None, 0
    for i, stage_predictions in enumerate(zip(*staged_blocks), 1):
        pred = pd.concat(stage_predictions, ignore_index=True)
        score = metrics.build_reports(pred)["overall"]["macro"].get("primary_macro_rmse", np.nan)
        if np.isfinite(score) and score < best_score - 1e-12:
            best_score, best_iter, best_pred, bad = float(score), i, pred, 0
        else:
            bad += 1
        if bad >= patience: break
    return best_score, best_iter, best_pred, fitted, pd.concat(importance, ignore_index=True) if importance else pd.DataFrame()

def _subset(candidates, n): return candidates if n is None else candidates[:n]

def _seeded_order(candidates, seed):
    indices=np.random.RandomState(seed).permutation(len(candidates))
    return [candidates[int(i)] for i in indices]

def hierarchical_search(context, args, run_dir, exp, joint, finetune=False):
    if context["masks"]["test"].any() and context["evaluation"] != "validation":
        raise ValueError("test context is locked during tuning")
    rows=[]; fits=0; trees=0; trial=0; phase_times={}
    max_est = 20 if args.development else args.max_estimators
    budgets = tuple(sorted(set([min(200,max_est), min(600,max_est), max_est])))
    def evaluate(phase, round_no, candidates, budget):
        nonlocal fits, trees, trial
        scored=[]; tick=perf_counter()
        for params in candidates:
            trial += 1; t=perf_counter(); row={"phase":phase,"round":round_no,"trial_id":trial,
                "experiment_id":exp,"parameters":json.dumps(params, sort_keys=True),"budget":budget}
            try:
                score, it, prediction, fitted, importance = _stage_predictions(
                    context, params, joint, args.seed, budget, patience=args.stage_patience)
                if finetune:
                    composed={"prediction":prediction,"fitted":fitted,"importance":importance,
                              "records":[],"feature_columns":context["columns"]}
                    composed=_apply_local_residual_tuning(composed,context,params,joint,args.seed,it,"validation")
                    prediction=composed["prediction"]
                    score=metrics.build_reports(prediction)["overall"]["macro"].get("primary_macro_rmse",np.nan)
                    local_fits=len(composed["records"])+composed.get("residual_oof_model_fits",0)
                else:
                    local_fits=0
                horizon_metrics=metrics.build_reports(prediction)["by_horizon"]
                row.update(best_iteration=it,validation_primary_macro_rmse=score,status="ok",failure_reason="")
                for _, metric_row in horizon_metrics.iterrows():
                    row[f"rmse_t{int(metric_row['horizon'])}"]=metric_row["rmse"]
                fits += sum(1 if joint else 7 for _, est, _ in fitted) + local_fits
                trees += sum(len(est.estimators_) if joint else sum(len(e.estimators_) for e,_,_ in est) for _,est,_ in fitted)
                if finetune:
                    trees += sum(len(rec["estimator"]["model"].estimators_) for rec in composed["records"])
                    trees += composed.get("residual_oof_weak_trees_fit",0)
                scored.append((score,params))
            except Exception as exc:
                row.update(best_iteration=None,validation_primary_macro_rmse=np.nan,status="failed",failure_reason=f"{type(exc).__name__}: {exc}")
                LOG.exception("Trial %s failed",trial)
            row["elapsed_seconds"]=perf_counter()-t; rows.append(row)
        phase_times[phase]=phase_times.get(phase,0)+perf_counter()-tick
        pd.DataFrame(rows).to_csv(run_dir/"tuning_trials.csv",index=False)
        # Python's stable sort preserves the seeded candidate order for ties.
        return sorted(scored,key=lambda x:x[0])
    # Phase 1 successive halving: deterministic grid, three increasing budgets.
    p1=_subset(_seeded_order(phase1_candidates(),args.seed),args.max_phase1_candidates)
    survivors=p1
    for r,budget in enumerate(budgets,1):
        scores=evaluate("phase1",r,survivors,budget)
        if not scores: raise RuntimeError("Phase 1 produced no successful candidates")
        survivors=[p for _,p in scores[:max(1,(len(scores)+2)//3)]] if r<len(budgets) and args.phase1_pruning else [p for _,p in scores]
    pd.DataFrame([r for r in rows if r["phase"]=="phase1"]).to_csv(run_dir/"phase1_results.csv",index=False)
    phase1_survivors=survivors
    # Phase 2 around strongest structural configuration, grid-pruned in a seeded stable order.
    anchor=dict(survivors[0]); anchor.update(FIXED)
    p2=_subset(_seeded_order(phase2_candidates(anchor),args.seed),args.max_phase2_candidates)
    # Limit expensive regularization grid with deterministic low-discrepancy evenly spaced indices.
    if len(p2)>36 and not args.development: p2=[p2[i] for i in np.linspace(0,len(p2)-1,36,dtype=int)]
    scores2=evaluate("phase2",1,p2,max_est); pd.DataFrame([r for r in rows if r["phase"]=="phase2"]).to_csv(run_dir/"phase2_results.csv",index=False)
    if not scores2: raise RuntimeError("Phase 2 produced no successful candidates")
    anchor=dict(scores2[0][1])
    p3=_subset(_seeded_order(phase3_candidates(anchor),args.seed),args.max_phase3_candidates)
    if len(p3)>36 and not args.development: p3=[p3[i] for i in np.linspace(0,len(p3)-1,36,dtype=int)]
    scores3=evaluate("phase3",1,p3,max_est); pd.DataFrame([r for r in rows if r["phase"]=="phase3"]).to_csv(run_dir/"phase3_results.csv",index=False)
    if not scores3: raise RuntimeError("Phase 3 produced no successful candidates")
    best=scores3[0][1]
    _, best_iter, pred, fitted, importance=_stage_predictions(context,best,joint,args.seed,max_est)
    summary={"phase1_candidate_count":len(p1),"phase1_survivors":len(phase1_survivors),
        "phase2_candidate_count":len(p2),"phase3_candidate_count":len(p3),"total_model_fits":fits,
        "total_weak_trees_fit":trees,"best_validation_score":scores3[0][0],"best_parameters":best,
        "best_iteration":best_iter,"phase_seconds":phase_times,"total_tuning_seconds":sum(phase_times.values()),
        "development_run":args.development}
    artifacts.write_json(run_dir/"best_params.json",{**best,"n_estimators":best_iter})
    artifacts.write_json(run_dir/"tuning_summary.json",summary)
    return {"params":best,"best_iteration":best_iter,"prediction":pred,"fitted":fitted,
            "importance":importance,"summary":summary}

def _write_result(run_dir, result, spec, df, stations, run_config, split):
    from common import models as s1models
    finalized=s1models._finalize(spec,result["prediction"], [result["importance"]],
                                 result["feature_columns"], result["records"])
    artifacts.save_run(run_dir=run_dir,run_id=spec.run_id,run_config=run_config,df=df,stations=stations,
        folds_manifest=pd.DataFrame(),reports=finalized["reports"],predictions=finalized["predictions"],
        feature_list=finalized["feature_list"],feature_importance=finalized["importance"],
        training_log="Stage 2 Gradient Boosting",split=split,model_records=[],prefer_gpu=False)
    model_dir=run_dir/"models"/split
    model_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for rec in result["records"]:
        group="_".join(str(rec["group"]).split())
        stem=(f"{group}_horizon_{rec['horizon']}" if rec["role"] in ("main","local_residual")
              else f"{group}_{rec['role']}")
        path=model_dir/f"{stem}.joblib"
        joblib.dump(rec["estimator"],path)
        manifest.append({"role":rec["role"],"group":rec["group"],"horizon":rec["horizon"],
            "file":str(path.relative_to(run_dir)),"feature_columns":rec.get("feature_columns",result["feature_columns"]),
            "architecture":"one joint multi-output estimator" if "joint" in rec["role"] else "scalar GradientBoostingRegressor"})
    artifacts.write_json(run_dir/"model_manifest.json",{"backend":"sklearn","device":"cpu","models":manifest})

def _fit_final(context, params, joint, seed, n_estimators, evaluation):
    # Reuse stage predictions for validation; test refit uses train + validation then scores test.
    score,it,pred,fitted,imp=_stage_predictions(context,params,joint,seed,n_estimators,evaluation=evaluation)
    records=[]
    for group, est, imputer in fitted:
        if joint:
            records.append({"estimator":{"imputer":imputer,"model":est},"group":group,"horizon":0,"role":"joint_multi_output"})
        else:
            for h,(m,imp_,_) in zip(config.FORECAST_HORIZONS,est):
                records.append({"estimator":{"imputer":imp_,"model":m},"group":group,"horizon":h,"role":"main"})
    return {"prediction":pred,"importance":imp,"records":records,"feature_columns":context["columns"],"score":score}

def _apply_local_residual_tuning(result, context, params, joint, seed, n_estimators, evaluation):
    """Stage 1 E1.3 residual recipe: station-local correctors trained on walk-forward OOF errors."""
    frame=context["frame"]
    boundary=pd.Timestamp(config.TEST_START if evaluation=="validation" else config.TEST_START)
    if evaluation=="test":
        boundary=pd.Timestamp(config.TEST_START)
        eval_rows=frame.loc[context["masks"]["test"]]
    else:
        eval_rows=frame.loc[context["masks"]["validation"]]
    oof_parts=[]; oof_model_fits=0; oof_trees=0
    folds=splits.residual_oof_folds(frame,boundary)
    cols=context["columns"]
    for fold in folds:
        tmask,vmask=splits.fold_masks(frame,fold)
        tr=frame.loc[tmask].copy(); va=frame.loc[vmask].copy()
        if joint:
            tr=tr.dropna(subset=TARGET_COLS)
            tr=tr[tr[config.DATE_COL]+pd.to_timedelta(int(max(config.FORECAST_HORIZONS)),unit="D")<fold.valid_start]
            va=va.dropna(subset=TARGET_COLS)
            va=va[va[config.DATE_COL]+pd.to_timedelta(int(max(config.FORECAST_HORIZONS)),unit="D")<boundary]
            if tr.empty or va.empty: continue
            imp=SimpleImputer(strategy="constant",fill_value=-999.0,keep_empty_features=True)
            Xtr=imp.fit_transform(tr[cols]); Xva=imp.transform(va[cols])
            model=_model(params,True,seed,n_estimators).fit(Xtr,tr[TARGET_COLS].to_numpy(float))
            oof_model_fits+=1; oof_trees+=len(model.estimators_)
            residual=va[TARGET_COLS].to_numpy(float)-model.predict(Xva)
            block=va[[config.STATION_ID_COL,config.DATE_COL]].copy()
            for j,h in enumerate(config.FORECAST_HORIZONS): block[f"residual_t{h}"]=residual[:,j]
            oof_parts.append(block)
        else:
            for h in config.FORECAST_HORIZONS:
                trh=tr.dropna(subset=[f"target_t{h}"])
                vah=va.dropna(subset=[f"target_t{h}"])
                trh=trh[trh[config.DATE_COL]+pd.to_timedelta(int(h),unit="D")<fold.valid_start]
                vah=vah[vah[config.DATE_COL]+pd.to_timedelta(int(h),unit="D")<boundary]
                if trh.empty or vah.empty: continue
                imp=SimpleImputer(strategy="constant",fill_value=-999.0,keep_empty_features=True)
                Xtr=imp.fit_transform(trh[cols]); Xva=imp.transform(vah[cols])
                model=_model(params,False,seed+h,n_estimators).fit(Xtr,trh[f"target_t{h}"].to_numpy(float))
                oof_model_fits+=1; oof_trees+=len(model.estimators_)
                block=vah[[config.STATION_ID_COL,config.DATE_COL]].copy()
                block[f"residual_t{h}"]=vah[f"target_t{h}"].to_numpy(float)-model.predict(Xva)
                oof_parts.append(block)
    if not oof_parts:
        LOG.warning("No walk-forward OOF rows available for local residual training; using global predictions")
        return result
    oof=pd.concat(oof_parts,ignore_index=True).groupby([config.STATION_ID_COL,config.DATE_COL],as_index=False).first()
    feature_rows=frame[[config.STATION_ID_COL,config.DATE_COL,*cols]].drop_duplicates(
        [config.STATION_ID_COL,config.DATE_COL])
    oof=oof.merge(feature_rows,on=[config.STATION_ID_COL,config.DATE_COL],how="left",validate="one_to_one")
    prediction=result["prediction"].copy()
    local_records=[]; local_importances=[]
    for sid,station_oof in oof.groupby(config.STATION_ID_COL,sort=True):
        station_eval=eval_rows[eval_rows[config.STATION_ID_COL]==sid]
        if len(station_oof)<config.MLP_PARAMS.min_station_rows or station_eval.empty: continue
        if joint:
            station_oof=station_oof.dropna(subset=[f"residual_t{h}" for h in config.FORECAST_HORIZONS])
            if len(station_oof)<config.MLP_PARAMS.min_station_rows: continue
            imp=SimpleImputer(strategy="constant",fill_value=-999.0,keep_empty_features=True)
            Xtr=imp.fit_transform(station_oof[cols]); Xev=imp.transform(station_eval[cols])
            local=_model(params,True,seed+int(sid) if str(sid).isdigit() else seed,n_estimators)
            local.fit(Xtr,station_oof[[f"residual_t{h}" for h in config.FORECAST_HORIZONS]].to_numpy(float))
            correction=local.predict(Xev)
            for j,h in enumerate(config.FORECAST_HORIZONS):
                mask=prediction[config.STATION_ID_COL].eq(sid)&prediction["horizon"].eq(h)
                keys=prediction.loc[mask,[config.STATION_ID_COL,config.DATE_COL]].merge(
                    station_eval[[config.STATION_ID_COL,config.DATE_COL]].reset_index(drop=True).assign(_i=np.arange(len(station_eval))),
                    on=[config.STATION_ID_COL,config.DATE_COL],how="left")
                correction_map=dict(zip(station_eval[config.DATE_COL],correction[:,j]))
                prediction.loc[mask,"y_pred"] += prediction.loc[mask,config.DATE_COL].map(correction_map).fillna(0).to_numpy()
            local_records.append({"estimator":{"imputer":imp,"model":local},"group":str(sid),"horizon":0,"role":"local_residual_joint"})
            local_importances.append(pd.DataFrame({"feature":cols,"importance":local.feature_importances_,"model":"local_joint"}).assign(group=str(sid)))
        else:
            for h in config.FORECAST_HORIZONS:
                residual_col=f"residual_t{h}"
                source=station_oof.dropna(subset=[residual_col])
                if len(source)<config.MLP_PARAMS.min_station_rows: continue
                imp=SimpleImputer(strategy="constant",fill_value=-999.0,keep_empty_features=True)
                Xtr=imp.fit_transform(source[cols]); Xev=imp.transform(station_eval[cols])
                local=_model(params,False,seed+h,n_estimators).fit(Xtr,source[residual_col].to_numpy(float))
                correction=local.predict(Xev)
                correction_map=dict(zip(station_eval[config.DATE_COL],correction))
                mask=prediction[config.STATION_ID_COL].eq(sid)&prediction["horizon"].eq(h)
                prediction.loc[mask,"y_pred"] += prediction.loc[mask,config.DATE_COL].map(correction_map).fillna(0).to_numpy()
                local_records.append({"estimator":{"imputer":imp,"model":local},"group":str(sid),"horizon":h,"role":"local_residual"})
                local_importances.append(pd.DataFrame({"feature":cols,"importance":local.feature_importances_,"model":f"local_t{h}"}).assign(group=str(sid)))
    result["prediction"]=prediction
    result["records"].extend(local_records)
    result["residual_oof_model_fits"]=oof_model_fits
    result["residual_oof_weak_trees_fit"]=oof_trees
    if local_importances: result["importance"]=pd.concat([result["importance"],*local_importances],ignore_index=True)
    return result

def main(argv=None):
    run_started=perf_counter()
    args=parse_args(argv)
    logging.basicConfig(level=getattr(logging,args.log_level.upper(),logging.INFO),format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed); np.random.seed(args.seed)
    preflight=cpu_preflight()
    exp=args.experiment; scope,forecast,finetune,label=EXPERIMENTS[exp]; joint=forecast=="multi"
    spec=make_spec(exp,args.seed)
    print("[Stage 2 Gradient Boosting Preflight]")
    print(json.dumps(preflight,indent=2)); print(f"Backend: CPU\nExperiment: {exp} {label}\nScope: {scope}\nForecast: {forecast}\nFine-tune: {'yes' if finetune else 'no'}\nSeed: {args.seed}")
    if not args.train:
        print("Preflight passed. Re-run with --train to load data and tune."); return 0
    df=runner.prepare_dataframe(spec,args.source)
    stations=data.resolve_stations(df,args.stations)
    if args.max_stations: stations=stations[:args.max_stations]
    runid=spec.run_id+("__DEV" if args.development else "")
    run_dir=Path(__file__).resolve().parent/"artifacts"/f"{runid}__{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True,exist_ok=False)
    # Keep held-out test rows and labels out of the tuning/refit object graph.
    validation_df=df[df[config.DATE_COL] < pd.Timestamp(config.TEST_START)].copy()
    context=build_context(spec,validation_df,stations); context["evaluation"]="validation"
    masks=context["masks"]
    print(f"Dataset rows={len(df)} stations={len(stations)} features={len(context['columns'])} targets={len(TARGET_COLS)}")
    split_counts=splits.date_masks(df)
    print(f"Train={split_counts['train'].sum()} Validation={split_counts['validation'].sum()} Test={split_counts['test'].sum()}")
    tuning_started=perf_counter()
    result=hierarchical_search(context,args,run_dir,exp,joint,finetune=finetune)
    tuning_seconds=perf_counter()-tuning_started
    refit_started=perf_counter()
    validation=_fit_final(context,result["params"],joint,args.seed,result["best_iteration"],"validation")
    if finetune:
        validation=_apply_local_residual_tuning(validation,context,result["params"],joint,args.seed,result["best_iteration"],"validation")
    final_refit_seconds=perf_counter()-refit_started
    rc={"experiment_id":exp,"scope":scope,"algorithm":"gradient_boosting","backend":"sklearn","device":"cpu",
        "training_strategy":scope,"forecast_strategy":forecast,"prediction_output":"joint_multi_output" if joint else "seven_independent_scalar_models",
        "joint_implementation":"one DecisionTreeRegressor fitted to the seven-dimensional residual matrix per stage" if joint else None,
        "multioutput_wrapper":"not used","fine_tune":finetune,"seed":args.seed,"target_columns":TARGET_COLS,
        "target_labels":TARGET_LABELS,"selection_split":"chronological_validation","test_locked_during_tuning":True,
        "development_run":args.development,"best_params":result["params"],"best_iteration":result["best_iteration"],
        "tuning_mode":"hierarchical-grid","phase1_pruning":args.phase1_pruning,"tuning_summary":result["summary"],
        "evaluation_splits":["validation"] if args.skip_test else ["validation","test"]}
    _write_result(run_dir,validation,spec,df,stations,rc,"validation")
    if args.skip_test:
        artifacts.write_json(run_dir/"runtime.json",{"phase1_seconds":result["summary"]["phase_seconds"].get("phase1",0),
            "phase2_seconds":result["summary"]["phase_seconds"].get("phase2",0),"phase3_seconds":result["summary"]["phase_seconds"].get("phase3",0),
            "tuning_total_seconds":tuning_seconds,"final_refit_seconds":final_refit_seconds,
            "test_prediction_seconds":0,"total_run_seconds":perf_counter()-run_started,
            "total_model_fits_during_tuning":result["summary"]["total_model_fits"],
            "total_weak_trees_fit_during_tuning":result["summary"]["total_weak_trees_fit"]})
        return 0
    test_started=perf_counter()
    test_spec=spec
    test_context=build_context(test_spec,df,stations); test_context["evaluation"]="test"
    test=_fit_final(test_context,result["params"],joint,args.seed,result["best_iteration"],"test")
    if finetune:
        test=_apply_local_residual_tuning(test,test_context,result["params"],joint,args.seed,result["best_iteration"],"test")
    _write_result(run_dir,test,spec,df,stations,rc,"test")
    artifacts.write_json(run_dir/"runtime.json",{"phase1_seconds":result["summary"]["phase_seconds"].get("phase1",0),
        "phase2_seconds":result["summary"]["phase_seconds"].get("phase2",0),"phase3_seconds":result["summary"]["phase_seconds"].get("phase3",0),
        "tuning_total_seconds":tuning_seconds,"final_refit_seconds":final_refit_seconds,
        "test_prediction_seconds":perf_counter()-test_started,"total_run_seconds":perf_counter()-run_started,
        "total_model_fits_during_tuning":result["summary"]["total_model_fits"],
        "total_weak_trees_fit_during_tuning":result["summary"]["total_weak_trees_fit"]})
    return 0

if __name__=="__main__":
    raise SystemExit(main())
