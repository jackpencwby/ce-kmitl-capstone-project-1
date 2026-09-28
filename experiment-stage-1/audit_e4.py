"""Read-only audit of E4 artifacts; writes separate analysis outputs, never trains."""
from pathlib import Path
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from common import config, models, runner, features, data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / 'E4_audit_20260925'
OUT.mkdir(exist_ok=True)
runs = {json.loads((p/'config.json').read_text())['spatial_mode']:p
        for p in (ROOT/'artifacts').glob('E4_LOCAL*')}
result = {'runs':{}, 'bootstrap':[], 'split_probe':{}}

# Probe the actual routing, replacing only expensive fitting and final reporting.
probe = pd.DataFrame({'station_id':[1]*3, 'date':pd.to_datetime(
    ['2025-11-18','2025-11-20','2026-04-14'])})
seen = []
def capture(spec, factory, tr, va, cols, records, group):
    seen.append({'train':tr.date.astype(str).tolist(), 'evaluation':va.date.astype(str).tolist()})
    return pd.DataFrame(), []
with patch.object(models, '_fit_partition', capture), patch.object(models, '_finalize', return_value={}):
    models.run_holdout(models.RunSpec(run_id='audit',algorithm='xgboost',training_strategy='local'),
                       probe,[1],evaluation='test')
result['split_probe'] = {'requested':'test','observed':seen,
    'correct_test_routing':seen[0]['evaluation']==['2026-04-14']}
print('Split probe:',result['split_probe'],flush=True)

preds = {}
for mode,p in runs.items():
    v=pd.read_parquet(p/'predictions_validation.parquet')
    t=pd.read_parquet(p/'predictions_test.parquet')
    c=json.loads((p/'config.json').read_text())
    preds[mode]=v.dropna(subset=['y_true','y_pred']).sort_values(['station_id','date','horizon']).reset_index(drop=True)
    result['runs'][mode]={'directory':p.name,'test_equals_validation':v.equals(t),
        'test_min_date':str(t.date.min()),'test_max_date':str(t.date.max()),
        'configured_test_start':c['test_start'],'n_valid':len(preds[mode]),
        'overall':json.loads((p/'metrics_overall.json').read_text())}
base=preds['none']
for mode,d in preds.items():
    assert d[['station_id','date','horizon','y_true']].equals(base[['station_id','date','horizon','y_true']])

# Paired circular moving-block bootstrap of origin dates, keeping all stations
# and horizons together. Recompute the exact station-horizon macro RMSE.
dates=pd.date_range(base.date.min(),base.date.max())
groups=pd.MultiIndex.from_frame(base[['station_id','horizon']]).unique()
di=dates.get_indexer(base.date)
gi=groups.get_indexer(pd.MultiIndex.from_frame(base[['station_id','horizon']]))
counts=np.zeros((len(dates),len(groups)))
np.add.at(counts,(di,gi),1)
ss={}
for mode,d in preds.items():
    ss[mode]=np.zeros_like(counts)
    np.add.at(ss[mode],(di,gi),(d.y_pred.to_numpy(dtype=float)-d.y_true.to_numpy(dtype=float))**2)
for block in [7,14,28]:
    rng=np.random.default_rng(42)
    w=np.zeros((2000,len(dates)))
    for i in range(len(w)):
        idx=((rng.integers(len(dates),size=int(np.ceil(len(dates)/block)))[:,None]
              +np.arange(block))%len(dates)).ravel()[:len(dates)]
        w[i]=np.bincount(idx,minlength=len(dates))
    den=w@counts
    scores={}
    for mode in ss:
        scores[mode]=np.nanmean(np.sqrt(np.divide(w@ss[mode],den,
            out=np.full_like(den,np.nan),where=den>0)),axis=1)
    for mode in ['unweighted','distance','wind']:
        gain=100*(scores['none']-scores[mode])/scores['none']
        row={'mode':mode,'block_days':block,'replicates':len(w),
             'gain_percent_ci95':np.percentile(gain,[2.5,97.5]).tolist()}
        result['bootstrap'].append(row)
    print('Bootstrap complete',block,flush=True)

# Reconstruct features with current pipeline to inspect coverage and lag semantics.
df=runner.prepare_dataframe(models.RunSpec(run_id='audit',algorithm='xgboost'),source='local')
coords=data.station_coordinates(df)
nb=features.build_neighbor_table(coords)
nc=nb.groupby('station_id').size()
stations=base.station_id.unique()
result['neighbors']={'count_distribution':nc.reindex(stations,fill_value=0).value_counts().sort_index().to_dict()}
selected=df[df.station_id.isin(stations)].copy()
date_shift=selected.groupby('station_id').date
result['lag_non_calendar_fraction']={str(lag):float(((selected.date-date_shift.shift(lag)).dt.days.dropna()!=lag).mean()) for lag in [1,3,7]}
spatial_quality={}
for mode in ['unweighted','distance','wind']:
    f=features.add_neighbor_features(df,nb,mode=mode)
    f=f.merge(base[['station_id','date']].drop_duplicates(),on=['station_id','date'],how='inner')
    spatial_quality[mode]={col:float(f[col].isna().mean()) for col in features.neighbor_feature_cols()}
result['spatial_missingness_validation']=spatial_quality
meta=features.assign_region(selected.drop_duplicates('station_id')).set_index('station_id')
station_tables=[]
for mode,p in runs.items():
    s=pd.read_csv(p/'metrics_by_station.csv').set_index('station_id')
    s['mode']=mode
    s['region']=meta.region_id.reindex(s.index)
    s['neighbor_count']=nc.reindex(s.index,fill_value=0)
    station_tables.append(s.reset_index())
st=pd.concat(station_tables)
st.to_csv(OUT/'station_analysis.csv',index=False)
region=st.groupby(['mode','region']).agg(stations=('station_id','size'),macro_station_rmse=('rmse','mean'))
region.to_csv(OUT/'region_analysis.csv')
station_rmse=st.pivot(index='station_id',columns='mode',values='rmse')
result['station_changes']={mode:{'improved':int((station_rmse[mode]<station_rmse['none']).sum()),
    'worse':int((station_rmse[mode]>station_rmse['none']).sum())} for mode in ['unweighted','distance','wind']}
pd.DataFrame(result['bootstrap']).to_csv(OUT/'bootstrap.csv',index=False)
(OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2,default=str),flush=True)
