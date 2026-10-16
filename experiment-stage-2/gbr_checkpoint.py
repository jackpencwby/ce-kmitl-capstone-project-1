"""Atomic trusted-local sklearn checkpoints; never load files from other users."""
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
import joblib
from filelock import FileLock

LOG = logging.getLogger(__name__)
class CheckpointError(ValueError): pass

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)

class Checkpoint:
    def __init__(self, run_dir, identity, resume=False):
        self.run_dir = Path(run_dir)
        self.root = self.run_dir/'checkpoint'
        self.identity = identity; self.resume = resume
        self.lock = FileLock(str(self.run_dir/'.training.lock'), timeout=0)
    def __enter__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True); self.lock.acquire()
        try:
            p = self.root/'manifest.json'
            if self.resume:
                if not p.exists() or digest(json.loads(p.read_text())) != digest(self.identity):
                    raise CheckpointError('Checkpoint mismatch: data, code, environment or arguments changed')
            elif p.exists(): raise CheckpointError('Use --resume for an existing checkpoint')
            else: atomic_json(p, self.identity)
        except BaseException:
            self.lock.release(); raise
        return self
    def __exit__(self, *exc): self.lock.release()
    def fit(self, key, train):
        p = self.root/'models'/(digest(key)+'.joblib'); marker = p.with_suffix('.json')
        if marker.exists():
            try:
                saved=json.loads(marker.read_text())
                if not isinstance(saved,dict) or not p.exists() or file_hash(p)!=saved['sha256']:
                    raise CheckpointError('Corrupt committed model: '+str(p))
                LOG.info('Resume fit %s',p.stem[:12])
                return joblib.load(p)
            except Exception as exc:
                raise CheckpointError(str(exc)) from exc
        model = train()  # Training errors retain the runner's candidate-failure policy.
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fd,tmp = tempfile.mkstemp(dir=p.parent,suffix='.tmp'); os.close(fd)
            try:
                joblib.dump(model,tmp)
                with open(tmp,'rb+') as f: os.fsync(f.fileno())
                os.replace(tmp,p)
                atomic_json(marker, {'sha256':file_hash(p)})
            finally: Path(tmp).unlink(missing_ok=True)
            LOG.info('Checkpoint fit %s',p.stem[:12])
            return model
        except Exception as exc:
            raise CheckpointError(str(exc)) from exc
    def load_trial(self,key):
        p=self.root/'trials'/(digest(key)+'.json')
        if not p.exists(): return None
        try:
            saved=json.loads(p.read_text())
            if digest(saved['value']) != saved['sha256']: raise CheckpointError('Trial checksum mismatch')
            return saved['value']
        except (OSError, ValueError, KeyError, TypeError) as exc: raise CheckpointError(str(exc)) from exc
    def save_trial(self,key,value):
        try:
            atomic_json(self.root/'trials'/(digest(key)+'.json'), {'value':value,'sha256':digest(value)})
        except (OSError, ValueError) as exc: raise CheckpointError(str(exc)) from exc
    def prune_models(self):
        # Called only after the full candidate's score has committed.
        shutil.rmtree(self.root/'models',ignore_errors=True)
    def complete(self):
        files={str(p.relative_to(self.run_dir)):file_hash(p) for p in self.run_dir.rglob('*')
               if p.is_file() and 'checkpoint' not in p.relative_to(self.run_dir).parts and not p.name.startswith('.')}
        atomic_json(self.root/'completed.json', {'files':files,'sha256':digest(files)})
    def completed(self):
        p=self.root/'completed.json'
        if not p.exists(): return False
        try:
            saved=json.loads(p.read_text()); files=saved['files']
            if digest(files)!=saved['sha256']: raise CheckpointError('Completion checksum mismatch')
            for name,h in files.items():
                target=(self.run_dir/name).resolve()
                if not target.is_relative_to(self.run_dir.resolve()) or file_hash(target)!=h:
                    raise CheckpointError('Completed output changed: '+name)
            return True
        except (OSError, ValueError, KeyError, TypeError) as exc: raise CheckpointError(str(exc)) from exc

def run_identity(frame,args,stations,root):
    import numpy as np
    import pandas as pd
    import sklearn
    sources=list((root/'experiment-stage-1/common').glob('*.py'))
    stage=root/'experiment-stage-2'
    sources += [stage/'run_stage2_gradient_boosting.py', stage/'gbr_checkpoint.py']
    sources += list((stage/'config').glob('*.py'))+list((stage/'models').glob('*.py'))
    return {'format':1,'data':joblib.hash(frame), 'stations':list(map(str,stations)),
        'options':{k:v for k,v in vars(args).items() if k not in {'resume','run_dir','checkpoint','log_level'}},
        'sources':{str(p.relative_to(root)):file_hash(p) for p in sources},
        'versions':[sys.version,np.__version__,pd.__version__,sklearn.__version__,joblib.__version__]}
