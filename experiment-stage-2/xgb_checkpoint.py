"""Crash-safe, single-writer checkpoints for residual XGBoost experiments.

Only completed fits are committed. A killed in-progress fit is repeated; a
finished candidate retains its score instead of thousands of intermediate trees.
No pickle is loaded. Checkpoints contain JSON and native XGBoost UBJSON models.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

LOGGER = logging.getLogger('stage2')


class CheckpointError(ValueError):
    """Persistence failure: abort search, never count as a failed candidate."""


@contextmanager
def checkpoint_io():
    try:
        yield
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CheckpointError(f'Checkpoint I/O failed: {exc}') from exc


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_json(path, value):
    """Commit the marker last so interrupted writes never count as completed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(value, output, sort_keys=True, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def run_identity(frame, stations, args, device):
    """Reject changed data, code, dependencies, or scientific settings."""
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost

    stage2 = Path(__file__).resolve().parent
    sources = list(stage2.glob('*.py')) + list((stage2.parent / 'experiment-stage-1/common').glob('*.py'))
    options = {k: v for k, v in vars(args).items()
               if k not in {'resume', 'checkpoint', 'log_level', 'train'}}
    options['device'] = device
    return {
        'format': 1, 'options': options, 'stations': [str(s) for s in stations],
        'data_hash': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        'columns': [(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)],
        'source_hashes': {str(p.relative_to(stage2.parent)): file_digest(p) for p in sorted(sources)},
        'versions': {'python': list(sys.version_info[:3]), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
    }


class RunCheckpoint:
    def __init__(self, run_dir, identity, resume=False):
        self.root = Path(run_dir) / 'checkpoint'
        self.identity = identity
        self.resume = resume
        self.lock = FileLock(str(self.root.parent / '.training.lock'), timeout=0)

    def __enter__(self):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.lock.acquire()
        except Timeout as exc:
            raise RuntimeError('Another process is already using this run directory') from exc
        try:
            manifest = self.root / 'manifest.json'
            if self.resume:
                if not manifest.is_file():
                    raise ValueError('No resumable checkpoint manifest in this run directory')
                saved = json.loads(manifest.read_text(encoding='utf-8'))
                if digest(saved) != digest(self.identity):
                    raise ValueError('Checkpoint mismatch: data, code, versions, or training arguments changed')
            elif manifest.exists():
                raise ValueError('Checkpoint already exists; use --resume RUN_DIR')
            else:
                atomic_json(manifest, self.identity)
        except BaseException:
            self.lock.release()
            raise
        LOGGER.info('%s checkpoint: %s', 'Resuming' if self.resume else 'Created', self.root)
        return self

    def __exit__(self, *exc):
        self.lock.release()

    def scope(self, identity):
        return FitScope(self.root / 'evaluations' / digest(identity))


class FitScope:
    def __init__(self, root):
        self.root = root

    def load_summary(self):
        path = self.root / 'summary.json'
        if not path.exists():
            return None
        LOGGER.info('Resume: completed candidate %s', self.root.name[:12])
        with checkpoint_io():
            return json.loads(path.read_text(encoding='utf-8'))

    def save_summary(self, result):
        with checkpoint_io():
            atomic_json(self.root / 'summary.json', result)
        # Once the score is committed, these candidate-only models are redundant.
        models = self.root / 'models'
        if models.exists():
            try:
                shutil.rmtree(models)
            except OSError as exc:
                LOGGER.warning('Could not clean redundant checkpoint models: %s', exc)

    def fit(self, key, train, device):
        import xgboost as xgb

        models = self.root / 'models'
        model_path = models / (digest(key) + '.ubj')
        marker = model_path.with_suffix('.json')
        if marker.exists():
            with checkpoint_io():
                saved = json.loads(marker.read_text(encoding='utf-8'))
                if not model_path.exists() or file_digest(model_path) != saved['sha256']:
                    raise ValueError(f'Checkpoint model is missing or corrupt: {model_path}')
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                model.set_params(device=device)
                LOGGER.info('Resume fit: %s', key)
                return model, saved['rounds']
        model, rounds = train()
        with checkpoint_io():
            models.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=models, suffix='.ubj')
            os.close(fd)
            try:
                model.save_model(name)
                with open(name, 'rb+') as source:
                    os.fsync(source.fileno())
                os.replace(name, model_path)
                atomic_json(marker, {'rounds': int(rounds), 'sha256': file_digest(model_path), 'key': key})
            finally:
                Path(name).unlink(missing_ok=True)
        LOGGER.info('Checkpoint fit: %s (%d rounds)', key, rounds)
        return model, rounds
