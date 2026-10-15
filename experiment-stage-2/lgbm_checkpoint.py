"""Native LightGBM checkpoints; reuse the atomic markers and locks from E13/14."""
import logging
import json
import os
import tempfile
from pathlib import Path

from xgb_checkpoint import (RunCheckpoint, FitScope, CheckpointError, atomic_json,
                            checkpoint_io, digest, file_digest, run_identity)

LOGGER = logging.getLogger('stage2')


def complete_run(checkpoint, experiment, test_skipped):
    root = checkpoint.root.parent
    files = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and checkpoint.root not in p.parents and not p.name.startswith('.')}
    result = {'experiment': experiment, 'test_skipped': test_skipped, 'files': files}
    atomic_json(checkpoint.root / 'completed.json', {'result': result, 'sha256': digest(result)})


def completed_run(checkpoint, experiment, test_skipped):
    marker = checkpoint.root / 'completed.json'
    if not marker.exists():
        return False
    with checkpoint_io():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        result = saved['result']
        if (digest(result) != saved['sha256'] or result['experiment'] != experiment
                or result['test_skipped'] is not test_skipped or not result['files']):
            raise ValueError('Invalid completed-run marker')
        root = checkpoint.root.parent.resolve()
        for name, expected in result['files'].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or file_digest(path) != expected:
                raise ValueError(f'Missing or corrupt final artifact: {name}')
    return True


def lightgbm_identity(frame, stations, args):
    import lightgbm
    identity = run_identity(frame, stations, args, args.device)
    identity['algorithm'] = 'lightgbm'
    identity['versions']['lightgbm'] = lightgbm.__version__
    identity['options'].pop('run_dir', None)
    return identity


class RestoredLGBM:
    """Prediction/importance/export interface around a native Booster, no pickle load."""
    def __init__(self, booster, params):
        self.booster_ = booster
        self.params = params

    def predict(self, X):
        return self.booster_.predict(X, num_threads=int(os.environ.get('OMP_NUM_THREADS', '1')))

    @property
    def feature_importances_(self):
        return self.booster_.feature_importance(importance_type=self.params.get('importance_type', 'split'))

    def get_params(self, deep=True):
        return self.params.copy()


class LGBMCheckpoint(RunCheckpoint):
    def scope(self, identity):
        return LGBMFitScope(self.root / 'evaluations' / digest(identity))


class LGBMFitScope(FitScope):
    def fit(self, key, train, device):
        import json
        import lightgbm as lgb
        models = self.root / 'models'
        model_path = models / (digest(key) + '.txt')
        marker = model_path.with_suffix('.json')
        if marker.exists():
            with checkpoint_io():
                saved = json.loads(marker.read_text(encoding='utf-8'))
                if saved['key'] != key or file_digest(model_path) != saved['sha256']:
                    raise ValueError(f'Missing or corrupt LightGBM checkpoint: {model_path}')
                if not isinstance(saved['rounds'], int) or saved['rounds'] < 1:
                    raise ValueError('Invalid boosting round count')
                try:
                    booster = lgb.Booster(model_file=str(model_path))
                except lgb.basic.LightGBMError as exc:
                    raise CheckpointError(f'Cannot load LightGBM checkpoint: {exc}') from exc
                LOGGER.info('Resume fit: %s', key)
                return RestoredLGBM(booster, saved['params']), saved['rounds']
        model, rounds = train()
        with checkpoint_io():
            models.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(dir=models, suffix='.txt')
            os.close(fd)
            try:
                model.booster_.save_model(name)
                with open(name, 'rb+') as output:
                    os.fsync(output.fileno())
                os.replace(name, model_path)
                atomic_json(marker, {'key': key, 'rounds': int(rounds),
                                     'sha256': file_digest(model_path), 'params': model.get_params()})
            finally:
                Path(name).unlink(missing_ok=True)
        LOGGER.info('Checkpoint fit: %s (%d rounds)', key, rounds)
        return model, rounds
