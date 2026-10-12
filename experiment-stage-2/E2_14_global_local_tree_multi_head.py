"""Experiment 14: XGBoost global + per-station native vector-tree residual."""
import sys
from run_xgb_residual import main

if __name__ == '__main__':
    raise SystemExit(main(['--experiment', '14', *sys.argv[1:]]))
