"""Experiment 13: XGBoost global + per-station tree residual, scalar heads."""
import sys
from run_xgb_residual import main

if __name__ == '__main__':
    raise SystemExit(main(['--experiment', '13', *sys.argv[1:]]))
