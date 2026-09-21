#!/usr/bin/env python3
"""E1.5 - Global XGBoost + Local MLP Residual (Experimental_Plan.md section 6).

Same residual-correction idea as E1.3, but the per-station corrector is a
small PyTorch MLP instead of a tree. The MLP input is the base feature set
plus the global prediction for that horizon; numeric inputs are standardised
with statistics fit on the training residual rows only. Early stopping,
dropout and L2 weight decay guard against overfitting; stations with too few
rows fall back to residual = 0. Runs on CUDA when available, else CPU.

Fixed axes: algorithm=XGBoost (global), forecast=Direct, spatial=No neighbor.

Usage:
  python E1_5.py --train
  python E1_5.py --train --stations 72 36
  python E1_5.py --train --prefer-cpu
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E1_GLOBAL_XGB__LOCAL_MLP_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="global_local_mlp",
        forecast_strategy="direct",
        spatial_mode="none",
        include_station_id=True,
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
