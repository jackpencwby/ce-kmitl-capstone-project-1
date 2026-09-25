#!/usr/bin/env python3
"""E2.1 - XGBoost algorithm (Experimental_Plan.md section 7).

Algorithm axis held at XGBoost with objective reg:squarederror and fixed
reasonable defaults (no Optuna in Stage 1). Other axes fixed at baseline:
training=Local, forecast=Direct, spatial=No neighbor. E2 runs real walk-forward
CV and selects tree counts on a training-only chronological tail before refit;
E1's validation-selected-model test policy is separate.

Usage:
  python E2_1.py --train
  python E2_1.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E2_LOCAL__XGB__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        walk_forward=True,
        selection_patience=config.SCREENING_EARLY_STOPPING_ROUNDS,
        forecast_strategy="direct",
        spatial_mode="none",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
