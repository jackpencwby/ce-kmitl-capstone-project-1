#!/usr/bin/env python3
"""E3.2 - Multi-output forecasting (Experimental_Plan.md section 8).

One system predicts all 7 horizons with seven independent estimators, fitted
explicitly to honor each horizon's parameters (equivalent to independent
multi-output wrapping, not a shared tree head). Trained and scored on rows that
have all 7 targets present, matching the direct-forecasting cohort in E3.1.

Fixed axes: training=Local, algorithm=XGBoost, spatial=No neighbor.

Usage:
  python E3_2.py --train
  python E3_2.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E3_LOCAL__XGB__MULTI__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        walk_forward=True,
        selection_patience=config.SCREENING_EARLY_STOPPING_ROUNDS,
        complete_target_evaluation=True,
        forecast_strategy="multi",
        spatial_mode="none",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
