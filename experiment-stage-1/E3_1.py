#!/usr/bin/env python3
"""E3.1 - Direct single-output forecasting (Experimental_Plan.md section 8).

Seven separate models per training unit, one per horizon t+1..t+7. Fixed
axes: training=Local, algorithm=XGBoost, spatial=No neighbor. For a fair
comparison with E3.2, both are scored on rows that have all 7 targets
present (handled by the multi-output cohort in E3.2); direct forecasting is
also evaluated per horizon here.

Usage:
  python E3_1.py --train
  python E3_1.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E3_LOCAL__XGB__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        forecast_strategy="direct",
        spatial_mode="none",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
