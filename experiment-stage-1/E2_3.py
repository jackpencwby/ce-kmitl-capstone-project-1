#!/usr/bin/env python3
"""E2.3 - Gradient Boosting Regressor (Experimental_Plan.md section 7).

Algorithm axis switched to sklearn.ensemble.GradientBoostingRegressor, the
classical boosting baseline. CPU only by design; timing must not be compared
directly against GPU algorithms (plan section 3.3), though accuracy is
comparable given the shared split/features/seed. Other axes fixed at
baseline: training=Local, forecast=Direct, spatial=No neighbor.

Usage:
  python E2_3.py --train
  python E2_3.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E2_LOCAL__GBR__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="gbr",
        training_strategy="local",
        walk_forward=True,
        forecast_strategy="direct",
        spatial_mode="none",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
