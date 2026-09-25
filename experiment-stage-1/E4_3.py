#!/usr/bin/env python3
"""E4.3 - Distance-weighted Neighbor (Experimental_Plan.md section 9).

Neighbour PM2.5 is averaged with weights 1/(d+epsilon) from Haversine
distance, so closer stations count more. Missing neighbours on a given day
are renormalised out. The aggregate is lagged 1/3/7 days. Radius, neighbour
count and epsilon are fixed in config before looking at validation. Fixed
axes: training=Local, algorithm=XGBoost, forecast=Direct.

Usage:
  python E4_3.py --train
  python E4_3.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E4_LOCAL__XGB__DIRECT__DISTANCE_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        walk_forward=True,
        selection_patience=config.SCREENING_EARLY_STOPPING_ROUNDS,
        forecast_strategy="direct",
        spatial_mode="distance",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
