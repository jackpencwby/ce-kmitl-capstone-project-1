#!/usr/bin/env python3
"""E4.1 - No Neighbor (Experimental_Plan.md section 9).

Spatial axis held at "no neighbor": only the target station's own PM2.5 lags
and the non-spatial baseline features are used. Fixed axes: training=Local,
algorithm=XGBoost, forecast=Direct. This is the spatial baseline that E4.2-
E4.4 are compared against on the same cohort.

Usage:
  python E4_1.py --train
  python E4_1.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E4_LOCAL__XGB__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        forecast_strategy="direct",
        spatial_mode="none",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
