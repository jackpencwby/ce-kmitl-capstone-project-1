#!/usr/bin/env python3
"""E2.1 - XGBoost algorithm (Experimental_Plan.md section 7).

Pool stations into one shared model per horizon, with station identity as a
feature, like E1.2. Fixed axes: Global / Direct / No neighbor.
E2 uses common/config_e2.py parameters and its existing evaluation policy.

--stations selects both the training pool and the stations scored.
Use --stations all (default) to pool every eligible station.

Usage:
  python E2_1.py --train
  python E2_1.py --train --stations all
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E2_GLOBAL__XGB__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="global",
        forecast_strategy="direct",
        spatial_mode="none",
        include_station_id=True,
        parameter_profile="e2",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
