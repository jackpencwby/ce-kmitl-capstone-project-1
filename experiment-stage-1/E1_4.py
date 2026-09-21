#!/usr/bin/env python3
"""E1.4 - Regional Model (Experimental_Plan.md section 6).

Stations are grouped into geographic regions (North / Northeast / Central /
South, derived from the province column, not from validation results). One
shared model set is trained per region; stations outside a defined region
fall back to the Central bucket.

Fixed axes: algorithm=XGBoost, forecast=Direct, spatial=No neighbor.

Usage:
  python E1_4.py --train
  python E1_4.py --train --stations 72 36 108
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E1_REGIONAL__XGB__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="regional",
        forecast_strategy="direct",
        spatial_mode="none",
        include_station_id=True,    # regional model keeps station_id ...
        include_region_id=True,     # ... and region_id (plan section 3.2)
        baseline_xgb=True,
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
