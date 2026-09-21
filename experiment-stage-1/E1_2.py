#!/usr/bin/env python3
"""E1.2 - Global Model (Experimental_Plan.md section 6).

Training strategy: pool every station into one shared model, adding
station identity as a feature. Fixed axes: algorithm=XGBoost,
forecast=Direct, spatial=No neighbor.

Note: --stations still selects which stations enter the pool AND are scored.
Use --stations all (default) for the true global model.

Usage:
  python E1_2.py --train
  python E1_2.py --train --stations 72 36 108
  python E1_2.py --train --source gcs
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E1_GLOBAL__XGB__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="global",
        forecast_strategy="direct",
        spatial_mode="none",
        include_station_id=True,    # global model gets station_id as a feature
        baseline_xgb=True,
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
