#!/usr/bin/env python3
"""E1.1 - Local Station Model (Experimental_Plan.md section 6).

Training strategy: one independent set of models per station.
Fixed axes for E1: algorithm=XGBoost, forecast=Direct, spatial=No neighbor.
This is the Stage 1 baseline: fixed notebook-inspired XGBoost parameters,
up to 2,000 trees and validation early stopping (100 rounds). It keeps the
current 09-19 feature set and Stage 1 split dates, with no parameter search.

Usage:
  python E1_1.py                       # dry run: describe config + stations
  python E1_1.py --train               # train on ALL eligible stations
  python E1_1.py --train --stations 72 # train only station 72
  python E1_1.py --train --stations 72 36 --source gcs   # pull data from GCS
  python E1_1.py --check-gcs           # verify GCP connectivity from .env
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E1_LOCAL__XGB_NOTEBOOK_FIXED__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        forecast_strategy="direct",
        spatial_mode="none",
        include_station_id=False,
        baseline_xgb=True,
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
