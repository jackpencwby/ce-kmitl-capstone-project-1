#!/usr/bin/env python3
"""E4.2 - Unweighted Neighbor (Experimental_Plan.md section 9).

Adds the simple average of neighbouring stations' PM2.5 as an extra signal,
lagged 1/3/7 days so the target day never sees a neighbour's same-day value
(leakage rule 4-5). Neighbours are the up-to-K nearest stations within the
configured radius. Fixed axes: training=Local, algorithm=XGBoost,
forecast=Direct.

Usage:
  python E4_2.py --train
  python E4_2.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E4_LOCAL__XGB__DIRECT__UNWEIGHTED_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        walk_forward=True,
        selection_patience=config.SCREENING_EARLY_STOPPING_ROUNDS,
        forecast_strategy="direct",
        spatial_mode="unweighted",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
