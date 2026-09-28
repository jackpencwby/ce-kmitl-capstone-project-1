#!/usr/bin/env python3
"""E4.4 - Wind + Distance-weighted Neighbor (Experimental_Plan.md section 9).

Neighbour weight combines distance decay with wind alignment:

    w_ij = max(0, cos(delta_theta))^p / (d_ij + epsilon)

where delta_theta is the angular difference between the meteorological wind
direction ("wind from", degrees) at the target and the bearing from target
to neighbour, so upwind neighbours are up-weighted. On calm days or when the
weight sum is zero, it falls back to the distance-weighted value. The
aggregate is lagged 1/3/7 days (no future values). Fixed axes:
training=Local, algorithm=XGBoost, forecast=Direct.

Usage:
  python E4_4.py --train
  python E4_4.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E4_LOCAL__XGB__DIRECT__WIND_DISTANCE__SEED42",
        algorithm="xgboost",
        training_strategy="local",
        walk_forward=True,
        selection_patience=config.SCREENING_EARLY_STOPPING_ROUNDS,
        forecast_strategy="direct",
        spatial_mode="wind",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
