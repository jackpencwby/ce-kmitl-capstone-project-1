#!/usr/bin/env python3
"""E1.3 - Global + Local Tree Residual Correction (Experimental_Plan.md 6).

Global XGBoost prediction plus a per-station tree residual model. Residuals
are computed from OUT-OF-FOLD global predictions on the training rows only
(never in-sample), then validation = global + local residual.

Fixed axes: algorithm=XGBoost, forecast=Direct, spatial=No neighbor.

Usage:
  python E1_3.py --train
  python E1_3.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E1_GLOBAL_XGB__LOCAL_TREE_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="xgboost",
        training_strategy="global_local_tree",
        forecast_strategy="direct",
        spatial_mode="none",
        include_station_id=True,
        baseline_xgb=True,
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
