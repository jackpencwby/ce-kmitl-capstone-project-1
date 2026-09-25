#!/usr/bin/env python3
"""E2.2 - LightGBM algorithm (Experimental_Plan.md section 7).

Algorithm axis switched to LightGBM (objective regression/L2) with fixed
reasonable defaults controlling num_leaves / max_depth / min_child_samples
to limit overfit. Uses the GPU build when the environment supports it, else
CPU (search space unchanged). Other axes fixed at baseline: training=Local,
forecast=Direct, spatial=No neighbor.

Usage:
  python E2_2.py --train
  python E2_2.py --train --stations 72 36
"""

from __future__ import annotations

import sys

from common import cli, config, models, runner


def main() -> int:
    parser = cli.base_parser(__doc__)
    args = parser.parse_args()
    cli.configure_logging(args.log_level)

    spec = models.RunSpec(
        run_id="E2_LOCAL__LGBM__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="lightgbm",
        training_strategy="local",
        walk_forward=True,
        selection_patience=config.SCREENING_EARLY_STOPPING_ROUNDS,
        forecast_strategy="direct",
        spatial_mode="none",
    )
    return runner.execute(spec, args)


if __name__ == "__main__":
    sys.exit(main())
