"""Stage 2 experiment E2_7: regional XGBoost with seven independent heads."""

from Tawan_run_stage2 import main


if __name__ == "__main__":
    import sys

    sys.argv[1:1] = ["--experiment", "7"]
    raise SystemExit(main())
