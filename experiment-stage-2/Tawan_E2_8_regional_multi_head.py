"""Stage 2 experiment E2_8: regional native XGBoost multi-output trees."""

from Tawan_run_stage2 import main


if __name__ == "__main__":
    import sys

    sys.argv[1:1] = ["--experiment", "8"]
    raise SystemExit(main())
