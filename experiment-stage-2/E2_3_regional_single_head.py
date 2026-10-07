"""Stage 2 experiment 3: regional LightGBM with direct/single-output heads."""

from run_stage2 import main


if __name__ == "__main__":
    import sys

    sys.argv.insert(1, "--experiment")
    sys.argv.insert(2, "3")
    raise SystemExit(main())
