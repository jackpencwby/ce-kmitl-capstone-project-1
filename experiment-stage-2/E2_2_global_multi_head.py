"""Stage 2 experiment 2: global LightGBM with multi-output heads."""

from run_stage2 import main


if __name__ == "__main__":
    import sys

    sys.argv.insert(1, "--experiment")
    sys.argv.insert(2, "2")
    raise SystemExit(main())
