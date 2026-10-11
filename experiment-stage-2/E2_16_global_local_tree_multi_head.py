"""Stage 2 experiment 16: global + station LightGBM residuals, multi-output."""

from run_stage2 import main


if __name__ == "__main__":
    import sys

    sys.argv.insert(1, "--experiment")
    sys.argv.insert(2, "16")
    raise SystemExit(main())
