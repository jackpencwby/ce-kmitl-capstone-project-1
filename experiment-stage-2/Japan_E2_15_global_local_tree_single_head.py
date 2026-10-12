"""Stage 2 experiment 15: global + station LightGBM residuals, single-output."""

from jack_run_stage2 import main


if __name__ == "__main__":
    import sys

    sys.argv.insert(1, "--experiment")
    sys.argv.insert(2, "15")
    raise SystemExit(main())
