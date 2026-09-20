"""Shared command-line interface for every Stage 1 experiment script.

Each E*_*.py builds on this so all scripts share the same flags:

  --train                 Actually run training (default is a dry describe).
  --stations S [S ...]    Which station(s) to train on ("all" or ids).
  --source {auto,local,gcs}   Where to read the master table from.
  --check-gcs             Verify the GCP connection from .env, then continue.
  --max-stations N        Cap number of stations (smoke tests).
  --prefer-cpu            Force CPU even if CUDA is available.
"""

from __future__ import annotations

import argparse
import logging

from . import config


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--train", action="store_true",
        help="Run training. Without this flag the script only describes the "
             "configuration and resolved stations (a dry run).",
    )
    parser.add_argument(
        "--stations", nargs="+", default=["all"],
        help="Station id(s) to train on, or 'all' for every eligible pm25 "
             "station. Example: --stations 72 36. Ineligible ids are dropped.",
    )
    parser.add_argument(
        "--source", choices=["auto", "local", "gcs"], default="auto",
        help="Master table source: local CSV, fresh GCS download, or auto.",
    )
    parser.add_argument(
        "--check-gcs", action="store_true",
        help="Verify GCS connectivity using .env credentials before running.",
    )
    parser.add_argument(
        "--max-stations", type=int, default=None,
        help="Cap the number of stations (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--prefer-cpu", action="store_true",
        help="Force CPU even when a CUDA GPU is available.",
    )
    parser.add_argument(
        "--upload-gcs", action="store_true",
        help="After training, upload the whole run artifact folder to "
             "Cloud Storage under GCS_ARTIFACTS_PREFIX (needs --train).",
    )
    parser.add_argument(
        "--gcs-artifacts-prefix", default=None,
        help="Override the destination folder in the bucket for uploaded "
             f"artifacts (default: '{config.GCS_ARTIFACTS_PREFIX}').",
    )
    parser.add_argument(
        "--keep-local", dest="keep_local", action="store_true", default=True,
        help="Keep the local artifact folder after upload (default).",
    )
    parser.add_argument(
        "--no-keep-local", dest="keep_local", action="store_false",
        help="Delete the local artifact folder after a successful GCS upload.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING).",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
