#!/usr/bin/env python3
"""Deprecated compatibility entry point; use download_pm25_to_gcs.py instead."""

import logging
import sys

from download_pm25_to_gcs import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.warning("download_pm25_to_s3.py is deprecated; using Google Cloud Storage instead")
    sys.exit(main())
