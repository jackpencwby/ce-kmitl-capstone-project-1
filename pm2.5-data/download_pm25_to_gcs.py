#!/usr/bin/env python3
"""Download daily GISTDA PM2.5 data and upload the CSV files to Cloud Storage."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Iterator
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

ENDPOINT: Final = "https://pm25.gistda.or.th/rest/getPM25by1dTambonAsCSV"
DEFAULT_TIMEOUT_SECONDS: Final = 120
DEFAULT_WORKERS: Final = 4


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def default_date() -> date:
    return datetime.now(ZoneInfo("Asia/Bangkok")).date() - timedelta(days=1)


def make_gcs_key(prefix: str, data_date: date) -> str:
    clean_prefix = prefix.strip("/")
    filename = f"pm25_tambon_{data_date.isoformat()}.csv"
    return "/".join(part for part in (clean_prefix, f"dt={data_date.isoformat()}", filename) if part)


def dates_between(start_date: date, end_date: date) -> Iterator[date]:
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def download_csv(data_date: date, destination: Path, timeout: int) -> None:
    """Download one CSV response, rejecting non-CSV or empty responses."""
    try:
        import requests
    except ImportError as error:
        raise RuntimeError("requests is not installed; run: pip install -r requirements.txt") from error

    url = f"{ENDPOINT}?{urlencode({'dt1': data_date.isoformat(), 'id': 0})}"
    try:
        with requests.get(url, headers={"User-Agent": "pm25-gcs-ingestion/1.0"}, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type not in {"text/csv", "application/csv", "application/octet-stream"}:
                raise RuntimeError(f"unexpected response Content-Type: {content_type or 'missing'}")
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
    except requests.RequestException as error:
        raise RuntimeError(f"could not download data for {data_date}: {error}") from error

    if destination.stat().st_size == 0:
        raise RuntimeError(f"downloaded CSV for {data_date} is empty")


def upload_to_gcs(csv_path: Path, bucket_name: str, object_name: str, project: str | None) -> None:
    """Upload using Application Default Credentials, including an attached VM service account."""
    try:
        from google.cloud import storage
        from google.api_core.exceptions import GoogleAPIError
        from google.auth.exceptions import GoogleAuthError
    except ImportError as error:
        raise RuntimeError("google-cloud-storage is not installed; run: pip install -r requirements.txt") from error

    try:
        client = storage.Client(project=project)
        blob = client.bucket(bucket_name).blob(object_name)
        blob.upload_from_filename(str(csv_path), content_type="text/csv; charset=utf-8")
    except (GoogleAPIError, GoogleAuthError) as error:
        raise RuntimeError(f"could not upload gs://{bucket_name}/{object_name}: {error}") from error


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, help="One data date (YYYY-MM-DD); default: yesterday in Thailand")
    parser.add_argument("--start-date", type=parse_date, help="First data date for a range (YYYY-MM-DD; inclusive)")
    parser.add_argument("--end-date", type=parse_date, help="Last data date for a range (YYYY-MM-DD; inclusive)")
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET"), help="Destination GCS bucket (or GCS_BUCKET)")
    parser.add_argument(
        "--prefix", default=os.getenv("GCS_PREFIX", "pm25/daily"), help="GCS object prefix (or GCS_PREFIX)"
    )
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="GCP project ID, optional")
    parser.add_argument("--output-dir", type=Path, help="Keep downloaded CSV files in this directory")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds")
    parser.add_argument(
        "--workers", type=int, default=int(os.getenv("PM25_WORKERS", DEFAULT_WORKERS)),
        help=f"Concurrent days for a date range (or PM25_WORKERS; default: {DEFAULT_WORKERS})",
    )
    parsed = parser.parse_args()
    if not parsed.bucket:
        parser.error("--bucket or GCS_BUCKET is required")
    if parsed.timeout <= 0 or parsed.workers <= 0:
        parser.error("--timeout and --workers must be greater than zero")
    uses_range = parsed.start_date is not None or parsed.end_date is not None
    if uses_range and (parsed.start_date is None or parsed.end_date is None):
        parser.error("--start-date and --end-date must be supplied together")
    if uses_range and parsed.date is not None:
        parser.error("--date cannot be used with --start-date/--end-date")
    if uses_range and parsed.start_date > parsed.end_date:
        parser.error("--start-date must be on or before --end-date")
    return parsed


def process_date(args: argparse.Namespace, data_date: date, remove_after_upload: bool) -> tuple[date, str | None]:
    object_name = make_gcs_key(args.prefix, data_date)
    csv_path = (args.output_dir / f"pm25_tambon_{data_date.isoformat()}.csv") if args.output_dir else Path("/tmp") / f"pm25_tambon_{data_date.isoformat()}.csv"
    try:
        logging.info("Downloading PM2.5 data for %s", data_date)
        download_csv(data_date, csv_path, args.timeout)
        logging.info("Uploading %s to gs://%s/%s", csv_path, args.bucket, object_name)
        upload_to_gcs(csv_path, args.bucket, object_name, args.project)
        logging.info("Completed: gs://%s/%s", args.bucket, object_name)
        return data_date, None
    except RuntimeError as error:
        return data_date, str(error)
    finally:
        if remove_after_upload:
            csv_path.unlink(missing_ok=True)


def main() -> int:
    args = arguments()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    remove_after_upload = not bool(args.output_dir)
    selected_dates = list(dates_between(args.start_date, args.end_date)) if args.start_date else [args.date or default_date()]
    worker_count = min(args.workers, len(selected_dates))
    logging.info("Processing %d day(s) with %d worker(s)", len(selected_dates), worker_count)

    failures: list[date] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_date, args, data_date, remove_after_upload) for data_date in selected_dates]
        for future in as_completed(futures):
            data_date, error = future.result()
            if error:
                failures.append(data_date)
                logging.error("Failed for %s: %s", data_date, error)
    if failures:
        logging.error("Finished with failures for: %s", ", ".join(day.isoformat() for day in failures))
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
