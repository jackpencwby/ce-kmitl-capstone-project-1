"""Dataset loading, GCP connection, and eligible-group filtering.

The master preprocessed table lives both locally
(``clean-data_preprocess_all_stations_daily.csv``) and in Cloud Storage
(``gs://<bucket>/clean-data/preprocess/all_stations_daily.csv``).

GCP access uses the credentials in the repo ``.env``. The access key there
starts with ``GOOG1E...`` which is a **Cloud Storage HMAC key**, so the
S3-compatible interoperability endpoint (https://storage.googleapis.com) is
used with boto3. The local CSV is preferred when present; ``--source gcs``
forces a download.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

LOGGER = logging.getLogger(__name__)

GCS_S3_ENDPOINT = "https://storage.googleapis.com"


# ---------------------------------------------------------------------------
# .env handling
# ---------------------------------------------------------------------------
def load_env(env_path: Optional[Path] = None) -> dict[str, str]:
    """Parse the repo .env into a dict and export into os.environ.

    A tiny parser is used so python-dotenv is not a hard dependency. Values
    may be wrapped in single or double quotes.
    """
    env_path = env_path or (config.REPO_ROOT / ".env")
    values: dict[str, str] = {}
    if not env_path.exists():
        LOGGER.warning("No .env found at %s; relying on ambient environment", env_path)
        return values
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        values[key] = val
        os.environ.setdefault(key, val)
    return values


def _s3_client(env: dict[str, str]):
    """Build a boto3 S3 client pointed at the GCS interoperability endpoint."""
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - env dependent
        raise RuntimeError(
            "boto3 is required for GCS access; run: pip install -r requirements.txt"
        ) from error

    access_key = env.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env")

    return boto3.client(
        "s3",
        endpoint_url=GCS_S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def download_master_from_gcs(destination: Optional[Path] = None) -> Path:
    """Download the master CSV from GCS via the S3-compatible endpoint."""
    env = load_env()
    bucket = env.get("GCS_BUCKET") or os.getenv("GCS_BUCKET")
    if not bucket:
        raise RuntimeError("GCS_BUCKET is not set in .env")
    destination = destination or config.LOCAL_MASTER_CSV
    client = _s3_client(env)
    LOGGER.info("Downloading gs://%s/%s", bucket, config.GCS_MASTER_OBJECT)
    obj = client.get_object(Bucket=bucket, Key=config.GCS_MASTER_OBJECT)
    body = obj["Body"].read()
    destination.write_bytes(body)
    LOGGER.info("Wrote %s (%d bytes)", destination, len(body))
    return destination


def check_gcs_connection() -> bool:
    """Best-effort connectivity check to GCS. Returns True on success."""
    try:
        env = load_env()
        bucket = env.get("GCS_BUCKET") or os.getenv("GCS_BUCKET")
        client = _s3_client(env)
        client.head_object(Bucket=bucket, Key=config.GCS_MASTER_OBJECT)
        LOGGER.info("GCS connection OK: gs://%s/%s", bucket, config.GCS_MASTER_OBJECT)
        return True
    except Exception as error:  # noqa: BLE001 - report and continue
        LOGGER.warning("GCS connection check failed: %s", error)
        return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_master(source: str = "auto") -> pd.DataFrame:
    """Load the master preprocessed table.

    source:
      - "auto"  : use the local CSV if present, else download from GCS.
      - "local" : require the local CSV.
      - "gcs"   : always download a fresh copy from GCS.
    """
    load_env()
    local = config.LOCAL_MASTER_CSV

    if source == "gcs" or (source == "auto" and not local.exists()):
        download_master_from_gcs(local)
    elif source == "local" and not local.exists():
        raise FileNotFoundError(f"Local master CSV not found: {local}")

    LOGGER.info("Reading master table from %s", local)
    df = pd.read_csv(local, low_memory=False)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df = df.sort_values([config.STATION_ID_COL, config.DATE_COL]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Group eligibility (weather-only station drop) & station selection
# ---------------------------------------------------------------------------
def _group_keys(df: pd.DataFrame) -> list[str]:
    keys = [config.STATION_ID_COL]
    if config.SEGMENT_ID_COL in df.columns:
        keys.append(config.SEGMENT_ID_COL)
    return keys


def eligible_target_stations(df: pd.DataFrame) -> list:
    """Station IDs whose pm25 missingness is under MAX_TARGET_NAN_FRAC.

    Uses only the training window (dates before VALIDATION_START) to decide
    eligibility, matching the leakage rules (plan section 15).
    """
    train = df[df[config.DATE_COL] < pd.Timestamp(config.VALIDATION_START)]
    frac = (
        train.groupby(config.STATION_ID_COL)[config.TARGET_COL]
        .apply(lambda s: s.isna().mean())
    )
    keep = frac[frac <= config.MAX_TARGET_NAN_FRAC].index.tolist()
    return sorted(keep)


def resolve_stations(df: pd.DataFrame, requested: Optional[list]) -> list:
    """Resolve a user-requested station selection to concrete station IDs.

    ``requested`` may be:
      - None or ["all"]  -> every eligible target station
      - a list of ids    -> intersected with eligible stations (warns on drops)

    Station ids in the CSV are integers; string inputs are coerced.
    """
    eligible = eligible_target_stations(df)
    if not requested or (len(requested) == 1 and str(requested[0]).lower() == "all"):
        return eligible

    # Coerce requested ids to the dtype of the column.
    coerced = []
    col_is_int = pd.api.types.is_integer_dtype(df[config.STATION_ID_COL])
    for r in requested:
        try:
            coerced.append(int(r) if col_is_int else str(r))
        except (TypeError, ValueError):
            coerced.append(r)

    eligible_set = set(eligible)
    chosen = [s for s in coerced if s in eligible_set]
    dropped = [s for s in coerced if s not in eligible_set]
    if dropped:
        LOGGER.warning(
            "Ignoring %d requested station(s) that are not eligible pm25 "
            "targets (weather-only or unknown): %s",
            len(dropped), dropped,
        )
    if not chosen:
        raise ValueError(
            "None of the requested stations are eligible pm25 targets. "
            f"Eligible examples: {eligible[:10]}"
        )
    return chosen


def station_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """One (station_id, lat, long) row per station for spatial features."""
    cols = [config.STATION_ID_COL, config.LAT_COL, config.LON_COL]
    coords = (
        df[cols].dropna().groupby(config.STATION_ID_COL, as_index=False).first()
    )
    return coords
