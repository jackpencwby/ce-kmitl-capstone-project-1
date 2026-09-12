# PM2.5 daily CSV → Google Cloud Storage

`download_pm25_to_gcs.py` downloads GISTDA PM2.5-by-tambon CSV files and uploads them to Google Cloud Storage (GCS). It supports one day or an inclusive date range, with concurrent downloads and uploads. Object names are deterministic, so running the same date again replaces the same GCS object.

## Setup

Requires Python 3.9+ and a GCP identity that can write to the destination bucket.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run once

Set the destination values and run a date:

```bash
export GCS_BUCKET='my-pm25-bucket'
export GCS_PREFIX='raw/gistda/pm25'
export GOOGLE_CLOUD_PROJECT='my-gcp-project'
python download_pm25_to_gcs.py --date 2026-01-02
```

This uploads to:

```
gs://my-pm25-bucket/raw/gistda/pm25/dt=2026-01-02/pm25_tambon_2026-01-02.csv
```

Without `--date`, the script downloads **yesterday in Thailand (Asia/Bangkok)**. Add `--output-dir data` to retain a local CSV copy.

## Date range and workers

The script processes four dates concurrently by default. Start with four workers and increase gradually only if the GISTDA endpoint remains stable:

```bash
python download_pm25_to_gcs.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --workers 4
```

Set `PM25_WORKERS=4` to make this default. If one day fails, other days continue; the script exits with code `1` and reports failed dates.

## GCP IAM and service account for Compute Engine

For production, use a **user-managed service account attached to the Compute Engine VM**. The Google Cloud Python library automatically uses that identity through Application Default Credentials; do not create or copy a JSON service-account key onto the VM.

Set shell variables on an administrator workstation that has `gcloud` configured:

```bash
export PROJECT_ID='my-gcp-project'
export GCS_BUCKET='my-pm25-bucket'
export SERVICE_ACCOUNT_NAME='pm25-uploader'
export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud config set project "$PROJECT_ID"
```

Create the service account and grant it the narrowest predefined role that allows the script to create **and overwrite** its deterministic CSV objects:

```bash
gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
  --display-name='PM2.5 GCS uploader'

gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role='roles/storage.objectUser'
```

`roles/storage.objectCreator` is insufficient here because it cannot overwrite an existing object. Grant `roles/storage.objectUser` at the **bucket**, not project, level; it permits object operations but not bucket administration. Cloud Storage documents these role capabilities in its [IAM roles reference](https://cloud.google.com/storage/docs/access-control/iam-roles).

Attach the service account when creating the VM, or attach it to an existing Compute Engine VM. For an existing VM, stop it first:

```bash
export INSTANCE_NAME='pm25-ingestion-vm'
export ZONE='asia-southeast1-b'

gcloud compute instances stop "$INSTANCE_NAME" --zone="$ZONE"
gcloud compute instances set-service-account "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --service-account="$SERVICE_ACCOUNT_EMAIL" \
  --scopes='https://www.googleapis.com/auth/cloud-platform'
gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE"
```

Use the `cloud-platform` access scope and control actual permissions using the service account's IAM role. Google recommends attaching a user-managed service account to a VM and using Application Default Credentials for production workloads. [Compute Engine service accounts](https://cloud.google.com/compute/docs/access/service-accounts), [ADC with attached service accounts](https://cloud.google.com/docs/authentication/set-up-adc-attached-service-account).

The person attaching the service account normally needs the Service Account User role (`roles/iam.serviceAccountUser`) on that service account, in addition to the relevant Compute Engine permissions. [Google Cloud IAM guidance](https://cloud.google.com/iam/docs/attach-service-accounts).

## Run on the VM with nohup

Create a non-secret environment file; the VM service account supplies credentials automatically:

```bash
sudo tee /etc/pm25-gcs.env > /dev/null <<'EOF'
export GCS_BUCKET=my-pm25-bucket
export GCS_PREFIX=raw/gistda/pm25
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export PM25_WORKERS=4
EOF
```

For a one-time historical import that survives an SSH disconnect:

```bash
cd /opt/pm25-gcs
. /etc/pm25-gcs.env
mkdir -p logs
nohup /opt/pm25-gcs/.venv/bin/python /opt/pm25-gcs/download_pm25_to_gcs.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  > logs/pm25-backfill-202601.log 2>&1 &
echo $! > logs/pm25-backfill-202601.pid
```

Monitor it with `tail -f logs/pm25-backfill-202601.log`. Use cron or a systemd timer for the normal daily run.

## Local development

For development outside GCP, authenticate your own identity with `gcloud auth application-default login` and grant it equivalent bucket access. Do not use a downloaded service-account key unless your organization has no safer authentication option.
