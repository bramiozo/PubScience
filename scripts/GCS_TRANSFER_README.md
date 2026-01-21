# GCS File Transfer Scripts

This directory contains scripts for efficiently transferring files from a local directory to Google Cloud Storage (GCS) with duplicate checking and parallel uploads.

## Overview

The solution consists of two main components:

1. **`move_to_gcs.py`** - Python script that handles the actual file transfer
2. **`run_gcs_transfer.sh`** - Bash wrapper script that runs the transfer in detached mode

## Features

- **Duplicate Detection**: Checks which files already exist in GCS before uploading
- **MD5 Verification**: Compares MD5 hashes to detect if existing files have different content
- **Parallel Uploads**: Uses multiple workers for faster transfers
- **Progress Tracking**: Shows real-time progress with detailed logging
- **Detached Mode**: Can run in background, allowing you to close your terminal
- **Dry Run**: Preview what would be uploaded without actually transferring files

## Prerequisites

1. **Python 3.6+** installed
2. **Google Cloud SDK** or service account credentials
3. Required Python packages:
   ```bash
   pip install -r requirements-gcs.txt
   ```

## Authentication

The scripts use Google Cloud authentication. Set up one of the following:

1. **Application Default Credentials** (recommended):
   ```bash
   gcloud auth application-default login
   ```

2. **Service Account Key**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

## Usage

### Basic Usage

Transfer all files from a local directory to a GCS bucket:

```bash
./run_gcs_transfer.sh /path/to/local/dir my-bucket
```

### With GCS Prefix (subfolder)

Upload files to a specific folder in the bucket:

```bash
./run_gcs_transfer.sh /path/to/local/dir my-bucket data/2024
```

### Dry Run

Preview what would be uploaded without actually transferring:

```bash
./run_gcs_transfer.sh /path/to/local/dir my-bucket --dry-run
```

### Custom Number of Workers

Adjust parallel upload workers (default is 10):

```bash
./run_gcs_transfer.sh /path/to/local/dir my-bucket --workers 20
```

### Foreground Mode

Run in foreground instead of detached mode:

```bash
./run_gcs_transfer.sh /path/to/local/dir my-bucket --no-detach
```

## Direct Python Script Usage

You can also use the Python script directly:

```bash
python move_to_gcs.py \
  --source-dir /path/to/local/dir \
  --bucket my-bucket \
  --prefix data/2024 \
  --workers 15 \
  --dry-run
```

## Monitoring

When running in detached mode, the script provides commands to monitor progress:

1. **Watch real-time logs**:
   ```bash
   tail -f scripts/logs/gcs_transfer_YYYYMMDD_HHMMSS.log
   ```

2. **Check if process is running**:
   ```bash
   ps -p <PID>
   ```

3. **Stop the transfer**:
   ```bash
   kill <PID>
   ```

## Log Files

Logs are stored in `scripts/logs/` with timestamps:
- `gcs_transfer_YYYYMMDD_HHMMSS.log` - Main log file
- `gcs_transfer_YYYYMMDD_HHMMSS.err` - Error log file
- `move_to_gcs.log` - Python script detailed log

## How It Works

1. **File Discovery**: Scans the local directory recursively
2. **GCS Listing**: Lists all existing files in the target GCS location
3. **Comparison**: Identifies which files need to be uploaded
4. **MD5 Check**: For existing files, compares MD5 hashes to detect changes
5. **Parallel Upload**: Uploads files using multiple threads
6. **Progress Tracking**: Shows real-time progress with statistics

## Performance Tips

- **Workers**: Increase workers for better performance on fast connections
- **File Size**: For very large files (>1GB), consider using fewer workers
- **Network**: Performance depends heavily on your internet upload speed

## Troubleshooting

### Authentication Errors
```
Error: 403 Forbidden
```
Solution: Check your GCS permissions and authentication setup

### File Not Found
```
Error: Source directory does not exist
```
Solution: Provide the full absolute path to your local directory

### Process Already Running
If you see a warning about an existing process, you can:
1. Kill the existing process and start new
2. Exit without starting a new transfer

### Check Errors
Always check the error log if transfers fail:
```bash
cat scripts/logs/gcs_transfer_YYYYMMDD_HHMMSS.err
```

## Example Workflow

1. **Check what would be uploaded**:
   ```bash
   ./run_gcs_transfer.sh ~/data/project-files my-project-bucket research/2024 --dry-run
   ```

2. **Start the actual transfer**:
   ```bash
   ./run_gcs_transfer.sh ~/data/project-files my-project-bucket research/2024
   ```

3. **Monitor progress**:
   ```bash
   tail -f scripts/logs/gcs_transfer_*.log
   ```

## Safety Features

- **Duplicate Prevention**: Won't re-upload identical files
- **Overwrite Detection**: Warns when files exist with different content
- **Process Management**: Prevents multiple instances from running
- **Detailed Logging**: Comprehensive logs for debugging