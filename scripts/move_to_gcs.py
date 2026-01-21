#!/usr/bin/env python3
"""
Move files from local directory to Google Cloud Storage bucket with duplicate checking.

Usage:
    python move_to_gcs.py --source-dir <DIR> --bucket <BUCKET> [--prefix <PREFIX>] [--dry-run]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.cloud.exceptions import NotFound
import hashlib
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('move_to_gcs.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_file_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def list_gcs_files(bucket_name: str, prefix: str = "") -> Set[str]:
    """List all files in a GCS bucket with optional prefix."""
    logger.info(f"Listing files in gs://{bucket_name}/{prefix}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Get all blobs with the specified prefix
        blobs = bucket.list_blobs(prefix=prefix)

        # Return set of blob names (relative paths)
        gcs_files = set()
        for blob in blobs:
            # Remove prefix if it exists to get relative path
            relative_path = blob.name
            if prefix and relative_path.startswith(prefix):
                relative_path = relative_path[len(prefix):].lstrip('/')
            gcs_files.add(relative_path)

        logger.info(f"Found {len(gcs_files)} files in GCS bucket")
        return gcs_files

    except Exception as e:
        logger.error(f"Error listing GCS files: {e}")
        raise


def list_local_files(directory: Path) -> List[Tuple[Path, str]]:
    """List all files in local directory recursively."""
    logger.info(f"Scanning local directory: {directory}")

    local_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            # Get relative path from the source directory
            relative_path = file_path.relative_to(directory)
            local_files.append((file_path, str(relative_path)))

    logger.info(f"Found {len(local_files)} files in local directory")
    return local_files


def upload_file(file_path: Path, bucket_name: str, blob_name: str, dry_run: bool = False) -> bool:
    """Upload a single file to GCS."""
    try:
        if dry_run:
            logger.info(f"[DRY RUN] Would upload {file_path} to gs://{bucket_name}/{blob_name}")
            return True

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if file exists and compare MD5
        if blob.exists():
            blob.reload()
            remote_md5 = blob.md5_hash
            local_md5 = get_file_md5(file_path)

            if remote_md5 == local_md5:
                logger.debug(f"File {blob_name} already exists with same content, skipping")
                return False
            else:
                logger.warning(f"File {blob_name} exists but has different content, overwriting")

        # Upload the file
        blob.upload_from_filename(str(file_path))
        logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
        return True

    except Exception as e:
        logger.error(f"Error uploading {file_path}: {e}")
        raise


def move_files_to_gcs(source_dir: str, bucket_name: str, prefix: str = "",
                      dry_run: bool = False, max_workers: int = 10):
    """Main function to move files from local directory to GCS."""
    source_path = Path(source_dir).resolve()

    if not source_path.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        sys.exit(1)

    if not source_path.is_dir():
        logger.error(f"{source_dir} is not a directory")
        sys.exit(1)

    # List files in GCS
    try:
        gcs_files = list_gcs_files(bucket_name, prefix)
    except Exception as e:
        logger.error(f"Failed to list GCS files: {e}")
        sys.exit(1)

    # List local files
    local_files = list_local_files(source_path)

    # Determine files to upload
    files_to_upload = []
    files_already_exist = []

    for file_path, relative_path in local_files:
        gcs_path = f"{prefix}/{relative_path}" if prefix else relative_path

        if relative_path in gcs_files:
            files_already_exist.append((file_path, gcs_path))
        else:
            files_to_upload.append((file_path, gcs_path))

    # Log summary
    logger.info(f"\nSummary:")
    logger.info(f"  Total local files: {len(local_files)}")
    logger.info(f"  Files already in GCS: {len(files_already_exist)}")
    logger.info(f"  Files to upload: {len(files_to_upload)}")

    if files_already_exist:
        logger.info("\nFiles already in GCS (will check MD5):")
        for _, gcs_path in files_already_exist[:10]:  # Show first 10
            logger.info(f"  - {gcs_path}")
        if len(files_already_exist) > 10:
            logger.info(f"  ... and {len(files_already_exist) - 10} more")

    if not files_to_upload and not files_already_exist:
        logger.info("No files to process")
        return

    # Upload files with progress bar
    all_files = files_to_upload + files_already_exist
    uploaded_count = 0
    skipped_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_file = {}
        for file_path, gcs_path in all_files:
            future = executor.submit(upload_file, file_path, bucket_name, gcs_path, dry_run)
            future_to_file[future] = (file_path, gcs_path)

        # Process completed uploads with progress bar
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file_path, gcs_path = future_to_file[future]
                try:
                    was_uploaded = future.result()
                    if was_uploaded:
                        uploaded_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    error_count += 1
                pbar.update(1)

    # Final summary
    logger.info(f"\nTransfer complete:")
    logger.info(f"  Uploaded: {uploaded_count}")
    logger.info(f"  Skipped (identical): {skipped_count}")
    logger.info(f"  Errors: {error_count}")

    if not dry_run and uploaded_count > 0:
        logger.info(f"\nFiles have been uploaded to gs://{bucket_name}/{prefix if prefix else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Move files from local directory to Google Cloud Storage with duplicate checking"
    )
    parser.add_argument(
        "--source-dir", "-s",
        required=True,
        help="Source directory containing files to upload"
    )
    parser.add_argument(
        "--bucket", "-b",
        required=True,
        help="GCS bucket name (without gs:// prefix)"
    )
    parser.add_argument(
        "--prefix", "-p",
        default="",
        help="Optional prefix (folder path) in the GCS bucket"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Perform a dry run without actually uploading files"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of parallel upload workers (default: 10)"
    )

    args = parser.parse_args()

    # Validate bucket name
    bucket_name = args.bucket.replace("gs://", "")

    logger.info(f"Starting file transfer:")
    logger.info(f"  Source: {args.source_dir}")
    logger.info(f"  Destination: gs://{bucket_name}/{args.prefix if args.prefix else ''}")
    logger.info(f"  Dry run: {args.dry_run}")

    move_files_to_gcs(
        source_dir=args.source_dir,
        bucket_name=bucket_name,
        prefix=args.prefix,
        dry_run=args.dry_run,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
