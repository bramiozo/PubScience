"""
{"institute": "Erasmus", "file": "xx.pdf", "pseudo_pagenum": 0, "text": "bla"}
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import json


def collect_texts_to_parquet(folder_path, output_path):
    """
    Collect JSONL files from a folder and save as Parquet with Institute column.

    Args:
        folder_path (str): Path to folder containing [institute]_texts.jsonl files
        output_path (str): Path where to save the output parquet file
    """
    all_records = []

    # Convert to Path object for easier handling
    folder = Path(folder_path)

    # Find all files matching the pattern [institute]_texts.jsonl
    institutes = set()
    for jsonl_file in folder.glob("*_texts.jsonl"):
        # Extract institute name from filename
        institute_name = jsonl_file.stem.replace("_texts", "")
        institutes.add(institute_name)
        print(f"Processing {jsonl_file.name} for institute: {institute_name}")

        # Read JSONL file
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Add Institute column
                    record["Institute"] = institute_name
                    all_records.append(record)
                except Exception as e:
                    print(f"Error parsing line in {jsonl_file.name}: {e}")
                    continue

    if not all_records:
        print("No records found in any JSONL files.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    print(f"Collected {len(df)} records from {len(institutes)} institutes")
    print(f"Columns: {list(df.columns)}")

    # Save as Parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect JSONL files from a folder and save as Parquet with Institute column"
    )
    parser.add_argument(
        "--folder_path", help="Path to folder containing [institute]_texts.jsonl files"
    )
    parser.add_argument(
        "--output_path", help="Path where to save the output parquet file"
    )

    args = parser.parse_args()
    collect_texts_to_parquet(args.folder_path, args.output_path)
