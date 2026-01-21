import dotenv
import os
import numpy as np
from pathlib import Path
import json
import random
import time
import pandas as pd
from tqdm import tqdm
import deduce
from pubscience.label import text
from numpy import float32
import pyarrow as pa
import pyarrow.parquet as pq

dotenv.load_dotenv("../.env")

# add logger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress DEBUG logging from third-party libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
file_handler = logging.FileHandler("label_processing.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# deduce class
deducer = deduce.Deduce()

# parse jsonl with labeling tool
# JSONL {"text": abc, "label": 0}

# Results {"text": abc, "label": _label, "proba": proba}
# where _label is a tuple with (label, np.exp(logproba))
# place results in parquet, write streaming

FREEZE_EVERY_N_STEPS = 32
FREEZE_DURATION = 1
MIN_TOKEN_COUNT = 16
ID_COL = "studyId_0888"
TEXT_COL = "verslagtekst"
GPT_VERSION = "gpt-4.1"

PARQUET_LOC = os.environ.get("ARGUS")
assert PARQUET_LOC is not None, "PARQUET_LOC is not set"
PARQUET_DIR = os.path.dirname(PARQUET_LOC)
OUTPUT_LOC = os.path.join(
    PARQUET_DIR, f"{GPT_VERSION}_labeled_texts_single_dutch_v2.parquet"
)
LOG_LOC = os.path.join(PARQUET_DIR, "label_processing.log")

# Add file handler for logging
file_handler = logging.FileHandler(LOG_LOC)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.info(f"Logging to {LOG_LOC}")
kwargs = {
    "system_prompt": "",
    "instruction_list": [],
    "provider": "openai",
    "model": GPT_VERSION,
    "temperature": 0.0,
    "batch_size": 16,
    "max_tokens": 2048,
    "env_loc": "../.run_label.env",
    "logger": logger,
}
TextLabeler = text.extract(**kwargs)

# Define the parquet schema
parquet_schema = pa.schema(
    [
        ("k", pa.int64()),
        ("id", pa.int64()),
        ("success", pa.bool_()),
        ("text", pa.string()),
        ("inferred_label", pa.string()),
        ("model", pa.string()),
        ("instructions", pa.string()),
        ("meta", pa.string()),
        ("proba", pa.float64()),
        ("logprob", pa.float64()),
    ]
)

# check if OUTPUT_LOC already exists
if os.path.exists(OUTPUT_LOC):
    logger.info(f"Output file {OUTPUT_LOC} already exists")
    file_write_read = pd.read_parquet(OUTPUT_LOC)
    k_list = file_write_read["id"].tolist()
    logger.info(f"Found {len(k_list)} labeled texts")

    print(file_write_read.head())
else:
    logger.info(f"Output file {OUTPUT_LOC} does not exist")
    k_list = []

df = pd.read_parquet(PARQUET_LOC)
dfjson = df[[ID_COL, TEXT_COL]].to_dict(orient="records")

# Buffer for batch writing
BATCH_SIZE = 16
batch_buffer = []


def write_batch(writer, batch_buffer):
    """Write a batch of records to the parquet file."""
    if not batch_buffer:
        return
    batch_df = pd.DataFrame(batch_buffer)
    table = pa.Table.from_pandas(batch_df, schema=parquet_schema)
    writer.write_table(table)


# Use context manager to ensure proper closing of the parquet writer
success_list = []
with pq.ParquetWriter(OUTPUT_LOC, parquet_schema, compression="snappy") as writer:
    for k, data in enumerate(dfjson):
        if data[ID_COL] in k_list:
            continue

        if k % FREEZE_EVERY_N_STEPS == 0:
            time.sleep(FREEZE_DURATION)

        input_text = data[TEXT_COL]
        patient_id = data[ID_COL]

        # deduce de-identification
        deid_input_text = deducer.deidentify(input_text).deidentified_text

        try:
            raw_output = TextLabeler(deid_input_text)
            success = True
        except Exception as e:
            raw_output = text.LLMOutput(
                content="FAIL",
                logprob=None,
                model=kwargs["model"],
                provider=kwargs["provider"],
                instruction="|".join(TextLabeler.instruction_list),
                metadata=None,
            )
            success = False
            logger.error(f"Error processing text {deid_input_text}: {e}")

        output = {
            "k": k,
            "id": patient_id,
            "success": success,
            "text": input_text,
            "inferred_label": raw_output.content,
            "model": raw_output.model,
            "instructions": "|".join(TextLabeler.instruction_list),
            "meta": json.dumps(raw_output.metadata) if raw_output.metadata else None,
            "proba": float(np.exp(raw_output.logprob))
            if isinstance(raw_output.logprob, (int, float, np.number))
            else np.nan,
            "logprob": float(raw_output.logprob)
            if isinstance(raw_output.logprob, (int, float, np.number))
            else np.nan,
        }
        batch_buffer.append(output)
        success_list.append(success)

        # Write batch when buffer is full
        if len(batch_buffer) >= BATCH_SIZE:
            print(f"Writing to buffer...: {k}, with {sum(success_list)} successes")
            write_batch(writer, batch_buffer)
            batch_buffer = []
            success_list = []

    # Write any remaining records
    write_batch(writer, batch_buffer)

logger.info(f"Finished writing to {OUTPUT_LOC}")
