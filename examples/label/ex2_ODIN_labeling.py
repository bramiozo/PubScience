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

dotenv.load_dotenv("../.env")

# add logger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
ID_COL = "studyId_0831"
TEXT_COL = "consult"
LABEL_COL = "label"
AGE_COL = "age"
GENDER_COL = "gender"

PARQUET_LOC = os.environ.get("ODIN")
assert PARQUET_LOC is not None, "PARQUET_LOC is not set"
PARQUET_DIR = os.path.dirname(PARQUET_LOC)
OUTPUT_LOC = os.path.join(PARQUET_DIR, "labeled_texts.jsonl")
kwargs = {
    "system_prompt": "",
    "instruction_list": [],
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "batch_size": 16,
    "max_tokens": 8000,
    "env_loc": "../.run_label.env",
}
SAMPLE_POS_FREQUENCY = 0.1
TextLabeler = text.extract(**kwargs)

# check if OUTPUT_LOC already exists
if os.path.exists(OUTPUT_LOC):
    logger.info(f"Output file {OUTPUT_LOC} already exists")
    file_write = open(OUTPUT_LOC, "a", encoding="utf-8")
    file_write_read = open(OUTPUT_LOC, "r", encoding="utf-8")
    k_list = []
    for k, line in tqdm(enumerate(file_write_read)):
        try:
            d = json.loads(line)
            if d.get("success", False) | d.get("succes", False):
                k_list.append(line["id"])
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON at line {k}")
            continue
    file_write_read.close()

    logger.info(f"Found {len(k_list)} labeled texts")
else:
    logger.info(f"Output file {OUTPUT_LOC} does not exist")
    file_write = open(OUTPUT_LOC, "w", encoding="utf-8")
    k_list = []

df = pd.read_parquet(PARQUET_LOC)
dfjson = df[[ID_COL, TEXT_COL, LABEL_COL, AGE_COL, GENDER_COL]].to_dict(
    orient="records"
)
for k, data in enumerate(tqdm(dfjson)):
    if k in k_list:
        continue

    if k % FREEZE_EVERY_N_STEPS == 0:
        time.sleep(FREEZE_DURATION)

    input_text = data[TEXT_COL]
    input_label = data[LABEL_COL]
    patient_id = data[ID_COL]
    age = data[AGE_COL]
    gender = "mannelijk" if data[GENDER_COL] == 1 else "vrouwelijk"

    # deduce de-identification
    #
    deid_input_text = deducer.deidentify(input_text).deidentified_text

    # added age and gender
    #
    text = f"Consult tekst\n, Patient age: {str(age)}, Patient gender: {gender}.\n\n{deid_input_text}"

    if len(input_text.split()) <= MIN_TOKEN_COUNT:
        continue

    try:
        raw_output = TextLabeler(text)
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
        logger.error(f"Error processing text {text}: {e}")

    output = {
        "k": k,
        "id": patient_id,
        "success": success,
        "text": text,
        "inferred_label": raw_output.content,
        "coded_label": input_label,
        "model": raw_output.model,
        "instructions": "|".join(TextLabeler.instruction_list),
        "meta": raw_output.metadata,
        "proba": np.exp(raw_output.logprob)
        if isinstance(raw_output, float32)
        else np.nan,
        "logprob": raw_output.logprob,
    }

    file_write.write(json.dumps(output) + "\n")

file_write.close()
# write to parquet
import pandas as pd

df = pd.read_json(OUTPUT_LOC, lines=True)
df.to_parquet(os.path.join(PARQUET_DIR, "labeled_texts.parquet"))
