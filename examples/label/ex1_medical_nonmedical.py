import dotenv
import os
import numpy as np
from pathlib import Path
import json
import random
import time
from tqdm import tqdm
from pubscience.label import text
dotenv.load_dotenv('../.env')

# add logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parse jsonl with labeling tool
# JSONL {"text": abc, "label": 0}

# Results {"text": abc, "label": _label, "proba": proba}
# where _label is a tuple with (label, np.exp(logproba))
# place results in parquet, write streaming

FREEZE_EVERY_N_STEPS = 32
FREEZE_DURATION = 5
MIN_TOKEN_COUNT = 16

JSON_LOC= os.environ.get('MedNonMed')
assert(JSON_LOC is not None), "JSON_LOC is not set"
JSON_DIR= os.path.dirname(JSON_LOC)
OUTPUT_LOC = os.path.join(JSON_DIR, 'labeled_texts.jsonl')
kwargs = {
    'system_prompt': "",
    'instruction_list': [],
    'provider': 'openai',
    'model': 'gpt-4.1-nano-2025-04-14',
    'temperature': 0.25,
    'batch_size': 16,
    'max_tokens': 4096,
    'env_loc': '../.run_label.env'
}
SAMPLE_POS_FREQUENCY=0.1
TextLabeler = text.extract(**kwargs)

# check if OUTPUT_LOC already exists
if os.path.exists(OUTPUT_LOC):
    logger.info(f"Output file {OUTPUT_LOC} already exists")
    file_write = open(OUTPUT_LOC, 'a', encoding='utf-8')
    file_write_read = open(OUTPUT_LOC, 'r', encoding='utf-8')
    # get list of k's with success==True
    k_list = [k for k, line in enumerate(file_write_read) if json.loads(line)['succes']]
    logger.info(f"Found {len(k_list)} labeled texts")
else:
    logger.info(f"Output file {OUTPUT_LOC} does not exist")
    file_write = open(OUTPUT_LOC, 'w', encoding='utf-8')
    k_list = []


with open(JSON_LOC, 'r') as f:
    for k, line in enumerate(tqdm(f)):
        # every FREEZE_EVERY_N_SECS

        if k in k_list:
            continue

        if k % FREEZE_EVERY_N_STEPS == 0:
            time.sleep(FREEZE_DURATION)

        data = json.loads(line)
        input_text = data['text']
        input_label = data['label']

        if len(input_text.split())<=MIN_TOKEN_COUNT:
            continue

        if (input_label == 0):
            try:
                raw_output = TextLabeler(input_text)
                success = True
            except Exception as e:
                raw_output = text.LLMOutput(content="FAIL",
                        logprob=None,
                        model=kwargs['model'],
                        provider=kwargs['provider'],
                        instruction=TextLabeler.instruction_list,
                        metadata=None)
                success = False
                logger.error(f"Error processing text {input_text}: {e}")

            output = {
                'k': k,
                'success': success,
                'text': input_text,
                'label': raw_output.content,
                'assumed_label': input_label,
                'model': raw_output.model,
                'instructions': raw_output.instruction,
                'meta': raw_output.metadata,
                'proba': np.exp(raw_output.logprob) if isinstance(raw_output, float) else np.nan
            }
        else:
            # in this case we assume 1==1 because we have specifically selected medical texts
            # at SAMPLE_POS_FREQUENCY rate we do use the LLM to produce an estimate, for gauging
            # the false negatives
            rsamp = False
            success = True
            if random.random() < SAMPLE_POS_FREQUENCY:
                rsamp = True
                try:
                    raw_output = TextLabeler(input_text)
                    success = True
                except Exception as e:
                    raw_output = text.LLMOutput(content="FAIL",
                            logprob=None,
                            model=kwargs['model'],
                            provider=kwargs['provider'],
                            instruction=TextLabeler.instruction_list,
                            metadata=None)
                    success = False
                    logger.error(f"Error processing text {input_text}: {e}")

            output = {
                'k': k,
                'succes': success,
                'text': input_text,
                'label': input_label if rsamp==False else raw_output.content,
                'assumed_label': input_label,
                'model': 'n/a' if rsamp==False else raw_output.model,
                'instructions': 'n/a' if rsamp==False else raw_output.instruction,
                'meta': 'n/a' if rsamp==False else raw_output.metadata,
                'proba': 'n/a'
            }

        file_write.write(json.dumps(output) + '\n')

file_write.close()
# write to parquet
import pandas as pd
df = pd.read_json(OUTPUT_LOC, lines=True)
df.to_parquet(os.path.join(JSON_DIR, 'labeled_texts.parquet'))
