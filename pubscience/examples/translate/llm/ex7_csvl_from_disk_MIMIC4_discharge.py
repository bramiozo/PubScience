import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import re

from pubscience.translate import ntm

dotenv.load_dotenv('.env')
csv_example = os.getenv('MIMIC4_discharge')
csv_name = Path(csv_example).stem

text_df = pd.read_csv(csv_example, sep=",", encoding='latin1')
text_df = text_df[['note_type', 'text']]

OUTPUT_LOC = os.getenv('ex7_output')
BATCH_SIZE = 64
USE_GPU = True
TEXT_IDS = ['text']
ID_COL = 'id'
META_COLS = ['note_type']
MAX_LENGTH = 228
MAX_NUM_LINES = text_df.shape[0]
LONG_TEXTS = True


# load translation model
# single: 'vvn/en-to-dutch-marianmt'
# multi: 'facebook/nllb-200-distilled-600M'
translator = ntm.TranslationNTM(model_name='vvn/en-to-dutch-marianmt', multilingual=False,
                max_length=MAX_LENGTH, use_gpu=USE_GPU, target_lang='nld_Latn')

id_cache = set()
try:
    with open(OUTPUT_LOC, 'r') as input_file:
        for line in input_file:
            try:
                d = json.loads(line)
                id_cache.add(d['id'])
            except json.JSONDecodeError:
                print(f"Invalid JSON on line: {line}")
            except KeyError:
                print(f"Missing 'id' key in JSON object: {d}")
except:
    pass

print(f"{len(id_cache)} already in dataset")

list_of_dicts = text_df[['note_type', 'text']].to_dict(orient='records')

batch_size = BATCH_SIZE
batch = []
batch_ids = []
meta_vals = []
output_list = []
words_counts = []
for _id, line in tqdm(enumerate(list_of_dicts), total=MAX_NUM_LINES):
    if _id not in id_cache:
        input_text = line['text']
        # remove repeating occurrences of "_"
        input_text = re.sub(r"_{2,}", " ", input_text)
        batch.append(input_text)
        batch_ids.append({ID_COL:_id})
        meta_vals.append({_META:line[_META] for _META in META_COLS})
        words_counts.append(len(input_text.split(" ")))

        # TODO: enable short/long batch processing
        if (len(batch) == batch_size):
            # Apply your function to the batch here
            # Example: process_batch(batch)
            if LONG_TEXTS:
                if batch_size>1:
                    translated_batch = translator.translate_long_batch(batch,
                        batch_size=8)
                else:
                    translated_batch = [translator.translate_long(batch[0])]
            else:
                translated_batch = translator.translate_batch(batch)

            batch = []
            for i in range(len(batch_ids)):
                d = batch_ids[i].copy()  # Copy the original dictionary to avoid mutating it
                d.update({'text': translated_batch[i]})
                d.update(meta_vals[i])
                d.update({'approx_word_count_original': words_counts[i]})
                d.update({'approx_word_count_translated': len(translated_batch[i].split(" "))})
                output_list.append(d)

            with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
                for item in output_list:
                    output_file.write(json.dumps(item) + '\n')

            batch = []
            batch_ids = []
            meta_vals = []
            output_list = []
            words_counts = []

# Process any remaining lines in the last batch
if batch:
    # Apply your function to the batch here
    # Example: process_batch(batch)
    translated_batch = translator.translate_batch(batch)
    output_list = [batch_ids[i].update({'text': translated_batch[i]}) for i in range(len(batch_ids))]
    with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
        for item in output_list:
            output_file.write(json.dumps(item) + '\n')
