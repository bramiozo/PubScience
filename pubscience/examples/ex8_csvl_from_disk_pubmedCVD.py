import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import re # for removing repeated underscores
from pubscience.translate import ntm

dotenv.load_dotenv('.env')
cvd_dir = os.getenv('PMC_CVD_folder')

# list of file
#
file_list = os.listdir(cvd_dir)
BATCH_SIZE = 48
USE_GPU = True
TEXT_ID = 'TEXTS'
ID_COL = 'id'
MAX_LENGTH = 228
LONG_TEXTS = False

# load translation model
# single: 'vvn/en-to-dutch-marianmt'
# multi: 'facebook/nllb-200-distilled-600M'
translator = ntm.TranslationNTM(model_name='vvn/en-to-dutch-marianmt', multilingual=False,
                max_length=MAX_LENGTH, use_gpu=USE_GPU, target_lang='nld_Latn')

for file in file_list:
    print(f"Processing {file}...")
    name = re.sub(r'_+', '_', file.split('.')[0])

    OUTPUT_LOC = os.path.join(os.getenv('ex8_output_folder'), f"{name}.jsonl")
    print(f"Output location: {OUTPUT_LOC}")

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

    with open(os.path.join(cvd_dir, file), 'r', encoding='latin-1') as f:
        list_of_texts = f.readlines()
    MAX_NUM_LINES = len(list_of_texts)

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    meta_vals = []
    output_list = []
    words_counts = []
    for _id, line in tqdm(enumerate(list_of_texts), total=MAX_NUM_LINES):
        if _id not in id_cache:
            input_text = line
            batch.append(input_text)
            id_dict = {ID_COL:_id}
            batch_ids.append(id_dict)
            words_counts.append(len(input_text.split(" ")))

            # TODO: enable short/long batch processing
            if (len(batch) == batch_size):
                # Apply your function to the batch here
                # Example: process_batch(batch)
                if LONG_TEXTS:
                    if batch_size>1:
                        translated_batch = translator.translate_long_batch(batch,
                            batch_size=24)
                    else:
                        translated_batch = [translator.translate_long(batch[0])]
                else:
                    translated_batch = translator.translate_batch(batch)

                batch = []
                for i in range(len(batch_ids)):
                    d = batch_ids[i].copy()  # Copy the original dictionary to avoid mutating it
                    d.update({'text': translated_batch[i]})
                    d.update({'approx_word_count_original': words_counts[i]})
                    d.update({'approx_word_count_translated': len(translated_batch[i].split(" "))})
                    output_list.append(d)

                with open(OUTPUT_LOC, 'a', encoding='latin-1') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + '\n')

                batch = []
                batch_ids = []
                output_list = []
                words_counts = []

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)
        translated_batch = translator.translate_batch(batch)
        output_list = [batch_ids[i].update({'text': translated_batch[i]}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a', encoding='latin-1') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')

    # reset GPU memory/cache
    translator.reset()