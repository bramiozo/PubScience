import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd

from pubscience.translate import ntm

dotenv.load_dotenv('.env')
csv_example_dir = os.getenv('MIMIC3_folder')

# list of file
#
file_list = os.listdir(csv_example_dir)
BATCH_SIZE = 64
USE_GPU = True
TEXT_ID = 'TEXTS'
ID_COL = 'id'
META_COLS = ['ICD9_CODES']
MAX_LENGTH = 228
LONG_TEXTS = True

for file in file_list:
    print(f"Processing {file}...")
    csv_name = Path(file).stem
    name = csv_name.split('.')[0]
    text_df = pd.read_csv(os.path.join(csv_example_dir, file),
        sep=",", encoding='latin1')

    OUTPUT_LOC = os.path.join(os.getenv('ex6_output_folder'), name, ".jsonl")
    MAX_NUM_LINES = text_df.shape[0]

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

    list_of_dicts = text_df.to_dict(orient='records')

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    meta_vals = []
    output_list = []
    words_counts = []
    for _id, line in tqdm(enumerate(list_of_dicts), total=MAX_NUM_LINES):
        if _id not in id_cache:
            input_text = line[TEXT_ID]
            batch.append(input_text)
            id_dict = {ID_COL:_id}
            id_dict.update({'HADM_ID': line['HADM_ID']})
            batch_ids.append(id_dict)
            meta_vals.append({_META:line[_META] for _META in META_COLS})
            words_counts.append(len(input_text.split(" ")))

            # TODO: enable short/long batch processing
            if (len(batch) == batch_size):
                # Apply your function to the batch here
                # Example: process_batch(batch)
                if LONG_TEXTS:
                    if batch_size>1:
                        translated_batch = translator.translate_long_batch(batch,
                            batch_size=32)
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
