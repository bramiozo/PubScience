import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm

from pubscience.translate import ntm

dotenv.load_dotenv('../../.env')
json_example = os.getenv('Apollo_wiki')
json_name = Path(json_example).stem

OUTPUT_LOC = os.getenv('Apollo_wiki_output_folder')
MAX_NUM_LINES = 238_657
BATCH_SIZE = 4
INNER_BATCH_SIZE = 16
USE_GPU = True
MAX_LENGTH = 256 # 456 for nllb-200-distilled-600M, 228 for maria-nmt
LONG_TEXTS = False
USE_QUANTISATION = False

# load translation model
# single: 'vvn/en-to-dutch-marianmt'
# multi: 'facebook/nllb-200-distilled-600M'
translator = ntm.TranslationNTM(model_name='vvn/en-to-dutch-marianmt',
    multilingual=False, max_length=MAX_LENGTH,
    use_gpu=USE_GPU, target_lang='nld_Latn', use_quantisation=USE_QUANTISATION)
#translator = ntm.TranslationNTM(model_name='vvn/en-to-dutch-marianmt',
#multilingual=False, max_length=MAX_LENGTH,
# use_gpu=USE_GPU, target_lang='nld_Latn', use_quantisation=USE_QUANTISATION)

id_cache = set()
try:
    with open(OUTPUT_LOC, 'r', encoding='utf-8') as input_file:
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

with open(json_example, 'r') as file:
    docs = file.readlines()

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    output_list = []
    token_counts = []
    for doc_count, line in tqdm(enumerate(docs), total=MAX_NUM_LINES):
        if doc_count not in id_cache:
            input_text = line
            batch.append(input_text)
            batch_ids.append({'id': doc_count})
            token_counts.append(len(input_text.split(" ")))

            # TODO: enable short/long batch processing
            if (len(batch) == batch_size):
                # Apply your function to the batch here
                # Example: process_batch(batch)
                if LONG_TEXTS:
                    if batch_size>1:
                        translated_batch = translator.translate_long_batch(batch,
                            batch_size=INNER_BATCH_SIZE)
                    else:
                        translated_batch = [translator.translate_long(batch[0])]
                else:
                    translated_batch = translator.translate_batch(batch)
                batch = []

                for k, _t in enumerate(translated_batch):
                    d = batch_ids[k]
                    d.update({'text': _t})
                    d.update({'approx_token_counts_original': token_counts[k]})
                    d.update({'approx_token_counts_translated': len(_t.split(" "))})
                    output_list.append(d)

                with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item, ensure_ascii=False) + '\n')

                batch = []
                batch_ids = []
                output_list = []
                token_counts = []

if batch:
    if LONG_TEXTS:
        if batch_size > 1:
            translated_batch = translator.translate_long_batch(batch, batch_size=INNER_BATCH_SIZE)
        else:
            translated_batch = [translator.translate_long(batch[0])]
    else:
        translated_batch = translator.translate_batch(batch)

    for i, translated_text in enumerate(translated_batch):
        d = batch_ids[i]
        d.update({'text': translated_text})
        d.update({'approx_token_counts_original': token_counts[i]})
        d.update({'approx_token_counts_translated': len(translated_text.split(" "))})
        output_list.append(d)

    with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
        for item in output_list:
            output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
