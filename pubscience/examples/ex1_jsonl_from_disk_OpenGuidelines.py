import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm

from pubscience.translate import ntm

dotenv.load_dotenv('.env')
jsonl_example = os.getenv('JSON_FILE')
jsonl_name = Path(jsonl_example).stem

OUTPUT_LOC = os.getenv('ex1_output')
MAX_NUM_LINES = 37_971
BATCH_SIZE = 16
USE_GPU = True
TEXT_IDS = ['title', 'clean_text']
ID_COLS = ['id', 'source']
MAX_LENGTH = 496
LONG_TEXTS = True


# load translation model
translator = ntm.TranslationNTM(model_name='vvn/en-to-dutch-marianmt', multilingual=False, max_length=MAX_LENGTH, use_gpu=USE_GPU)

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

with open(jsonl_example, 'r') as file:
    json_iterator = (json.loads(line) for line in file)

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    output_list = []
    token_counts = []
    for line in tqdm(json_iterator, total=MAX_NUM_LINES):
        if line['id'] not in id_cache:
            input_text = "\n".join([line[_ID] for _ID in TEXT_IDS])
            batch.append(input_text)
            batch_ids.append({_ID:line[_ID] for _ID in ID_COLS})
            token_counts.append(1.5*len(input_text.split(" ")))

            # TODO: enable short/long batch processing
            if (len(batch) == batch_size):
                # Apply your function to the batch here
                # Example: process_batch(batch)
                if LONG_TEXTS:
                    translated_batch = translator.translate_long_batch_v2(batch)
                else:
                    translated_batch = translator.translate_batch(batch)
                batch = []

                for k, _t in enumerate(translated_batch):
                    d = batch_ids[k]
                    d.update({'text': _t})
                    d.update({'approx_token_counts': token_counts[k]})
                    output_list.append(d)

                with open(OUTPUT_LOC, 'a') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + '\n')

                batch = []
                batch_ids = []
                output_list = []
                token_counts = []

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)
        translated_batch = translator.translate_batch(batch)
        output_list = [batch_ids[i].update({'text': translated_batch[i]}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')
