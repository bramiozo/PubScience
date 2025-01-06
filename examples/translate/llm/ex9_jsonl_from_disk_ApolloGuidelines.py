import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
from time import sleep

from pubscience.translate import llm

dotenv.load_dotenv('../../.env')
json_example = os.getenv('Apollo_guidelines')
json_name = Path(json_example).stem

OUTPUT_LOC = os.getenv('Apollo_guidelines_output')
MAX_NUM_LINES = 99_687
BATCH_SIZE = 8
USE_GPU = True
MAX_LENGTH = 1024
LONG_TEXTS = False
SYSTEM_PROMPT = "You are a faithful and truthful translator in the medical/clinical domain. The user query is formatted as a dictionary {'source_language':..,'target_language':.., 'text_to_translate':..}, your response should ONLY consist of your translation"

vars = {
    'model': 'gpt-4o-mini',
    'provider': 'openai',
    'source_lang': 'english',
    'target_lang': 'dutch',
    'max_tokens': MAX_LENGTH,
    'system_prompt': SYSTEM_PROMPT,
    'env_loc': '../../.run.env',
}

# load translation model
# single: 'vvn/en-to-dutch-marianmt'
# multi: 'facebook/nllb-200-distilled-600M'
translator = llm.TranslationLLM(**vars)

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
                translated_batch = translator.translate_batch(batch)
                batch = []

                for k, _t in enumerate(translated_batch):
                    d = batch_ids[k]
                    d.update({'text': _t['translated_text']})
                    d.update({'approx_token_counts_original': token_counts[k]})
                    d.update({'approx_token_counts_translated': len(_t['translated_text'].split(" "))})
                    output_list.append(d)

                with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + '\n')

                batch = []
                batch_ids = []
                output_list = []
                token_counts = []

                sleep(1)

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)
        translated_batch = translator.translate_batch(batch)
        output_list = [batch_ids[i].update({'text': translated_batch[i]['translated_text']}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')
