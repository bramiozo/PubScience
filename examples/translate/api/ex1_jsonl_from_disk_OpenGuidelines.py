import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
import time

from pubscience.translate import api
from pubscience.translate.api import google_cost_estimator

dotenv.load_dotenv('../../.env')
jsonl_example = os.getenv('OpenGuidelines')
jsonl_name = Path(jsonl_example).stem

OUTPUT_LOC = os.getenv('ex1_output')
MAX_NUM_LINES = 37_971
TEXT_IDS = ['title', 'clean_text']
ID_COLS = ['id', 'source']
MAX_CHUNK_SIZE = 10_000
API_PROVIDER = 'google'

translator = api.TranslationAPI(provider=API_PROVIDER,
    glossary={}, source_language='en', target_language='nl',
    max_chunk_size=MAX_CHUNK_SIZE)

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

character_counts = []
try:
    with open(jsonl_example, 'r') as file:
        json_iterator = (json.loads(line) for line in file)
        output_list = []
        token_counts = []
        for line in tqdm(json_iterator, total=MAX_NUM_LINES):
            if line['id'] not in id_cache:
                input_text = "\n".join([line[_ID] for _ID in TEXT_IDS])
                trans_ids = {_ID:line[_ID] for _ID in ID_COLS}
                token_count = len(input_text.split(" "))
                character_count = len(input_text)
                character_counts.append(character_count)

                translated = translator.translate(input_text)

                trans_ids.update({'text': translated})
                trans_ids.update({'approx_token_counts_original': token_count})
                trans_ids.update({'approx_token_counts_translated': len(translated.split(" "))})
                trans_ids.update({'approx_character_counts_original': character_count})
                trans_ids.update({'approx_character_counts_translated': len(translated)})

                time.sleep(1)
                with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
                    output_file.write(json.dumps(trans_ids) + '\n')
except KeyboardInterrupt:
    print(f"""Processed {sum(character_counts)} characters in {len(character_counts)} documents.""")

    tot_chars = int(os.popen(f'wc -c {jsonl_example}').read().split()[0])
    print(f"""Total: {tot_chars} characters in total, at a cost of ${api.google_cost_estimator(tot_chars)}""")
    pass
