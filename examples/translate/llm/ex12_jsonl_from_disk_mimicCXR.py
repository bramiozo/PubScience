import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
from time import sleep
from pubscience.translate import llm
import re

dotenv.load_dotenv('../../.env')
json_example = os.getenv('CXR')
json_name = Path(json_example).stem

OUTPUT_LOC = os.getenv('CXR_output')
BATCH_SIZE = 4
TEXT_IDS = ['text']
ID_COLS = ['id']
MAX_LENGTH = 2_500
MAX_NUM_LINES = 227_835
SLEEP = 3
SYSTEM_PROMPT = "You are a faithful and truthful translator in the medical/clinical domain. The user query is formatted as a dictionary {'source_language':..,'target_language':.., 'text_to_translate':..}, your response should ONLY consist of your translation"

vars = {
    'model': 'gemini-1.5-flash',
    'provider': 'google',
    'source_lang': 'english',
    'target_lang': 'dutch',
    'max_tokens': MAX_LENGTH,
    'system_prompt': SYSTEM_PROMPT,
    'temperature': 0.15,
    'env_loc': '../../.run.env',
}

translator = llm.TranslationLLM(**vars)

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

with open(json_example, 'r') as file:
    list_of_dicts = json.load(file)

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    output_list = []
    words_counts = []
    for line in tqdm(list_of_dicts, total=MAX_NUM_LINES):
        if line['id'] not in id_cache:
            input_text = "\n".join([line[_ID] for _ID in TEXT_IDS])
            input_text = re.sub(r"_{2,}", "[PII]", input_text)
            batch.append(input_text)
            batch_ids.append({_ID:line[_ID] for _ID in ID_COLS})
            words_counts.append(len(input_text.split(" ")))

            # TODO: enable short/long batch processing
            if (len(batch) == batch_size):
                # Apply your function to the batch here
                # Example: process_batch(batch)
                translated_batch = translator.translate_batch(batch)

                batch = []
                for i in range(len(batch_ids)):
                    d = batch_ids[i].copy()  # Copy the original dictionary to avoid mutating it
                    translated_text = translated_batch[i]['translated_text']
                    if translated_text is not None:
                        d.update({'text': translated_text})
                        d.update({'approx_word_count_original': words_counts[i]})
                        d.update({'approx_word_count_translated': len(translated_batch[i]['translated_text'].split(" "))})
                        output_list.append(d)

                with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + '\n')

                batch = []
                batch_ids = []
                output_list = []
                words_counts = []
                sleep(SLEEP)

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)
        translated_batch = translator.translate_batch(batch)
        output_list = [batch_ids[i].update({'text': translated_batch[i]['translated_text']}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')
