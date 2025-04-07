import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import re # for removing repeated underscores
from pubscience.translate import llm
import argparse

dotenv.load_dotenv('../../.env')
cvd_dir = os.getenv('PMC_CVD_folder')

argparser = argparse.ArgumentParser()
argparser.add_argument('--skip_existing', action='store_true', help='Skip existing files in the output folder')

# list of file
#
file_list = os.listdir(cvd_dir)
BATCH_SIZE = 4
TEXT_ID = 'TEXTS'
ID_COL = 'id'
MAX_LENGTH = 16_000
MIN_LENGTH = 30
SYSTEM_PROMPT = "You are a faithful and truthful translator in the medical/clinical domain. The user query is formatted as a dictionary {'source_language':..,'target_language':.., 'text_to_translate':..}, your response should ONLY consist of your translation"

vars = {
    'model': 'gemini-2.0-flash',
    'provider': 'google',
    'source_lang': 'english',
    'target_lang': 'dutch',
    'max_tokens': MAX_LENGTH,
    'system_prompt': SYSTEM_PROMPT,
    'env_loc': '../../.run.env',
}

translator = llm.TranslationLLM(**vars)

for file in file_list:
    print(f"Processing {file}...")
    name = re.sub(r'_+', '_', file.split('.')[0])

    out_file_list = os.listdir(os.getenv('PMC_CVD_output'))
    if (f"{name}.jsonl" in out_file_list) & (argparser.parse_args().skip_existing):
        print(f"Skipping {name}.jsonl")
        continue

    OUTPUT_LOC = os.path.join(os.getenv('PMC_CVD_output'), f"{name}.jsonl")
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
            input_text = re.sub(r'\[\,*\]', '', input_text)
            wc = len(input_text.split(" "))
            words_counts.append(wc)

            if (wc > MIN_LENGTH):
                if (wc > MAX_LENGTH):
                    # chop up the input_text in blocks of MAX_LENGTH//2
                    chunks = []
                    words = input_text.split()
                    chunk_size = MAX_LENGTH // 2

                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        chunks.append(chunk)

                    for cdx, chunk in enumerate(chunks):
                        batch.append(chunk)
                        id_dict = {ID_COL: f"{_id}_{cdx}"}
                        batch_ids.append(id_dict)
                        words_counts.append(len(chunk.split()))
                else:
                    batch.append(input_text)
                    id_dict = {ID_COL:_id}
                    batch_ids.append(id_dict)

                # TODO: enable short/long batch processing
                if (len(batch) >= batch_size):
                    # Apply your function to the batch here
                    # Example: process_batch(batch)
                    translated_batch = translator.translate_batch(batch)

                    batch = []
                    for i in range(len(batch_ids)):
                        d = batch_ids[i].copy()  # Copy the original dictionary to avoid mutating it
                        d.update({'text': translated_batch[i]['translated_text']})
                        d.update({'approx_word_count_original': words_counts[i]})
                        d.update({'approx_word_count_translated': len(translated_batch[i]['translated_text'].split(" "))})
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
        output_list = [batch_ids[i].update({'text': translated_batch[i]['translated_text']}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a', encoding='latin-1') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')
