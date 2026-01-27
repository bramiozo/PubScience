import json
import os
from pathlib import Path
from time import sleep

import dotenv
from datasets import load_dataset
from tqdm import tqdm

from pubscience.translate import llm

dotenv.load_dotenv("../../.env")
hf_dataset_name = os.getenv("MIRIAD", "miriad/miriad-4.4M")
OUTPUT_LOC = os.getenv("MIRIAD_output")
BATCH_SIZE = 8
DATA_SIZE = 4.4e6
IDX_ID = "qa_id"
QUESTION_ID = "question"
ANSWER_ID = "answer"
MAX_LENGTH = 1024
SLEEP = 2
SYSTEM_PROMPT = (
    "You are a faithful and truthful translator in the medical/clinical domain. "
    "The user query is formatted as a dictionary {'source_language':..,'target_language':.., 'text_to_translate':..}, "
    "your response should ONLY consist of your translation. IMPORTANT: When you encounter the special marker |SPLIT|, "
    "you MUST preserve it EXACTLY as-is in your translation. Do NOT translate this marker. Keep it unchanged in the exact same position."
)

META_IDS = [
    "paper_id",
    "paper_url",
    "paper_title",
    "year",
    "venue",
    "specialty",
]

vars = {
    "model": "gpt-4.1-mini",
    "provider": "openai",
    "source_lang": "english",
    "target_lang": "dutch",
    "max_tokens": MAX_LENGTH,
    "system_prompt": SYSTEM_PROMPT,
    "temperature": 0.15,
    "env_loc": "../../.run_translate.env",
}

translator = llm.TranslationLLM(**vars)

# Build id_cache for resuming
id_cache = set()
if OUTPUT_LOC and Path(OUTPUT_LOC).exists():
    with open(OUTPUT_LOC, "r") as input_file:
        for line in input_file:
            try:
                d = json.loads(line)
                id_cache.add(d["id"])
            except Exception:
                continue
elif OUTPUT_LOC:
    print(f"{OUTPUT_LOC} does not exist yet, creating..")
    id_cache = set()
    os.makedirs(os.path.dirname(OUTPUT_LOC), exist_ok=True)

print(f"{len(id_cache)} already in dataset")

# Load HF dataset in streaming mode
hf_dataset = load_dataset(hf_dataset_name, split="train", streaming=True)

batch, batch_ids, words_counts = [], [], []


def process_and_write_batch(batch, batch_samples, words_counts):
    translated_batch = translator.translate_batch(batch)
    output_list = []
    for i, sample in enumerate(batch_samples):
        result = {"id": sample[IDX_ID]}
        # Add meta fields
        for meta_key in META_IDS:
            result[meta_key] = sample.get(meta_key)
        translated_text = translated_batch[i]["translated_text"]
        if translated_text:
            components = translated_text.split("|SPLIT|")
            result[QUESTION_ID] = components[0]
            result[ANSWER_ID] = components[1] if len(components) > 1 else ""
            result["approx_word_count_original"] = words_counts[i]
            result["approx_word_count_translated"] = len(translated_text.split(" "))
            output_list.append(result)
        else:
            print(f"Translation failed for {sample[IDX_ID]}")
    if output_list:
        with open(OUTPUT_LOC, "a", encoding="utf-8") as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + "\n")


batch, batch_samples, words_counts = [], [], []

for sample in tqdm(hf_dataset, total=DATA_SIZE):
    if sample[IDX_ID] not in id_cache:
        text_to_translate = f"{sample[QUESTION_ID]}|SPLIT|{sample[ANSWER_ID]}"
        batch.append(text_to_translate)
        batch_samples.append(sample)  # Keep the full sample for meta fields
        words_counts.append(len(text_to_translate.split(" ")))
        if len(batch) == BATCH_SIZE:
            process_and_write_batch(batch, batch_samples, words_counts)
            batch, batch_samples, words_counts = [], [], []
            sleep(SLEEP)

if batch:
    process_and_write_batch(batch, batch_samples, words_counts)
