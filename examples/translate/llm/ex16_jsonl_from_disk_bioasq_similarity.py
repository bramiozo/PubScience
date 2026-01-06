# incoming jsonl: question, multichoice answer, meta_info, answer_idx
# outgoing jsonl: question, open answer
#
import json
import os
import re
from pathlib import Path
from time import sleep

import dotenv
from mpmath import ker
from tqdm import tqdm

from pubscience.translate import llm

dotenv.load_dotenv("../../.env")
json_input = os.getenv("BIOASQ")
json_name = Path(json_input).stem

OUTPUT_LOC = os.getenv("BIOASQSim_output")
OUTPUT_LOC_prev = os.getenv("BIOASQ_output")
BATCH_SIZE = 8
QUESTION_ID = "body"
ANSWER_ID = "ideal_answer"
SIM_ID = "snippets"
META_IDS = []
MAX_LENGTH = 4096
MAX_NUM_LINES = 14_369
SLEEP = 3
SYSTEM_PROMPT = "You are a faithful and truthful translator in the medical/clinical domain. The user query is formatted as a dictionary {'source_language':..,'target_language':.., 'text_to_translate':..}, your response should ONLY consist of your translation. IMPORTANT: When you encounter the special marker |SPLIT|, you MUST preserve it EXACTLY as-is in your translation. Do NOT translate this marker. Keep it unchanged in the exact same position."

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

# load jsonl from OUTPUT_LOC_prev
# put in dict with {[id]: [body]}


id_cache = set()
try:
    with open(OUTPUT_LOC, "r") as input_file:
        for line in input_file:
            try:
                d = json.loads(line)
                id_cache.add(d["id"])
            except json.JSONDecodeError:
                print(f"Invalid JSON on line: {line}")
            except KeyError:
                print(f"Missing 'id' key in JSON object: {d}")
except:
    pass

print(f"{len(id_cache)} already in dataset")

with open(json_input, "r") as file:
    list_of_dicts = json.load(file)["questions"]

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    output_list = []
    words_counts = []
    for k, line in tqdm(enumerate(list_of_dicts)):
        if k not in id_cache:
            question_text = line[QUESTION_ID]
            gold_text = line[ANSWER_ID][0]
            silver_texts = [d["text"] for d in line[SIM_ID]]

            text_to_translate = f"{question_text}|SPLIT|{gold_text}|SPLIT|{'|SPLIT|'.join([s for s in silver_texts])}"

            batch.append(text_to_translate)
            batch_ids.append({"id": k})
            words_counts.append(len(text_to_translate.split(" ")))

            if len(batch) == batch_size:
                # Apply your function to the batch here
                # Example: process_batch(batch)
                translated_batch = translator.translate_batch(batch)

                for i in range(len(batch_ids)):
                    d = batch_ids[
                        i
                    ].copy()  # Copy the original dictionary to avoid mutating it
                    translated_text = translated_batch[i]["translated_text"]

                    if translated_text is not None:
                        components = translated_text.split("|SPLIT|")
                        if len(components) >= 2:
                            d.update({QUESTION_ID: components[0]})
                            d.update({"answer": components[1]})
                            if len(components) >= 3:
                                d.update({"answer_snippets": components[2:]})
                            else:
                                d.update({"answer_snippets": []})
                            d.update({"approx_word_count_original": words_counts[i]})
                            d.update(
                                {
                                    "approx_word_count_translated": len(
                                        translated_batch[i]["translated_text"].split(
                                            " "
                                        )
                                    )
                                }
                            )
                            output_list.append(d)
                        else:
                            print(
                                f"Unexpected number of components: for {d['id']}, we found {len(components)} components"
                            )

                with open(OUTPUT_LOC, "a", encoding="utf-8") as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + "\n")

                batch = []
                batch_ids = []
                output_list = []
                words_counts = []
                batch_original = []
                batch_choices = []
                sleep(SLEEP)

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)

        translated_batch = translator.translate_batch(batch)

        for i in range(len(batch_ids)):
            d = batch_ids[i].copy()  # Copy the original dictionary to avoid mutating it
            translated_text = translated_batch[i]["translated_text"]

            if translated_text is not None:
                components = translated_text.split("|SPLIT|")
                if len(components) >= 2:
                    d.update({QUESTION_ID: components[0]})
                    d.update({"answer": components[1]})
                    if len(components) >= 3:
                        d.update({"answer_snippets": components[2:]})
                    else:
                        d.update({"answer_snippets": []})
                    d.update({"approx_word_count_original": words_counts[i]})
                    d.update(
                        {
                            "approx_word_count_translated": len(
                                translated_batch[i]["translated_text"].split(" ")
                            )
                        }
                    )
                    output_list.append(d)
                else:
                    print(
                        f"Unexpected number of components: for {d['id']}, we found {len(components)} components"
                    )

        with open(OUTPUT_LOC, "a", encoding="utf-8") as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + "\n")
