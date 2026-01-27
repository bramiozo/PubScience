# incoming jsonl: question, multichoice answer, meta_info, answer_idx
# outgoing jsonl: question, open answer
#
import json
import os
import re
from pathlib import Path
from time import sleep

import dotenv
from tqdm import tqdm

from pubscience.translate import llm

dotenv.load_dotenv("../../.env")
json_input = os.getenv("MEDQA")
json_name = Path(json_input).stem

OUTPUT_LOC = os.getenv("MEDQA_output")
BATCH_SIZE = 8
QUESTION_ID = "question"
ANSWER_ID = "answer"
OPTIONS_ID = "options"
META_IDS = ["meta_id"]
MAX_LENGTH = 10240
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
    "temperature": 0.25,
    "env_loc": "../../.run_translate.env",
}

translator = llm.TranslationLLM(**vars)

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
    list_of_lines = file.readlines()
    list_of_dicts = [json.loads(line) for line in list_of_lines]

    batch_size = BATCH_SIZE
    batch = []
    batch_choices = []
    batch_ids = []
    output_list = []
    words_counts = []
    for k, line in tqdm(enumerate(list_of_dicts)):
        if k not in id_cache:
            question_text = line[QUESTION_ID]
            answer_choice = line[ANSWER_ID]
            options_dict = line[OPTIONS_ID]
            options_text = f"The options are: {','.join(options_dict.values())}."

            input_text = f"{question_text}\n{options_text}\n"
            answer_text = f"The answer is: {options_dict[answer_choice]}"

            text_to_translate = f"{input_text}|SPLIT|{answer_text}"

            batch.append(text_to_translate)
            batch_choices.append(answer_choice)
            batch_ids.append({"id": k})
            words_counts.append(len(text_to_translate.split(" ")))

            # TODO: enable short/long batch processing
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
                        if len(components) == 2:
                            d.update({QUESTION_ID: components[0]})
                            d.update({"answer": components[1]})
                            d.update({f"original_{QUESTION_ID}_answer": batch[i]})
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
                if len(components) == 2:
                    d.update({QUESTION_ID: components[0]})
                    d.update({"answer": components[1]})
                    d.update({f"original_{QUESTION_ID}_answer": batch[i]})
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
