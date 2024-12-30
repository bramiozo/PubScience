import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm

from pubscience.translate import ntm

dotenv.load_dotenv('.env')
text_example = os.getenv('AGCT_FILE')
text_name = Path(text_example).stem

OUTPUT_LOC = os.getenv('ex2_output')
MAX_NUM_LINES = 421_216
BATCH_SIZE = 16
USE_GPU = True
MAX_LENGTH = 501
LONG_TEXTS = False

# load translation model
# single: 'vvn/en-to-dutch-marianmt'
# multi: 'facebook/nllb-200-distilled-600M'
translator = ntm.TranslationNTM(model_name='vvn/en-to-dutch-marianmt',
                                multilingual=False,
                                max_length=MAX_LENGTH,
                                use_gpu=USE_GPU,
                                target_lang='nld_Latn')

lines_processed = 0
try:
    with open(OUTPUT_LOC, 'r') as input_file:
        for line in input_file:
            lines_processed += 1
except:
    pass

print(f"{lines_processed} already in dataset")

with open(text_example, 'r', encoding='latin1') as file:
    text_iterator = (line for line in file)

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    output_list = []
    token_counts = []
    processed_count = 0
    for line in tqdm(text_iterator, total=MAX_NUM_LINES):
        if processed_count > lines_processed:
            input_text = line
            batch.append(input_text)
            batch_ids.append({'id': processed_count+lines_processed})
            token_counts.append(len(input_text.split(" ")))

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

                for k, _t in enumerate(translated_batch):
                    d = batch_ids[k]
                    d.update({'text': _t})
                    d.update({'approx_token_counts': token_counts[k]})
                    output_list.append(d)

                with open(OUTPUT_LOC, 'a', encoding='latin1') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + '\n')

                batch = []
                batch_ids = []
                output_list = []
                token_counts = []

        processed_count += 1

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)
        translated_batch = translator.translate_batch(batch)
        output_list = [batch_ids[i].update({'text': translated_batch[i]}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a', encoding='latin1') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')
