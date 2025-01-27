import dotenv
import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import re # for removing repeated underscores
from pubscience.translate import ntm

dotenv.load_dotenv('../../.env')
csv_example_dir = os.getenv('MIMIC3_folder')

# list of file
#
file_list = os.listdir(csv_example_dir)
BATCH_SIZE = 3
USE_GPU = True
TEXT_ID = 'TEXTS'
ID_COL = 'id'
META_COLS = ['ICD9_CODES']
MAX_LENGTH = 128
LONG_TEXTS = True

DEID_NORMALISATION_REGEX = {
    'DATE_1':               (r'\[\*\*[0-9]{4,5}\-[0-9]{1,2}\-[0-9]{1,2}\*\*\]', '[DATE]'),
    'DATE_2':               (r'\[\*\*Month \(only\) [0-9]{1,4}\*\*\]', '[DATE]'),
    'PATIENT_NAME_1':       (r'\[\*\*Name[0-9]{1,4} \(NameIs\)\s[0-9]{0,4}\*\*\]', '[PATIENT_NAME]'),
    'PATIENT_NAME_2':       (r'\[\*\*Known firstname [0-9]{1,6}\*\*\]', '[PATIENT_FIRST_NAME]'),
    'PATIENT_NAME_3':       (r'\[\*\*Known lastname [0-9]{1,6}\*\*\]', '[PATIENT_LAST_NAME]'),
    'FIRST_NAME_1':         (r'\[\*\*First Name[0,9]{0,6} \(LF\)\s\*\*\]', '[FIRST_NAME]'),
    'FIRST_NAME_2':         (r'\[\*\*First Name[0,9]{0,6} \(Titles\)\s\*\*\]', '[FIRST_NAME]'),
    'FIRST_NAME_3':         (r'\[\*\*First Name[0,9]{0,6} \(NamePattern[0-9]{0,6}\)\s\*\*\]', '[FIRST_NAME]'),
    'FIRST_NAME_4':         (r'\[\*\*First Name[0,9]{0,6} \(NamePattern[0-9]{0,6}\)\s[0-9]{1,4}\*\*\]', '[FIRST_NAME]'),
    'LAST_NAME_1':          (r'\[\*\*Last Name[0,9]{0,6} \(STitle\)\s\*\*\]', '[LAST_NAME]'),
    'LAST_NAME_2':          (r'\[\*\*Last Name[0,9]{0,6} \(Titles\)\s\*\*\]', '[LAST_NAME]'),
    'LAST_NAME_3':          (r'\[\*50\*Last Name[0,9]{0,6} \(NamePattern[0-9]{0,6}\)\s\*\*\]', '[LAST_NAME]'),
    'LAST_NAME_4':          (r'\[\*\*Last Name[0,9]{0,6} \(NamePattern[0-9]{0,6}\)\s[0-9]{1,4}\*\*\]', '[LAST_NAME]'),
    'LAST_NAME_5':          (r'\[\*\*Known lastname [0,9]{0,5}\*\*\]', '[LAST_NAME]'),
    'DOCTOR_ID':            (r'\[\*\*MD Number([0,9]{0,3}) [0-9]{1,4}\*\*\]', '[DOCTOR_ID]'),
    'HOSPITAL_1':           (r'\[\*\*Hospital[0-9]{0,2} [0-9]{1,4}\*\*\]', '[HOSPITAL]'),
    'HOSPITAL_2':           (r'\[\*\*Hospital[0-9]{0,2} \*\*\]', '[HOSPITAL]'),
    'HOSPITAL_3':           (r'\[\*\*Location \(un\) [0-9]{0,4}\*\*\]', '[HOSPITAL]'),
    'HOSPITAL_4':           (r'\[\*\*Hospital Unit Name [0-9]{1,4}\*\*\]', '[HOSPITAL]'),
    'HOSPITAL_5':           (r'\[\*\*Hospital Ward Name [0-9]{1,4}\*\*\]', '[HOSPITAL]'),
    'DOCTOR_LAST_NAME_1':   (r'\[\*\*Doctor Last Name \*\*\]', '[DOCTOR_LAST_NAME]'),
    'DOCTOR_FIRST_NAME_1':  (r'\[\*\*Doctor First Name \*\*\]', '[DOCTOR_FIRST_NAME]'),
    'MEASUREMENT_VALUE_1':  (r'\[\*\*[0-9]{1,2}\-[0-9]{1,2}\*\*\]', '[DETAIL]'),
    'MEASUREMENT_VALUE_2':  (r'\[\*\*[0-9]{1,6}\*\*\]', '[MEAS_VALUE]'),
    'PHONENUMBER_1':        (r'\[\*\*Telephone/Fax \([0-9]{1,2}\) [0-9]{3,6}\*\*\]', '[PHONE_NUMBER]'),
    'ADDRESS':              (r'\[\*\*Apartment Address\([0-9]{1,2}\) [0-9]{1,6}\*\*\]', '[ADDRESS]'),
    'IDENTIFIER_1':         (r'\[\*\*Numeric Identifier [0-9]{1,4}\*\*\]', '[IDENTIFIER]'),
    'IDENTIFIER_2':         (r'\[\*\*Job Number [0-9]{1,6}\*\*\]', '[IDENTIFIER]'),
    'IDENTIFIER_3':         (r'\[\*\*Clip Number [0-9]{1,6}\*\*\]', '[IDENTIFIER]'),
    'IDENTIFIER_4':         (r'\[\*\*Clip Number (Radiologie) [0-9]{1,6}\*\*\]', '[IDENTIFIER]'),
    'SPURIOUS_REP_1':       (r'\b(\w+)(?:\s+\1\b)+', r'\1'),
    'SPURIOUS_REP_2':       (r'(\W)\1+', r'\1'),
}

# re.compile the regexes
DEID_NORMALISATION_REGEX = {k: (re.compile(v[0]), v[1]) for k, v in DEID_NORMALISATION_REGEX.items()}

# load translation model
# single: 'vvn/en-to-dutch-marianmt'
# multi: 'facebook/nllb-200-distilled-600M'
# multi: 

print("LOADING MODEL...")
translator = ntm.TranslationNTM(model_name='facebook/nllb-200-distilled-600M', multilingual=True,
                max_length=MAX_LENGTH, use_gpu=USE_GPU, target_lang='nld_Latn')

for file in file_list:
    print(f"Processing {file}...")
    csv_name = Path(file).stem
    name = file.split('.')[0]
    text_df = pd.read_csv(os.path.join(csv_example_dir, file), sep=",", encoding='latin1')

    OUTPUT_LOC = os.path.join(os.getenv('MIMIC3_output'), f"{name}.jsonl")
    print(f"Output location: {OUTPUT_LOC}")

    MAX_NUM_LINES = text_df.shape[0]

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

    list_of_dicts = text_df.to_dict(orient='records')

    batch_size = BATCH_SIZE
    batch = []
    batch_ids = []
    meta_vals = []
    output_list = []
    words_counts = []
    for _id, line in tqdm(enumerate(list_of_dicts), total=MAX_NUM_LINES):
        if _id not in id_cache:
            input_text = line[TEXT_ID]

            input_text = re.sub(r'_{2,}', r' ', input_text)
            for k, v in DEID_NORMALISATION_REGEX.items():
                input_text = re.sub(v[0], v[1], input_text)

            batch.append(input_text)
            id_dict = {ID_COL:_id}
            id_dict.update({'HADM_ID': line['HADM_ID']})
            batch_ids.append(id_dict)
            meta_vals.append({_META:line[_META] for _META in META_COLS})
            words_counts.append(len(input_text.split(" ")))

            # TODO: enable short/long batch processing
            if (len(batch) == batch_size):
                # Apply your function to the batch here
                # Example: process_batch(batch)
                if LONG_TEXTS:
                    if batch_size>1:
                        translated_batch = translator.translate_long_batch(batch,
                            batch_size=12)
                    else:
                        translated_batch = [translator.translate_long(batch[0])]
                else:
                    translated_batch = translator.translate_batch(batch)

                batch = []
                for i in range(len(batch_ids)):
                    d = batch_ids[i].copy()  # Copy the original dictionary to avoid mutating it
                    d.update({'text': translated_batch[i]})
                    d.update(meta_vals[i])
                    d.update({'approx_word_count_original': words_counts[i]})
                    d.update({'approx_word_count_translated': len(translated_batch[i].split(" "))})
                    output_list.append(d)

                with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
                    for item in output_list:
                        output_file.write(json.dumps(item) + '\n')

                batch = []
                batch_ids = []
                meta_vals = []
                output_list = []
                words_counts = []

    # Process any remaining lines in the last batch
    if batch:
        # Apply your function to the batch here
        # Example: process_batch(batch)
        translated_batch = translator.translate_batch(batch)
        output_list = [batch_ids[i].update({'text': translated_batch[i]}) for i in range(len(batch_ids))]
        with open(OUTPUT_LOC, 'a', encoding='utf-8') as output_file:
            for item in output_list:
                output_file.write(json.dumps(item) + '\n')

    # reset GPU memory/cache
    translator.reset()
