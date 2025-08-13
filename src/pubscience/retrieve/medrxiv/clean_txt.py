import re
import os
import sys
import argparse
from tqdm import tqdm

def clean(txt):
    lines = txt.split("\n")
    clean = []
    for line in lines:
        line = re.sub(r'\s+',' ', line)
        line = re.sub(r'[0-9]+', '#', line)

        if len(line.strip())>20:
            clean.append(line)
    return "\n".join(clean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="What folder would you like to clean?")
    parser.add_argument('--input-folder', type=str, metavar='input_folder', dest='input_folder')
    args = parser.parse_args()

    print(f'processing text files in {args.input_folder}')
    for f in tqdm(os.listdir(args.input_folder)):
        floc = os.path.join(args.input_folder,f)
        if os.path.isfile(floc):
            with open(floc, 'r') as rf:
                get_text = rf.read()
                get_clean = clean(get_text)
            with open(f'cleaned_{f}', 'w') as wf:
                wf.write(get_clean)
        else:
            print(f'{f} not recognised as a regular file?')

