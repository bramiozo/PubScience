import os
import sys
import re
import gc
from tqdm import tqdm

dir_name = sys.argv[1]
files = os.listdir(dir_name)
files = [f for f in files if '.xml' in f]

for filename in files:
    rawout = os.path.join(dir_name, 'out', filename+".full.raw")

    with open(os.path.join(dir_name, filename), encoding='latin1') as rf:
        xmls = [line.strip() for line in rf if len(line.strip())>0]
        xmls = "\r".join(xmls)

    gc.collect()

    with open(rawout, 'w', encoding='utf-8') as wf:
        try:
            idx = 0
            paragraphs = re.findall(r'<p>(.*?)<\/p>', xmls, re.DOTALL)
            for idx, paragraph in tqdm(enumerate(paragraphs)):
                text = re.sub(r'<ext-link[^>]*>.*?</ext-link>', '', paragraph)
                text = re.sub(r'<xref[^>]*>.*?</xref>', '', text)
                wf.write(re.sub(r'<[^>]+>', '', text).strip() + "\n")

            print(f"Number of paragraphs in this chunk: {idx}")
        except Exception as e:
            print(f"XML processing failed with error; {e}")
            print("first characters:")
            print(xmls[:100])
            print("last characters:")
            print(xmls[-100:])
