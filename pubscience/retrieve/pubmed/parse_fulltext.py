import os
import sys
import re
import gc
from tqdm import tqdm

dir_name = sys.argv[1]
max_words = int(sys.argv[2])
files = os.listdir(dir_name)
files = [f for f in files if '.xml' in f]

for filename in files:
    rawout = os.path.join(dir_name, 'out', filename+".full.raw")

    with open(os.path.join(dir_name, filename), mode='r', encoding='latin1') as rf:
        xmls = [line.strip() for line in rf if len(line.strip())>0]
        xmls = "\r".join(xmls)

    gc.collect()
    print(f"Processing {filename}")
    with open(rawout, 'w', encoding='utf-8') as wf:
        try:
            # Set maximum word count for paragraph chunks
            MAX_WORDS = max_words
            SEPARATOR = " "

            bodies = re.findall(r'<body>(.*?)<\/body>', xmls, re.DOTALL)
            idx = 0
            for body in tqdm(bodies):
                paragraphs = re.findall(r'<p>(.*?)<\/p>', body, re.DOTALL)
                current_chunk = []
                current_word_count = 0
                for paragraph in paragraphs:
                    # Clean the paragraph text
                    text = re.sub(r'<ext-link[^>]*>.*?</ext-link>', '', paragraph)
                    text = re.sub(r'<xref[^>]*>.*?</xref>', '', text)
                    text = re.sub(r'<[^>]+>', '', text).strip()

                    # Skip empty paragraphs
                    if not text:
                        continue

                    # Count words in this paragraph
                    words_in_paragraph = len(re.findall(r'\S+', text))

                    # If adding this paragraph would exceed the limit, write the current chunk
                    if current_word_count > 0 and (current_word_count + words_in_paragraph) > MAX_WORDS:
                        wf.write(SEPARATOR.join(current_chunk) + "\n")
                        current_chunk = [text]
                        current_word_count = words_in_paragraph
                        idx += 1
                    else:
                        # Add to the current chunk
                        current_chunk.append(text)
                        current_word_count += words_in_paragraph

                # Write any remaining content
                if current_chunk:
                    wf.write(SEPARATOR.join(current_chunk) + "\n")
                    idx += 1
        except Exception as e:
            print(f"XML processing failed with error; {e}")
            print("first characters:")
            print(xmls[:100])
            print("last characters:")
            print(xmls[-100:])
