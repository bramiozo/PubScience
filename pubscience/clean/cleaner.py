import argparse
import re
import os
import sys
import argparse
from benedict import benedict
from bs4 import BeautifulSoup
from tqdm import tqdm
import mimetypes
import io
import math
from collections import Counter

'''
Takes raw data in from:
raw PubMed/PMC, MIMICIII in tabular format, raw text or xml. --> CHOOSE MOTHERF*CKER!

Outputs:
a. Cleaned, text-only corpus without tables/lists, per line of some maximum length.
b. ditto but per document in tabular format with identifiers/labels, if available
'''

#text = re.sub(r'(http\:\/\/[A-z0-9.\/\?\-\=]+)|(www\.[A-z0-9.\/\?\-\=]+)', '<WEBLINK>', text)
#text = re.sub(r'DOI\s[0-9\.\/\-\_]+', '<SCIREF>', text, flags=re.IGNORECASE)
#text = re.sub(r'[\w\.]\@[\w]+\.\w{2,5}', '<EMAIL>' , text, flags=re.IGNORECASE)
#text = re.sub(r'\ufffd', '', text)
#text = re.sub(r"[^\x00-\x7F]+", "", text)
#text = deduce_fun(text)            
#text = re.sub(r'[0-9]', '#', text)
#text = re.sub(r'\#{3,}', '###', text)
#text = re.sub(r'\-\#\>', ">", text)
#text = re.sub(r'([A-z])([.:])', r'\1 \2', text)
#text = re.sub(r'\n', ' ', text)
#text = re.sub(r'\s{2,}', ' ', text)

encoding_fixes = [('Ã«', 'ë'),
                  ('Ã¯', 'ï'),
                  ('Ã¨', 'è'),
                  ('Ã©', 'é'),
                  ('Ã¶', 'ö')]

def get_char_seq(txt):
    """
    Convert the text string into a list (sequence) of character IDs.
    Here we simply use the built-in `ord` to get the ASCII/Unicode code.
    """
    return [ord(ch) for ch in txt]

def compute_entropy(counter):
    """
    Given a frequency counter of items in a window, compute Shannon entropy.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def get_char_entr(char_id_seq, window=5, stride=1):
    """
    Compute local entropies over a sliding window, and record the spans (start, end).
    Returns:
        char_id_entr (list[float]): A list of entropy values for each window.
        spans (list[tuple]): Corresponding (start, end) indices for each window.
    """
    char_id_entr = []
    spans = []
    n = len(char_id_seq)
    
    for start in range(0, n - window + 1, stride):
        end = start + window
        # Count frequencies of each character ID in the window
        window_ids = char_id_seq[start:end]
        freq_counter = Counter(window_ids)
        
        e = compute_entropy(freq_counter)
        char_id_entr.append(e)
        spans.append((start, end))
    
    return char_id_entr, spans

def get_cand_dupl(char_id_entr, spans, threshold=1.0):
    """
    Identify spans of low entropy (potential spurious repetition).
    
    Args:
        char_id_entr: A list of entropy values for each window.
        spans: The corresponding spans (start, end) for each window.
        threshold: Entropy threshold below which a span is considered "repetitive".
    
    Returns:
        candidate_spans (list[tuple]): Spans that may be spurious duplicates.
    """
    candidate_spans = []
    for e, (start, end) in zip(char_id_entr, spans):
        if e < threshold:
            candidate_spans.append((start, end))
    return candidate_spans

def recombine(spans, candidate_spans, text):
    """
    Removes characters in all candidate_spans from the original text.
    
    Args:
        spans (list[tuple]): Not strictly necessary if we already have candidate_spans,
                             but sometimes you might use 'spans' to figure out something else.
        candidate_spans (list[tuple]): The (start, end) index ranges to remove.
        text (str): The original text string.

    Returns:
        new_text (str): Reconstructed text with spurious spans removed.
    """
    to_remove = set()
    for (start, end) in candidate_spans:
        # Mark all indices in the range [start, end) for removal
        to_remove.update(range(start, end))

    # Build new text by skipping removed indices
    new_chars = []
    for i, ch in enumerate(text):
        if i not in to_remove:
            new_chars.append(ch)
    
    return "".join(new_chars)

class Cleaner():
    def __init__(self,
                input_format='csv',
                output_tabular=False,
                sep=';',
                sectionize=False,
                clean_schema='mimic',
                config_loc='config/settings.yaml',
                input_loc=None,
                output_loc="../assets/corpus_cleaned.dat",
                terms_required=None):
        '''
        input_format : str, input format (tsv/csv/xml/txt)
        output_tabular : boolean, output as table
        sep : str, separator
        sectionize : boolean, process only text sections

        Notes
         sectionize not built, for references see # https://github.com/medspacy/medspacy/blob/master/medspacy/section_detection/sectionizer.py
         and https://allenai.github.io/scispacy/
        '''
        self.output_tabular = output_tabular
        self.input_format = 'csv'
        self.tabular_separator = sep
        self.sectionize = sectionize
        self.clean_schema = clean_schema
        self.input_loc = input_loc
        self.output_loc = output_loc
        self.terms_required = terms_required
        self.accepted_files = ['text/plain', 'text/csv', 'text/tab-separated-values', 'text/jsonl']

        assert isinstance(input_loc, str), f'input_loc should be a non-empty string'
        assert isinstance(config_loc, str), f'config_loc should be a non-empty string'
        assert isinstance(output_loc, str), f'output_loc should be a non-empty string'
        assert (terms_required is None) | isinstance(terms_required, list), f'terms_required should be None, or a list of strings'
        if terms_required is not None:
            if len(terms_required)==0:
                self.terms_required = None

        self.params = benedict(config_loc, format='yaml')
        self.clean_params = self.params['cleaning']
        self.re_delimited = re.compile(r''+self.clean_params['sentence_delim'])
        self.re_replacement = [(re.compile(r''+v[0]), v[1]) for v in self.clean_params['replace_characters']]
        self.sentence = ""


    def _spurious_repetitions(self, txt):
        '''
            Takes in text and removes spurious repetitions
            txt: text to process
        '''
        # using regex is too expensive (search space very large)
        # better to look at character entropy first.

        char_id_seq = get_char_seq(txt)
        char_id_entr, spans = get_char_entr(char_id_seq, window=5, stride=1)
        get_candidate_spans = get_cand_dupl(char_id_entr, spans)
        txt = recombine(spans, get_candidate_spans, txt)

        return txt

    def _clean(self, txt):
        for r in encoding_fixes:
            txt = txt.replace(r[0], r[1])
        for r in self.re_replacement:
            txt = r[0].sub(r[1], txt)
        return txt

    def _writer(self):
        return open(self.output_loc,
                    self.params['out']['write_mode'],
                    encoding=self.params['out']['encoding'])

    def _reader(self):
        assert(os.path.isfile(self.input_loc)), "Input file-location does not seem to refer to an actual file"
        assert(mimetypes.guess_type(self.input_loc)[0] in self.accepted_files), f"The file is present but does not seem to be the correct type:"+mimetypes.guess_type(self.input_loc)

        with open(self.input_loc, 'r', encoding=self.params['out']['encoding']) as reader:
            for line in reader.readlines():
                if self.terms_required is not None:
                    if not any([term in line for term in self.terms_required]):
                        pass
                else:
                    yield line

    def _reader_buffered(self):
        assert(os.path.isfile(self.input_loc)), "Input file-location does not seem to refer to an actual file"
        assert(mimetypes.guess_type(self.input_loc)[0] in self.accepted_files), f"The file is present but does not seem to be the correct type:"+mimetypes.guess_type(self.input_loc)

        with open(self.input_loc, 'r', encoding=self.params['out']['encoding']) as reader:
            f_id = io.FileIO(reader.fileno(), mode='r')
            f_buf = io.BufferedReader(f_id)
            while True:
                line = f_buf.readline().decode(self.params['out']['encoding'])
                if not line:
                    break
                if self.terms_required is not None:
                    if not any([term in line for term in self.terms_required]):
                        pass
                    else:
                        yield line
                else:
                    yield line

    def _sentencer(self, txt):
        '''
            collect whole sentences (sequence of tokens bounded by separators)
        '''
        self.sentence += txt
        if (len(self.re_delimited.split(self.sentence))>=2) | \
             (len(self.sentence)>self.clean_params['max_sentence_length']):
            return True

    def clean(self):
        assert(self.input_loc is not None), "Input file-location is not set, please set it first"

        reader = self._reader_buffered()
        writer = self._writer()

        for l in tqdm(reader):
            lp = self._clean(l)
            if len(lp)<self.clean_params['min_sentence_character_length']:
                continue
            if self._sentencer(lp):
                writer.write(self.sentence)
                self.sentence=""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing input for the cleaning routine')
    parser.add_argument('--in', dest='file_location', help='Absolute input-file location', type=str)
    parser.add_argument('--out', dest='output_location', help='Absolute output-file location', type=str)
    parser.add_argument('--config', dest='config_location', help='Absolute config-file location',
                        type=str, default='config/settings.yaml')
    parser.add_argument('--schema', dest='clean_schema', help='Cleaning settings', type=str)

    args = parser.parse_args()

    file_location = args.file_location
    output_location = args.output_location

    TextCleaner = Cleaner(input_loc=file_location, output_loc=output_location)
    TextCleaner.clean()
