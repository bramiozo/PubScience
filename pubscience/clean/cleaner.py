import argparse
import re
import sys
import argparse
from benedict import benedict
from beautifulsoup import Beautifulsoup

'''
Takes raw data in from:
raw PubMed/PMC, MIMICIII in tabular format, raw text or xml.

Outputs:
a. Cleaned, text-only corpus without tables/lists, per line of some maximum length.
b. ditto but per document in tabular format with identifiers/labels, if available
'''

class Cleaner():
    def __init__(self, 
                input_format='csv', 
                output_tabular=False, 
                sep=';', 
                sectionize=False, 
                clean_schema='mimic', 
                config_loc='config/settings.yaml',
                input_loc=None,
                output_loc="corpus_cleaned.dat"):
        '''
        input_format : str, input format (csv/xml/txt)
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

        assert isinstance(input_loc, str), f'input_loc should be a non-empty string'
        assert isinstance(config_loc, str), f'config_loc should be a non-empty string'
        assert isinstance(output_loc, str), f'output_loc should be a non-empty string'

        params = benedict(config_loc, format='yaml')
        self.clean_params = params['cleaning']
        self.re_delimited = re.compile(r''+self.clean_params['sentence_delim'])
        self.re_remove = re.compile(r''+self.clean_params['remove_characters'])
        self.re_replacement = [(re.compile(r''+v[0]), v[1]) for v in self.clean_params['replace_characters'].values()]
        self.sentence = ""

    def _clean(self, txt):
        txt = self.re_remove.sub('', txt)
        for r in self.re_replacement:
            txt = r[0].sub(r[1], txt)
        return txt

    def _writer(self):
        return open(self.output_loc, self.clean_params['write_mode'])           
            
    def _reader(self):
        # TODO: check if self.input_loc is a file
        with open(self.input_loc, 'r') as reader:
            for line in reader.readlines():
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
        reader = self._reader
        writer = self._writer()

        for l in reader():
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
    parser.add_argument('--schema', dest='output_location', help='Cleaning settings', type=str)

    args = parser.parse_args()

    file_location = args.file_location
    output_location = args.output_location

    TextCleaner = Cleaner(input_loc=file_location, output_loc=output_location)
    TextCleaner.clean()
