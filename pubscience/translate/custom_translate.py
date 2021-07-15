import six
from google.cloud import translate
import argparse
import benedict
import re
import os
from collections import deque

class Translate():
    def __init__(self, 
                input_loc = None,
                output_loc = None,
                glos_loc = None,
                source_lang='en', 
                dest_lang='nl', 
                method='public', 
                batch_size=50, 
                write_mode = 'w',
                public_key_file=None,
                config_file='./config/settings.yaml'):
        '''
            Batchsize in number of lines, batching is used to reduce the number of API calls
            glos_loc, location in google cloud, projects/locations/glossaries/*.
        '''
        assert input_loc is not None, "Input location should be provided"
        assert output_loc is not None, "Output location should be provided"

        self.input_loc = input_loc
        self.output_loc = output_loc
        self.glos_loc = glos_loc

        if os.path.exists(config_file):
            if os.path.isfile(config_file):
                self.source_lang = source_lang
                self.dest_lang = dest_lang
                self.method = method
                self.batch_size = batch_size
                self.write_mode = write_mode
                self.config_file = config_file
        else:
            params = benedict(config_file, format='yaml')
            for k,v in params['translation'].items():
                setattr(self, k, v)

        if self.method == 'public':
            assert isinstance(public_key_file, str), "Public key file should be the location of a .json/.p12/.pem file"
            if os.path.exists(public_key_file):
                if os.path.isfile(public_key_file):
                    self.translate_client = translate.TranslationServiceClient(keyFilename=public_key_file)
                    if self.glos_loc is not None:
                        glossary_config = self.translate_client.TranslateTextGlossaryConfig(glossary=self.glos_loc)


    def translate_text_google(txt):
        """Translates text into the target language.

        Target must be an ISO 639-1 language code.
        See https://g.co/cloud/translate/v2/translate-reference#supported_languages

        For more information on setting up the google translation API:
            * https://cloud.google.com/translate/docs/setup
            * https://cloud.google.com/translate/docs/advanced/glossary
            * https://cloud.google.com/translate/docs/samples/translate-v3-create-glossary
        """
        if isinstance(txt, six.binary_type):
            text = txt.decode("utf-8")
        result = self.translate_client.translate(txt, target_language_code='nl', source_language_code='en')
        return result['translatedText']

    def translate_text_nmt(txt):
        # https://github.com/UKPLab/EasyNMT
        return True
               
    def _reader(self):
        # TODO: check if self.input_loc is a file
        batch = deque(['' for i range(self.batch_size)], maxlen=self.batch_size)
        with open(self.input_loc, 'r') as reader:
            for cdx, line in enumerate(reader.readlines()):
                batch.appendleft(line.encode('utf-8'))
                if (cdx+1)%batch_size==0:
                    yield "\n".join(batch)
    
    def _make_glossary():
        '''Create glossary file in cloud that is accessible for project, based on .csv input.
        '''




    def translate(self):
        with open(self.output_loc, self.write_mode) as w:
            for b in self._reader():
                w.write(self.translate_text_google(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing input for the cleaning routine')
    parser.add_argument('--in', dest='file_loc', help='Absolute input-file location', type=str)
    parser.add_argument('--out', dest='out_loc', help='Absolute output-file location', type=str)
    parser.add_argument('--glossary', dest='glos_loc', help='Glossary to enforce translation pairs', type=str)
    parser.add_argument('--config', dest='config_loc', help='Absolute config-file location', 
                        type=str, default='config/settings.yaml')
    parser.add_argument('--pubkey', dest='pub_key_file', help='Absolute credentials-file location', 
                        type=str, default=None)

    args = parser.parse_args()

    translator = Translate(input_loc=args.file_loc, 
                           output_loc=args.out_loc, 
                           config_file=args.config_loc,
                           public_key_file=args.public_key_file
                           )