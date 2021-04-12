import six
from google.cloud import translate_v2 as translate
import argparse

def translate_text(txt):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language='nl', source_language='en')

    return result['translatedText']

def _writer(self):
    return open(self.output_loc, self.clean_params['write_mode'])           
            
def _reader(self):
    # TODO: check if self.input_loc is a file
    with open(self.input_loc, 'r') as reader:
        for line in reader.readlines():
            yield line

if __name__ == '__main__':
    writer = open(args.outgoing, 'a')
    with open(args.incoming, 'r') as rf:
        for line in rf.readlines():
            
    writer.close()