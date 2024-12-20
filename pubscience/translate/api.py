from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Tuple
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate_legacy
from google.cloud import translate_v3 as translate
from google.oauth2 import service_account
import deepl
import re
import os
from functools import lru_cache

# DeepL: https://www.deepl.com/en/pro-api
# Google: https://cloud.google.com/translate/pricing
#
load_dotenv('.env')

GOOGLE_AUTH_FILE = os.getenv('GOOGLE_AUTH_FILE')
GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
DEEPL_TRANSLATE_API_KEY = os.getenv('DEEPL_TRANSLATE_API_KEY')

def dict_to_tuple(d: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    """
    Convert a dictionary to a sorted tuple of key-value pairs.
    Sorting ensures that the tuple is consistent regardless of insertion order.
    """
    return tuple(sorted(d.items()))

def tuple_to_dict(t: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    """
    Convert a tuple of key-value pairs back to a dictionary.
    """
    return dict(t)

# Define the TranslationProvider interface
class TranslationProvider(ABC):
    @abstractmethod
    def translate(self, text: str, glossary: Dict[str, str], source_language: str, target_language: str) -> str:
        pass

class DeepLProvider(TranslationProvider):
    def __init__(self, api_key: str):
        self.translator = deepl.Translator(api_key)

    @lru_cache(maxsize=128_000)
    def translate(self,
                  text: str,
                  glossary_tuple: Tuple[Tuple[str, str], ...],
                  source_language: str,
                  target_language: str) -> str:

        # Create a regular expression pattern for glossary terms
        glossary = tuple_to_dict(glossary_tuple)
        pattern = '|'.join(map(re.escape, glossary.keys()))

        # Split the text into translatable and non-translatable parts
        parts = re.split(f'({pattern})', text)

        translated_parts = []
        for part in parts:
            if part in glossary:
                translated_parts.append(glossary[part])
            elif part.strip():  # Only translate non-empty parts
                result = self.translator.translate_text(
                    part,
                    source_lang=source_language,
                    target_lang=target_language
                )
                translated_parts.append(result.text)
            else:
                translated_parts.append(part)  # Keep empty parts (spaces, newlines) as is

        return ''.join(translated_parts)

class GoogleTranslateProvider(TranslationProvider):
    def __init__(self,
        legacy: bool = False,
        credentials_path: str = None,
        project_id: str = None):
        self.legacy = legacy
        self.project_id = project_id
        self.translator = self.initialize_translate_client(credentials_path)

    def initialize_translate_client(self, credentials_path=None):
        assert isinstance(credentials_path, str), "Credentials path is required."
        credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if self.legacy:
            client = translate_legacy.Client(credentials=credentials)
        else:
            client = translate.TranslationServiceClient(credentials=credentials)
        return client

    @lru_cache(maxsize=128_000)
    def translate(self,
                  text: str,
                  glossary_tuple: Tuple[Tuple[str, str], ...],
                  source_language: str,
                  target_language: str) -> str:

        glossary = tuple_to_dict(glossary_tuple)
        if self.legacy is False:
            parent = f"projects/{self.project_id}/locations/global"
        else:
            parent = ""
        # Create a regular expression pattern for glossary terms
        pattern = '|'.join(map(re.escape, glossary.keys()))

        # Split the text into translatable and non-translatable parts
        parts = re.split(f'({pattern})', text)

        translated_parts = []
        for part in parts:
            if part in glossary:
                translated_parts.append(glossary[part])
            elif part.strip():  # Only translate non-empty parts
                if self.legacy:
                    result = self.translator.translate(
                        part,
                        source_language=source_language,
                        target_language=target_language
                    )
                    translated_parts.append(result['translatedText'])
                else:
                    result = self.translator.translate_text(
                        request={
                            "parent": parent,
                            "contents": [part],
                            "mime_type": "text/plain",
                            "source_language_code": source_language,
                            "target_language_code" :target_language
                        }
                    )
                    translated_parts.append(result.translations[0].translated_text)
            else:
                translated_parts.append(part)  # Keep empty parts (spaces, newlines) as is

        return ''.join(translated_parts)

class TranslationAPI:
    def __init__(self,
                 provider: str,
                 glossary: Dict[str, str],
                 source_language: str,
                 target_language: str,
                 legacy_google: bool = False):
        if not (GOOGLE_AUTH_FILE and GOOGLE_PROJECT_ID) or not DEEPL_TRANSLATE_API_KEY:
            raise ValueError("API keys for Google Translate and DeepL are required.")

        self.glossary = glossary
        self.source_language = source_language
        self.target_language = target_language
        if provider.lower() == 'deepl':
            self.provider = DeepLProvider(DEEPL_TRANSLATE_API_KEY)
        elif provider.lower() == 'google':
            self.provider = GoogleTranslateProvider(legacy=legacy_google,
                                                    credentials_path=GOOGLE_AUTH_FILE,
                                                    project_id=GOOGLE_PROJECT_ID)
        else:
            raise ValueError("Unsupported provider. Use 'deepl' or 'google'.")

    def translate(self, text_file_stream: Iterator[str]) -> Iterator[str]:
        glossary_tuple = dict_to_tuple(self.glossary)
        for line in text_file_stream:
            yield self.provider.translate(line, glossary_tuple,
                self.source_language, self.target_language)

############################################################################################################
def google_cost_estimator(number_of_characters: int) -> float:
    if number_of_characters <= 250_000_000:
        return (number_of_characters / 1_000_000) * 80
    elif number_of_characters <= 2_500_000_000:
        return (250 * 80) + ((number_of_characters - 250_000_000) / 1_000_000) * 60
    elif number_of_characters <= 4_000_000_000:
        return (250 * 80) + (2_250 * 60) + ((number_of_characters - 2_500_000_000) / 1_000_000) * 40
    else:
        return (250 * 80) + (2_250 * 60) + (1_500 * 40) + ((number_of_characters - 4_000_000_000) / 1_000_000) * 30

def deepl_cost_estimator(number_of_characters: int) -> float:
    return 5 + (number_of_characters / 1_000_000) * 20

def text_file_stream():
    # This is a mock function to simulate a text file stream
    yield from ["Everything is lie",
                "Hello, world!",
                "How are you?",
                "Goodbye!",
                "My life is wonderful"]

if __name__ == '__main__':
    # Instantiate the TranslationAPI
    translator = TranslationAPI(
        provider='deepl',
        glossary={'Hello': 'Hola', 'Goodbye': 'Adiós'},
        source_language='en',
        target_language='es'
    )
    # Use the translator
    translated_text_iterator = translator.translate(text_file_stream())

    # Print the translated text
    print("DeepL Translation:")
    for translated_line in translated_text_iterator:
        print(translated_line)

    # Instantiate the TranslationAPI
    translator = TranslationAPI(
        provider='google',
        glossary={'Hello': 'Hola', 'Goodbye': 'Adiós'},
        source_language='en',
        target_language='es',
        legacy_google=False
    )
    # Use the translator
    translated_text_iterator = translator.translate(text_file_stream())

    # Print the translated text
    print("Google Translation v3:")
    for translated_line in translated_text_iterator:
        print(translated_line)

    # Instantiate the TranslationAPI
    translator = TranslationAPI(
        provider='google',
        glossary={'Hello': 'Hola', 'Goodbye': 'Adiós'},
        source_language='en',
        target_language='es',
        legacy_google=True
    )
    # Use the translator
    translated_text_iterator = translator.translate(text_file_stream())

    # Print the translated text
    print("Google Translation v2:")
    for translated_line in translated_text_iterator:
        print(translated_line)
