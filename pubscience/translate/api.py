from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
import deepl
import re
import os

# DeepL: https://www.deepl.com/en/pro-api
# Google: https://cloud.google.com/translate/pricing
#
#
load_dotenv('.env')

GOOGLE_TRANSLATE_API_KEY = os.getenv('GOOGLE_TRANSLATE_API_KEY')
DEEPL_TRANSLATE_API_KEY = os.getenv('DEEPL_TRANSLATE_API_KEY')

# Define the TranslationProvider interface
class TranslationProvider(ABC):
    @abstractmethod
    def translate(self, text: str, glossary: Dict[str, str], source_language: str, target_language: str) -> str:
        pass

class DeepLProvider(TranslationProvider):
    def __init__(self, api_key: str):
        self.translator = deepl.Translator(api_key)

    def translate(self, text: str, glossary: Dict[str, str], source_language: str, target_language: str) -> str:
        # Create a regular expression pattern for glossary terms
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
    def __init__(self, api_key: str):
        self.translator = translate.Client(api_key=api_key)

    def translate(self, text: str, glossary: Dict[str, str], source_language: str, target_language: str) -> str:
        # Create a regular expression pattern for glossary terms
        pattern = '|'.join(map(re.escape, glossary.keys()))

        # Split the text into translatable and non-translatable parts
        parts = re.split(f'({pattern})', text)

        translated_parts = []
        for part in parts:
            if part in glossary:
                translated_parts.append(glossary[part])
            elif part.strip():  # Only translate non-empty parts
                result = self.translator.translate(
                    part,
                    source_language=source_language,
                    target_language=target_language
                )
                translated_parts.append(result['translatedText'])
            else:
                translated_parts.append(part)  # Keep empty parts (spaces, newlines) as is

        return ''.join(translated_parts)

class TranslationAPI:
    def __init__(self, provider: str, glossary: Dict[str, str], source_language: str, target_language: str):
        if not GOOGLE_TRANSLATE_API_KEY or not DEEPL_TRANSLATE_API_KEY:
            raise ValueError("API keys for Google Translate and DeepL are required.")

        self.glossary = glossary
        self.source_language = source_language
        self.target_language = target_language
        if provider.lower() == 'deepl':
            self.provider = DeepLProvider(DEEPL_TRANSLATE_API_KEY)
        elif provider.lower() == 'google':
            self.provider = GoogleTranslateProvider(GOOGLE_TRANSLATE_API_KEY)
        else:
            raise ValueError("Unsupported provider. Use 'deepl' or 'google'.")

    def translate(self, text_file_stream: Iterator[str]) -> Iterator[str]:
        for line in text_file_stream:
            yield self.provider.translate(line, self.glossary, self.source_language, self.target_language)

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
    for translated_line in translated_text_iterator:
        print(translated_line)
