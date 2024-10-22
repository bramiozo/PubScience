'''
This module contains classes to translate annotated and non-annotated corpora from
a source language to a target language.
'''

import os
from anthropic.resources import Messages
from dotenv import load_dotenv
import benedict

import google.generativeai as google_gen
from anthropic import Client as anthropic_client
from openai import Client as openai_client
from openai import NotFoundError as openai_NotFoundError
from openai import RateLimitError as openai_RateLimitError
from groq import Groq

from typing import Optional, Dict, List, Any
from pydantic import BaseModel

import argparse

load_dotenv(".env")

"""
This module contains classes to translate annotated and non-annotated corpora.

Output: {'translated_text': bla,
         'proba_char_range': [(0, 30, 0.942), (30, 35, 0.72)...
         'source_lang': bla,
         'target_lang': bla
         }
"""

#TODO: add support for bulk translations, using async methods.

class llm_input(BaseModel):
    source_language: str
    target_language: str
    text_to_translate: str

    def __str__(self) -> str:
        return "{" + f"'source_language': '{self.source_language}', 'target_language': '{self.target_language}', 'text_to_translate': '{self.text_to_translate}'" + "}"
    def __repr__(self) -> str:
        return self.__str__()

def _get_available_google_models(google_gen) -> List[str]:
    available_models = []
    for m in google_gen.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    return available_models

class TranslationLLM:
    def __init__(self,
        model: str,
        provider: str,
        source_lang: str,
        target_lang: str,
        system_prompt: str="",
        max_tokens: int=1024):

        self.model = model
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_tokens = max_tokens

        # parse yaml
        if system_prompt!="":
            self.system_prompt = system_prompt
        else:
            try:
                settings_loc = os.getenv('SETTINGS_YAML')
                llm_settings = benedict.benedict.from_yaml(settings_loc)
                self.system_prompt = llm_settings['translation']['method']['llm']['system_prompt']
            except Exception as e:
                raise FutureWarning(f"Could not parse system_prompt from yaml: {e}.\nContinuing with None")
                self.system_prompt = None

        if provider == 'openai':
            self.client = openai_client(api_key=os.getenv('OPENAI_LLM_API_KEY'))
        elif provider == 'anthropic':
            self.client = anthropic_client(api_key=os.getenv('ANTHROPIC_LLM_API_KEY'))
        elif provider == 'google':
            google_gen.configure(api_key=os.getenv('GOOGLE_LLM_API_KEY'))
            gGenConfig = google_gen.GenerationConfig(temperature=0.0, max_output_tokens=max_tokens)

            AvailableModels = _get_available_google_models(google_gen)

            if f"models/{model}" not in AvailableModels:
                raise ValueError(f"Model {model} not available. Available models are: {AvailableModels}")

            self.client = google_gen.GenerativeModel(model_name=model, safety_settings=None, system_instruction=self.system_prompt, generation_config=gGenConfig)
        elif provider == 'groq':
            self.client = Groq(api_key=os.getenv('GROQ_LLM_API_KEY'))


    def translate(self, text: str) -> Dict[str, Any]:
        InputText = llm_input(source_language=self.source_lang,
                              target_language=self.target_lang,
                              text_to_translate=text)

        if self.provider == 'openai':
            return self._translate_openai(InputText)
        elif self.provider == 'anthropic':
            return self._translate_anthropic(InputText)
        elif self.provider == 'google':
            return self._translate_google(InputText)
        elif self.provider == 'groq':
            return self._translate_groq(InputText)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _translate_openai(self, InputText: llm_input) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                temperature=0.0,
                max_tokens=self.max_tokens,
                model=self.model,
                messages=[{
                        'role': 'system',
                        'content': f"{self.system_prompt}"
                    },
                    {
                        'role': 'user',
                        'content': str(InputText)
                    }
                ]
            )
        except openai_NotFoundError as e:
            raise ValueError(f"Model {self.model} not found. {e}. Allowable models are: {self.client.models.list()}")
        except openai_RateLimitError as e:
            raise ValueError(f"Rate limit reached. {e}")

        return {'translated_text': response.choices[0].message.content.strip()}

    def _translate_anthropic(self, InputText: llm_input) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            temperature=0.0,
            system= f"{self.system_prompt}",
            messages=[{
                "role": "user",
                "content": str(InputText)
            }
            ],
            max_tokens=self.max_tokens
        )
        return {'translated_text': response.content[0].text.strip()}

    def _translate_google(self, InputText: llm_input) -> Dict[str, Any]:
        response = self.client.generate_content(
            str(InputText)
        )
        return {'translated_text': response.text.strip()}

    def _translate_groq(self, InputText: llm_input) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": f"{self.system_prompt}"
                },
                {
                    "role": "user",
                    "content": str(InputText)
                }
            ],
            model = self.model
        )
        return {'translated_text': response.choices[0].message.content.strip()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate annotated and non-annotated corpora using LLMs.")
    parser.add_argument('--model', type=str, required=True, help='The model to use for translation.')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'anthropic', 'google', 'groq'], help='The engine to use for translation.')
    parser.add_argument('--source_lang', type=str, required=True, help='The source language of the text.')
    parser.add_argument('--target_lang', type=str, required=True, help='The target language for the translation.')
    parser.add_argument('--system_prompt', type=str, default="", help='Optional system prompt for the translation model.')
    args = parser.parse_args()

    translator = TranslationLLM(**vars(args))

    text = """Recursion is the process a procedure goes through when one of the steps
    of the procedure involves invoking the procedure itself.
    A procedure that goes through recursion is said to be 'recursive'.
    To understand recursion, one must recognize the distinction between
    a procedure and the running of a procedure."""

    print(translator.translate(text))
