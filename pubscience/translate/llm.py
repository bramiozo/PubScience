'''
This module contains classes to translate annotated and non-annotated corpora from
a source language to a target language.
'''

import os
from re import M
from anthropic.resources import Messages
from dotenv import load_dotenv
import benedict

import google.generativeai as google_gen
from anthropic import Client as anthropic_client
from openai import Client as openai_client
from openai import NotFoundError as openai_NotFoundError
from openai import RateLimitError as openai_RateLimitError
from groq import Groq

from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel

from unsloth import FastLanguageModel
import torch

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

unsloth_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit"
]

prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

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
        provider: Literal['openai', 'anthropic', 'google', 'groq', 'local'],
        source_lang: str,
        target_lang: str,
        system_prompt: str="",
        max_tokens: int=1024):

        self.model = model
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gen_kwargs = {
            'early_stopping': False,
            'top_p': 0.95,
            'top_k': 50,
            'temperature': 0.77,
            'do_sample': True,
            'min_p': 0.05,
            'repetition_penalty': 1.5,
            'encoder_repetition_penalty': 1.5,
            'length_penalty': 0.25,
            'no_repeat_ngram_size': 0,
            'num_return_sequences': 1,
        }

        # parse yaml
        if system_prompt!="":
            self.system_prompt = system_prompt
        else:
            try:
                settings_loc = os.getenv('SETTINGS_YAML')
                llm_settings = benedict.benedict.from_yaml(settings_loc)
                self.system_prompt = llm_settings['translation']['method']['llm']['system_prompt']
            except Exception as e:
                self.system_prompt = None
                raise FutureWarning(f"Could not parse system_prompt from yaml: {e}.\nContinuing with None")


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
        elif provider == 'local':
            if model not in unsloth_models:
                raise ValueError(f"""Model {model} not available.
                    Available models are: {unsloth_models}.
                    For more models see: https://huggingface.co/unsloth""")

            self.client, self.tokenizer = FastLanguageModel.from_pretrained(model_name=model,
                max_seq_length=max_tokens, load_in_4bit=True
            )
            FastLanguageModel.for_inference(self.client) # Enable native 2x faster inference

        else:
            raise ValueError(f"Unsupported provider: {provider}")


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
        elif self.provider == 'local':
            return self._translate_local(InputText)
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

    def _translate_local(self, InputText: llm_input) -> Dict[str, Any]:
        _InputText = str(InputText)
        inputs = self.tokenizer([
            prompt_format.format(
                self.system_prompt,
                _InputText,
                ""
            )
        ],
            return_tensors="pt").to(self.device)

        response = self.client.generate(
            **inputs,
            **self.gen_kwargs,
            max_new_tokens = min(1.5*len(_InputText.split()), self.max_tokens),
            use_cache=True,
            pad_token_id = self.tokenizer.eos_token_id,
        )

        # TODO: add parser to extract only the response
        return {'translated_text': self.tokenizer.batch_decode(response)[0].strip()}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate annotated and non-annotated corpora using LLMs.")
    parser.add_argument('--model', type=str, required=True, help='The model to use for translation.')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'anthropic', 'google', 'groq', 'local'], help='The engine to use for translation.')
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
