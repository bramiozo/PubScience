'''
This module contains classes to translate annotated and non-annotated corpora from
a source language to a target language.
'''

import os
from dotenv import load_dotenv
import benedict
import asyncio

import google.genai as google_gen
from google.genai.types import (
    HarmCategory,
    HarmBlockThreshold,
    GenerateContentConfig,
    SafetySetting
)

from anthropic import Anthropic
from anthropic import Client as anthropic_client
from anthropic import AsyncAnthropic
from openai import Client as openai_client
from openai import AsyncOpenAI
from openai import NotFoundError as openai_NotFoundError
from openai import RateLimitError as openai_RateLimitError
from openai import BadRequestError as openai_BadRequestError
from groq import Groq, AsyncGroq

from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel
import torch
import httpx
import pysbd

import argparse
import warnings

print(f"Current directory: {os.path.join(os.path.dirname(__file__))}")

"""
This module contains classes to translate annotated and non-annotated corpora.

Output: {'translated_text': bla,
         'proba_char_range': [(0, 30, 0.942), (30, 35, 0.72)...
         'source_lang': bla,
         'target_lang': bla
         }
"""
# mav23/EuroLLM-1.7B-GGUF
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

#TODO: [x] add support for bulk translations, using async methods.
#TODO: [ ] add support for chunking large texts into smaller pieces for translation.

class llm_input(BaseModel):
    source_language: str
    target_language: str
    text_to_translate: str

    def __str__(self) -> str:
        return "{" + f"'source_language': '{self.source_language}', 'target_language': '{self.target_language}', 'text_to_translate': '{self.text_to_translate}'" + "}"
    def __repr__(self) -> str:
        return self.__str__()

class llm_inputs(BaseModel):
    source_language: str
    target_language: str
    text_to_translate: str
    max_words_per_chunk: int = 1024

    def __iter__(self):
        # calling list(llm_inputs(**vars)) will return a list of dictionaries
        for txt in self.get_text_chunks():
            yield {
                'source_language': self.source_language,
                'target_language': self.target_language,
                'text_to_translate': txt
            }

    def __repr__(self) -> str:
        return f"llm_inputs(source_language={self.source_language!r}, " \
               f"target_language={self.target_language!r}, " \
               f"text_to_translate={self.text_to_translate!r}, " \
               f"max_words_per_chunk={self.max_words_per_chunk!r})"

    def __len__(self):
        return len(self.get_text_chunks())

    def get_text_chunks(self) -> List[str]:
        seg = pysbd.Segmenter(language="en", clean=False)
        sentences = seg.segment(self.text_to_translate)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            word_count = len(sentence_words)

            if current_word_count + word_count <= self.max_words_per_chunk:
                current_chunk.append(sentence)
                current_word_count += word_count
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

def _get_available_google_models(google_gen) -> List[str]:
    available_models = []
    for m in google_gen.models.list():
        available_models.append(m.name)
    return available_models

class TranslationLLM:
    def __init__(self,
        model: str,
        provider: Literal['openai', 'anthropic', 'google', 'groq', 'deepseek', 'local'],
        source_lang: str,
        target_lang: str,
        env_loc: str,
        system_prompt: str="",
        max_tokens: int=5000,
        max_tokens_truncate: bool=False,
        max_chunk_size: int=1024,
        max_processes: int=8,
        temperature: float=0.0,
        ):

        load_dotenv(env_loc)

        SETTINGS_YAML = os.getenv('SETTINGS_YAML')

        self.model = model
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_tokens = max_tokens
        self.max_chunk_size = max_chunk_size
        self.max_tokens_truncate = max_tokens_truncate
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_processes = max_processes

        self.gen_kwargs = {
            'early_stopping': False,
            'top_p': 0.95,
            'top_k': 50,
            'temperature': temperature,
            'do_sample': True,
            'min_p': 0.05,
            'repetition_penalty': 1.5,
            'encoder_repetition_penalty': 1.5,
            'length_penalty': 0.25,
            'no_repeat_ngram_size': 0,
            'num_return_sequences': 1,
        }
        google_gen_kwargs = {
            'top_p': 0.95,
            'top_k': 50,
            'temperature': temperature,
            #'frequency_penalty': 1.5,
            #'presence_penalty': 0.25,
            'candidate_count': 1
        }

        # parse yaml
        if system_prompt!="":
            self.system_prompt = system_prompt
        else:
            try:
                print(f"Loading settings from: {SETTINGS_YAML}")
                llm_settings = benedict.benedict.from_yaml(SETTINGS_YAML)
                self.system_prompt = llm_settings['translation']['method']['llm']['system_prompt']
            except Exception as e:
                self.system_prompt = None
                raise FutureWarning(f"Could not parse system_prompt from yaml: {e}.\nContinuing with None")


        if provider == 'openai':
            self.client = openai_client(api_key=os.getenv('OPENAI_LLM_API_KEY'))
            self.aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_LLM_API_KEY'))

            # Check if model is available
            if model not in [m.id for m in self.client.models.list()]:
                raise ValueError(f"Model {model} not available. Allowable models are: {self.client.models.list()}")

        elif provider == 'anthropic':
            self.client = anthropic_client(api_key=os.getenv('ANTHROPIC_LLM_API_KEY'))
            self.aclient = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_LLM_API_KEY'))

            anthropic_models = self._get_anthropic_models()
            # Check if model is available
            if model not in anthropic_models:
                raise ValueError(f"Model {model} not available. Allowable models are: {anthropic_models}")

        elif provider == 'google':
            safety_settings=[
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE)
            ]
            self.GoogleConfig = GenerateContentConfig(
                system_instruction = self.system_prompt,
                max_output_tokens = max_tokens,
                safety_settings = safety_settings,
                **google_gen_kwargs,
            )

            self.client = google_gen.Client(api_key=os.getenv('GOOGLE_LLM_API_KEY'))
            try:
                model_info = self.client.models.get(model=model)
                print(f"Model info: {model_info}")
            except Exception as e:
                print(f"Problem with model{model}: {e}")
                AvailableModels = _get_available_google_models(self.client)
                raise ValueError(f"Model {model} not available. Available models are: {AvailableModels}")


        elif provider == 'groq':
            self.client = Groq(api_key=os.getenv('GROQ_LLM_API_KEY'))
            self.aclient = AsyncGroq(api_key=os.getenv('GROQ_LLM_API_KEY'))

            groq_models = self._get_groq_models()
            # Check if model is available
            if model not in groq_models:
                raise ValueError(f"Model {model} not available. Allowable models are: {groq_models}")
        elif provider == 'deepseek':
            self.client = openai_client(api_key=os.getenv('DEEPSEEK_LLM_API_KEY'), base_url="https://api.deepseek.com/v1")
            self.aclient = AsyncOpenAI(api_key=os.getenv('DEEPSEEK_LLM_API_KEY'), base_url="https://api.deepseek.com/v1")

            if model != 'deepseek-chat':
                raise ValueError(f"Model {model} not available. Allowable models are: ['deepseek-chat']")
        elif provider == 'local':
            # EuroLLM-9B-Instruct
            # unsloth/Mixtral-8x7B-v0.1-bnb-4bit
            from unsloth import FastLanguageModel
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

    def _get_anthropic_models(self) -> List[str]:
        key = os.getenv('ANTHROPIC_LLM_API_KEY')
        response = httpx.get('https://api.anthropic.com/v1/models', headers={'x-api-key': key, "anthropic-version": "2025-03-01", 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            models = [model['id'] for model in data['data']]
            return models
        else:
            raise Exception(f"Failed to get models from Anthropic API: {response.status_code} {response.text}")

    def _get_groq_models(self) -> List[str]:
        key = os.getenv('GROQ_LLM_API_KEY')
        response = httpx.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {key}', 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
        if response.status_code == 200:
            data = response.json()
            models = [model['id'] for model in data['data']]
            return models
        else:
            raise Exception(f"Failed to get models from Groq API: {response.status_code} {response.text}")

    def translate(self, text: str) -> Dict[str, Any]:
        InputText = llm_input(source_language=self.source_lang,
                              target_language=self.target_lang,
                              text_to_translate=text)
        # check if text is too long
        if len(text.split()) > self.max_tokens:
            raise ValueError(f"Text is too long. Max tokens is {self.max_tokens}. Text has {len(text.split())} tokens.")

        if self.provider in ['openai', 'deepseek']:
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

    def translate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        assert (self.provider.lower() != "local"), "Batch translation not supported for the local LLM."
        assert (len(texts)<= self.max_processes), f"Batch size {len(texts)} exceeds maximum number of processes: {self.max_processes}"

        if self.max_tokens_truncate:
            # truncate all texts to max self.max_tokens
            texts = [" ".join(text.split()[:self.max_tokens]) for text in texts]
        else:
            assert (all([len(text.split()) <= self.max_tokens for text in texts])), f"One or more texts are too long. Max tokens is {self.max_tokens}."

        warnings.warn("""\n\nUsing `translate_batch()` is 50% more expensive than using `llm_batch()`. \n The latter is also faster for larger amounts. \n\n Please consider using `llm_batch()` instead.""", stacklevel=2)

        # turn into a list of llm_input objects
        #
        InputTexts = [
                        llm_input(
                            source_language=self.source_lang,
                            target_language=self.target_lang,
                            text_to_translate=text)
                        for text in texts
        ]

        if self.provider in ['openai', 'deepseek']:
            coroutines = [self._translate_openai_async(input_text) for input_text in InputTexts]
        elif self.provider == 'anthropic':
            coroutines = [self._translate_anthropic_async(input_text) for input_text in InputTexts]
        elif self.provider == 'google':
            coroutines = [self._translate_google_async(input_text) for input_text in InputTexts]
        elif self.provider == 'groq':
            coroutines = [self._translate_groq_async(input_text) for input_text in InputTexts]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*coroutines))
        return results

    def _translate_openai(self, InputText: llm_input) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                temperature=self.temperature,
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
        except openai_BadRequestError as e:
            print(f"Error: {e}")
            return {'translated_text': 'NOT TRANSLATED -- see error message', 'error_message': f'BadRequest: {str(e)}'}

        return {'translated_text': response.choices[0].message.content.strip()}

    async def _translate_openai_async(self, InputText: str) -> Dict[str, Any]:
        try:
            response = await self.aclient.chat.completions.create(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model=self.model,
                messages=[
                    {
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
            raise ValueError(
                f"Model {self.model} not found. {e}. "
                f"Allowable models are: {self.client.models.list()}"
            )
        except openai_RateLimitError as e:
            raise ValueError(f"Rate limit reached. {e}")
        except openai_BadRequestError as e:
            print(f"Error: {e}")
            return {'translated_text': 'NOT TRANSLATED -- see error message', 'error_message': f'BadRequest: {str(e)}'}

        return {'translated_text': response.choices[0].message.content.strip()}

    def _translate_anthropic(self, InputText: llm_input) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            temperature=self.temperature,
            system= f"{self.system_prompt}",
            messages=[{
                "role": "user",
                "content": str(InputText)
            }
            ],
            max_tokens=self.max_tokens
        )
        return {'translated_text': response.content[0].text.strip()}

    async def _translate_anthropic_async(self, InputText: llm_input) -> Dict[str, Any]:
        response = await self.aclient.messages.create(
            model=self.model,
            temperature=self.temperature,
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
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=str(InputText),
                config = self.GoogleConfig
            )
            return {'translated_text': response.text.strip(),
                'feedback': response.prompt_feedback}
        except Exception as e:
            print(f"Error: {e}")
            return {'translated_text': None, 'feedback': response.prompt_feedback}

    async def _translate_google_async(self, InputText: llm_input) -> Dict[str, Any]:
        try:
            # if this fails, try await self.client.aio.models.generate_content
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=str(InputText),
                config = self.GoogleConfig
            )
            return {'translated_text': response.text.strip(),
                'feedback': response.prompt_feedback}
        except Exception as e:
            print(f"Error: {e}")
            return {'translated_text': None, 'feedback': e}

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

    async def _translate_groq_async(self, InputText: llm_input) -> Dict[str, Any]:
        response = await self.aclient.chat.completions.create(
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
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'anthropic', 'google', 'groq', 'local', 'deepseek'], help='The engine to use for translation.')
    parser.add_argument('--source_lang', type=str, required=True, help='The source language of the text.')
    parser.add_argument('--target_lang', type=str, required=True, help='The target language for the translation.')
    parser.add_argument('--system_prompt', type=str, default="", help='Optional system prompt for the translation model.')
    parser.add_argument('--env_loc', type=str, default='.env', help='The location of the .env file.')
    args = parser.parse_args()

    translator = TranslationLLM(**vars(args))

    text = """Recursion is the process a procedure goes through when one of the steps
    of the procedure involves invoking the procedure itself.
    A procedure that goes through recursion is said to be 'recursive'.
    To understand recursion, one must recognize the distinction between
    a procedure and the running of a procedure."""

    print("Test TranslationLLM.translate()..")
    print(translator.translate(text))
    print("+"*50)
    print("Test TranslationLLM.translate_batch()..")

    texts = [text, text, text, text]

    print(translator.translate_batch(texts))
