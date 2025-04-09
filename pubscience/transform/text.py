"""
Text transformation functions.

Write functions to transform text with LLMs

A -> transform(A, InstructionForParaphrasing, n=1) -> transform(_, InstructionForTranslation, n=1)
"""

from typing import List
from pydantic import BaseModel

import os
from anthropic.resources import Messages
from dotenv import load_dotenv
import benedict

import google.genai as google_gen
from google.genai.types import (
    HarmCategory,
    HarmBlockThreshold,
    GenerateContentConfig,
    SafetySetting
)
from anthropic import Client as anthropic_client
from openai import Client as openai_client
from openai import NotFoundError as openai_NotFoundError
from openai import RateLimitError as openai_RateLimitError
from groq import Groq

from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel

import asyncio
from time import sleep

import json
from tqdm import tqdm
import argparse

load_dotenv(".env")

# TODO: add support for bulk translations, using async methods.

class llm_input(BaseModel):
    instruction: str
    text_to_transform: str

    def __str__(self) -> str:
        return "{" + f"'instruction': '{self.instruction}', 'text_to_transform': '{self.text_to_transform}'" + "}"
    def __repr__(self) -> str:
        return self.__str__()

def _get_available_google_models(google_gen) -> List[str]:
    available_models = []
    for m in google_gen.models.list():
        available_models.append(m.name)
    return available_models


class transform():
    def __init__(self,
                 system_prompt: str,
                 instruction_list: List[str],
                 provider: Literal['google', 'anthropic', 'openai', 'groq']=None,
                 model: str|None=None,
                 temperature: float=0.25,
                 batch_size: int=1,
                 max_tokens: int=5048):

        assert(
            provider in ['google', 'anthropic', 'openai', 'groq']
        ), f"Provider {provider} not supported. Supported providers are: ['google', 'anthropic', 'openai', 'groq']"

        self.system_prompt = system_prompt
        self.instruction_list = instruction_list
        self.max_tokens = max_tokens
        self.provider = provider

        settings_loc = os.getenv('SETTINGS_YAML')
        llm_settings = benedict.benedict.from_yaml(settings_loc)

        google_gen_kwargs = {
            'top_p': 0.95,
            'top_k': 50,
            'temperature': temperature,
            'frequency_penalty': 1.5,
            'presence_penalty': 0.25,
            'candidate_count': 1
        }


        # TODO: add support for n>1
        # if n>1:
        #     n = 1
        #     raise NotImplementedError("Support for n>1 not yet implemented. Continuing with n=1.")

        # parse yaml
        if isinstance(system_prompt,str):
            self.system_prompt = system_prompt
        else:
            try:
                self.system_prompt = llm_settings['transformation']['method']['llm']['system_prompt']
            except Exception as e:
                self.system_prompt = None
                raise FutureWarning(f"Could not parse system_prompt from yaml: {e}.\nContinuing with None")

        if isinstance(model, str):
            self.model = model
        else:
            try:
                self.model = llm_settings['transformation']['method']['llm']['model']
            except Exception as e:
                self.model = None
                raise ValueError(f"Could not parse model from yaml: {e}. Please identify an available model from the provider.")

        if isinstance(instruction_list, list) and all(isinstance(i, str) for i in instruction_list):
            self.instruction_list = instruction_list
        else:
            self.instruction_list = llm_settings['transformation']['instructions']


        if provider == 'openai':
            self.client = openai_client(api_key=os.getenv('OPENAI_LLM_API_KEY'))
        elif provider == 'anthropic':
            self.client = anthropic_client(api_key=os.getenv('ANTHROPIC_LLM_API_KEY'))
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

            AvailableModels = _get_available_google_models(self.client)

            if f"models/{model}" not in AvailableModels:
                raise ValueError(f"Model {model} not available. Available models are: {AvailableModels}")
        elif provider == 'groq':
            self.client = Groq(api_key=os.getenv('GROQ_LLM_API_KEY'))

        #self.write_per_instruction = llm_settings.get('transformation').get('out_per_instruction', False)
        self.intermediate_outputs = []

    def __call__(self, text: str):
        self.intermediate_outputs = []
        for instruction in self.instruction_list:
            text = self._transform(text, instruction)
            self.intermediate_outputs.append(text)
        return text

    def _transform(self, text: str, instruction: str):
        InputText = llm_input(instruction=instruction,
                            text_to_transform=text)

        if self.provider == 'openai':
            return self.__transform_openai(InputText)
        elif self.provider == 'anthropic':
            return self.__transform_anthropic(InputText)
        elif self.provider == 'google':
            return self.__transform_google(InputText)
        elif self.provider == 'groq':
            return self.__transform_groq(InputText)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def __transform_google(self, InputText: llm_input) -> Dict[str, Any]:
        # TODO: if self.n>1, this will return a list of responses...
        # TODO: the number of total outcome then becomes n^numInstructions, perhaps not what we want? :D
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=str(InputText),
                config = self.GoogleConfig
            )
            if response.parsed is None:
                return response.text.strip()
            else:
                return f"No response from Google LLM.<ERROR>{str(response.prompt_feedback)}</ERROR>"
        except:
            raise ValueError(f"Could not transform text with Google LLM for {str(InputText)}")

    def __transform_anthropic(self, InputText: llm_input) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            temperature=0.1,
            system= f"{self.system_prompt}",
            messages=[{
                "role": "user",
                "content": str(InputText)
            }
            ],
            max_tokens=self.max_tokens
        )
        return response.content[0].text.strip()

    def __transform_groq(self, InputText: llm_input) -> Dict[str, Any]:
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
        return response.choices[0].message.content.strip()

    def __transform_openai(self, InputText: llm_input) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                temperature=0.1,
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

        return response.choices[0].message.content.strip()

def parse_folder_with_txt(arguments: argparse.Namespace):
    list_of_files = os.listdir(arguments.folder_path)
    transformations = []

    already_parsed = [fn.replace("_transformed", "").replace("_step1", "").replace("_step2", "") for fn in os.listdir(arguments.output_folder)]

    for fname in tqdm(list_of_files):
        if fname in already_parsed:
            continue
        if fname.endswith(".txt"):
            with open(os.path.join(arguments.folder_path, fname), 'r', encoding='utf-8') as f:
                text = f.read()
                transformer = transform(
                    system_prompt=arguments.system_prompt if arguments.system_prompt else None,
                    instruction_list=arguments.instruction_list.split(",") if arguments.instruction_list else None,
                    provider=arguments.provider,
                    model=arguments.model,
                    max_tokens=arguments.max_tokens
                )
                trans = transformer(text)
                transformations.append((fname, trans))

            for k, _trans in enumerate(transformer.intermediate_outputs):
                out_path = os.path.join(arguments.output_folder, fname.replace(".txt", f"_transformed_step{k+1}.txt"))
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(_trans)

def parse_json(arguments: argparse.Namespace):
    try:
        input_file_name = os.path.splitext(os.path.basename(arguments.input_path))[0]
        out_path = os.path.join(arguments.output_folder, f"{input_file_name}_{arguments.model}.jsonl")
        with open(out_path, 'r', encoding='utf-8') as f:
            list_of_dicts = [json.loads(d) for d in f.readlines()]
            id_cache = [d[arguments.id_field] for d in list_of_dicts]
    except Exception as e:
        id_cache = []
        print(f"First run for this input file, continuing with {out_path}. Errpr: {e}")

    with open(arguments.input_path, 'r', encoding='utf-8') as f:
        list_of_dicts = json.load(f)

        _transformer = transform(
                        system_prompt=arguments.system_prompt if arguments.system_prompt else None,
                        instruction_list=arguments.instruction_list.split(",") if arguments.instruction_list else None,
                        provider=arguments.provider,
                        model=arguments.model,
                        n=arguments.n,
                        max_tokens=arguments.max_tokens)

        for d in tqdm(list_of_dicts):
            text = "\n".join([d[tf] for tf in arguments.text_fields])
            id = d[arguments.id_field]

            if (id in id_cache) | (not text) | (text.strip()==""):
                continue

            try:
                _ = _transformer(text)

                input_file_name = os.path.splitext(os.path.basename(arguments.input_path))[0]
                out_path = os.path.join(arguments.output_folder, f"{input_file_name}_{arguments.model}.jsonl")
                with open(out_path, 'a', encoding='utf-8') as f:
                    for k, _trans in enumerate(_transformer.intermediate_outputs):
                        json_line = json.dumps({arguments.id_field: id, "k": k, "transformed_text": _trans})
                        f.write(json_line + "\n")
                    sleep(1)
            except Exception as e:
                print(f"Error transforming text for {id}: {e}")

def parse_csv(arguments: argparse.Namespace):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform text using LLMs')
    parser.add_argument('--system_prompt', type=str, help='System prompt for the LLM', default=None)
    parser.add_argument('--instruction_list', type=str, help='List of instructions for the LLM, items separated by commas', default=None)
    parser.add_argument('--provider', type=str, help='Provider for the LLM', default='google')
    parser.add_argument('--model', type=str, help='Model for the LLM', default='gemini-1.5-flash')
    parser.add_argument('--n', type=int, help='Number of instructions to apply', default=1)
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens for the LLM', default=8_000)
    parser.add_argument('--folder_path', type=str, help='Path to folder with .txt files', default=None)
    parser.add_argument('--input_path', type=str, help='Path to input json', default=None)
    parser.add_argument('--text_fields', nargs='+', type=str, help='Fields in json to transform', default=['patient'])
    parser.add_argument('--id_field', type=str, help='Field in json to transform', default='patient_uid')
    parser.add_argument('--output_folder', type=str, help='Path to output folder', default=None)
    args = parser.parse_args()

    assert(args.folder_path is None or args.input_path is None), "Please provide either a folder path or an input path, or none"

    if args.folder_path:
        if not args.output_folder:
            raise ValueError("Please provide an output folder")
        parse_folder_with_txt(args)
    elif args.input_path:
        if not args.output_folder:
            raise ValueError("Please provide an output folder")
        parse_json(args)
    else:
        transformer = text.transform(
            system_prompt=args.system_prompt if args.system_prompt else None,
            instruction_list=args.instruction_list.split(",") if args.instruction_list else None,
            provider=args.provider,
            model=args.model,
            n=args.n,
            max_tokens=args.max_tokens
        )
        print(transformer("A 64-year-old female patient with a history of hyperthyroidism on treatment (thiamazole 5 mg once daily, and levothyroxine 62 μg once daily, currently euthyroid with normal thyroid-stimulating hormone, free thyroxine), was referred to our department from a regional hospital following a spider bite, which took place in western Greece (Aetolia-Acarnania region). The bite occurred in the pre-tibial area of the left lower extremity, while cleaning a building in a rural area, early November of 2013. The spider was described as black with red marks, about 2 cm in size and although it was not preserved for identification, the description as well as the clinical signs suggested European black widow spider envenomation."))
