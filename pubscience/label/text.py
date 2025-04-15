"""
Labeling functions using LLMs, on/off premise.
"""

from typing import List
from pydantic import BaseModel

import os
import re
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

unsloth_models = [
    "unsloth/gemma-3-1b-it-GGUF",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
    "unsloth/Phi-4-mini-instruct-GGUF"
]

class llm_input(BaseModel):
    instruction: str
    text_to_transform: str

    def __str__(self) -> str:
        return "{" + f"'instruction': '{self.instruction}', 'text_to_transform': '{self.text_to_transform}'" + "}"
    def __repr__(self) -> str:
        return self.__str__()

def _get_available_google_models(google_gen_client) -> List[str]:
    available_models = []
    for m in google_gen_client.models.list():
        available_models.append(m.name)
    return available_models


class extract():
    def __init__(self,
                 system_prompt: str,
                 instruction_list: List[str],
                 provider: Literal['google', 'anthropic', 'openai', 'groq', 'local']='local',
                 model: str|None=None,
                 temperature: float=0.25,
                 batch_size: int=1,
                 max_tokens: int=5048):

        assert(
            provider in ['google', 'anthropic', 'openai', 'groq', 'local']
        ), f"Provider {provider} not supported. Supported providers are: ['google', 'anthropic', 'openai', 'groq', 'local']"
        assert isinstance(system_prompt,str),f"system_prompt must be a string"
        assert isinstance(instruction_list,list),f"instruction_list must be a list"

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
        if system_prompt.strip()!="":
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
        if all(isinstance(i, str) for i in instruction_list) & len(instruction_list) > 0:
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
        elif self.provider == 'local':
            return self.__translate_local(InputText)
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
            return response.text.strip()
        except Exception as e:
            raise ValueError(f"Could not transform text with Google LLM: {e}")

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


    def __translate_local(self, InputText: llm_input) -> Dict[str, Any]:
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
        return self.tokenizer.batch_decode(response)[0].strip()

def parse_folder_with_txt(arguments: argparse.Namespace):
    list_of_files = os.listdir(arguments.folder_path)
    transformations = []

    re_replace_steps = re.compile(r'\_step[-0-9]{1,2}')
    already_parsed = [re_replace_steps.sub("", fn.replace("_transformed", "")) for fn in os.listdir(arguments.output_folder)]

    for fname in tqdm(list_of_files):
        if fname in already_parsed:
            continue
        if fname.endswith(".txt"):
            with open(os.path.join(arguments.folder_path, fname), 'r', encoding='utf-8') as f:
                text = f.read()
                transformer = extract(
                    system_prompt=arguments.system_prompt if arguments.system_prompt else "",
                    instruction_list=arguments.instruction_list.split(",") if arguments.instruction_list else [],
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

    output_json = {t[0]:t[1] for t in transformations}
    with open(os.path.join(arguments.output_folder, 'Collected.json'), 'w') as fp:
        json.dump(output_json, fp)
    return output_json


def parse_json(arguments: argparse.Namespace):
    input_file_name = os.path.splitext(os.path.basename(arguments.input_path))[0]
    out_path = os.path.join(arguments.output_folder, f"{input_file_name}_{arguments.model}.jsonl")
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            list_of_dicts = [json.loads(d) for d in f.readlines()]
            id_cache = [d[arguments.id_field] for d in list_of_dicts]
    except Exception as e:
        id_cache = []
        print(f"First run for this input file, continuing with {out_path}. Errpr: {e}")

    with open(arguments.input_path, 'r', encoding='utf-8') as f:
        list_of_dicts = json.load(f)
        _transformer = extract(
                        system_prompt=arguments.system_prompt if arguments.system_prompt else "",
                        instruction_list=arguments.instruction_list.split(",") if arguments.instruction_list else [],
                        provider=arguments.provider,
                        model=arguments.model,
                        max_tokens=arguments.max_tokens
                    )
        transformations = []
        for d in tqdm(list_of_dicts):
            text = "\n".join([d[tf] for tf in arguments.text_fields])
            id = d[arguments.id_field]

            if (id in id_cache) | (not text) | (text.strip()==""):
                continue

            try:
                trans = _transformer(text)
                transformations.append((id, trans))

                input_file_name = os.path.splitext(os.path.basename(arguments.input_path))[0]
                out_path = os.path.join(arguments.output_folder, f"{input_file_name}_{arguments.model}.jsonl")
                with open(out_path, 'a', encoding='utf-8') as f:
                    for k, _trans in enumerate(_transformer.intermediate_outputs):
                        json_line = json.dumps({arguments.id_field: id, "k": k, "transformed_text": _trans})
                        f.write(json_line + "\n")
                    sleep(1)
            except Exception as e:
                print(f"Error transforming text for {id}: {e}")

    output_json = {t[0]:t[1] for t in transformations}
    with open(os.path.join(arguments.output_folder, 'Collected.json'), 'w') as fp:
        json.dump(output_json, fp)
    return output_json

def parse_txt(arguments: argparse.Namespace):
    # .txt file with a document per line
    input_file_name = os.path.splitext(os.path.basename(arguments.input_path))[0]
    out_path = os.path.join(arguments.output_folder, f"{input_file_name}_{arguments.model}.jsonl")
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            list_of_dicts = [json.loads(d) for d in f.readlines()]
            id_cache = [int(d["id"]) for d in list_of_dicts]
    except Exception as e:
        id_cache = []
        print(f"First run for this input file, continuing with {out_path}. Error: {e}")

    with open(arguments.input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        _transformer = extract(
                        system_prompt=arguments.system_prompt if arguments.system_prompt else "",
                        instruction_list=arguments.instruction_list.split(",") if arguments.instruction_list else [],
                        provider=arguments.provider,
                        model=arguments.model,
                        max_tokens=arguments.max_tokens
                    )
        transformations = []
        for idx, text in tqdm(enumerate(lines)):
            text = text.strip()

            if (idx in id_cache) | (not text) | (text.strip()==""):
                continue

            try:
                trans = _transformer(text)
                transformations.append((idx, trans))

                input_file_name = os.path.splitext(os.path.basename(arguments.input_path))[0]
                out_path = os.path.join(arguments.output_folder, f"{input_file_name}_{arguments.model}.jsonl")
                with open(out_path, 'a', encoding='utf-8') as f:
                    for k, _trans in enumerate(_transformer.intermediate_outputs):
                        json_line = json.dumps({"id": idx, "k": k, "transformed_text": _trans})
                        f.write(json_line + "\n")
                    sleep(1)
            except Exception as e:
                print(f"Error transforming text for line {idx}: {e}")

    output_json = {str(t[0]):t[1] for t in transformations}
    with open(os.path.join(arguments.output_folder, 'Collected.json'), 'w') as fp:
        json.dump(output_json, fp)
    return output_json


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
        if args.input_path.endswith('.json'):
            parse_json(args)
        elif args.input_path.endswith('.txt'):
            parse_txt(args)
        else:
            raise ValueError("Input path must be a json or txt file")
    else:
        transformer = extract(
            system_prompt=args.system_prompt if args.system_prompt else "",
            instruction_list=args.instruction_list.split(",") if args.instruction_list else [],
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens
        )
        print(transformer("A 64-year-old female patient with a history of hyperthyroidism on treatment (thiamazole 5 mg once daily, and levothyroxine 62 μg once daily, currently euthyroid with normal thyroid-stimulating hormone, free thyroxine), was referred to our department from a regional hospital following a spider bite, which took place in western Greece (Aetolia-Acarnania region). The bite occurred in the pre-tibial area of the left lower extremity, while cleaning a building in a rural area, early November of 2013. The spider was described as black with red marks, about 2 cm in size and although it was not preserved for identification, the description as well as the clinical signs suggested European black widow spider envenomation."))
