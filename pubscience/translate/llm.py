'''
This module contains classes to translate annotated and non-annotated corpora from
a source language to a target language.
'''

import os
from dotenv import load_dotenv

import google.generativeai as google_gen
from anthropic import Client as anthropic_client
from openai import Client as openai_client

from typing import Optional, Dict, List, Any

load_dotenv('.env')

"""
This module contains classes to translate annotated and non-annotated corpora.

Output: {'translated_text': bla,
         'proba_char_range': [(0, 30, 0.942), (30, 35, 0.72)...
         'source_lang': bla,
         'target_lang': bla
         }
"""



class TranslationLLM:
    def __init__(self,
        api_key: str,
        model: str,
        engine: str):

        self.api_key = api_key
        self.model = model
        self.engine = engine

        if engine == 'openai':
            self.client = openai_client(api_key=os.getenv('OPENAI_LLM_API_KEY'))
        elif engine == 'anthropic':
            self.client = anthropic_client(api_key=os.getenv('ANTHROPIC_LLM_API_KEY'))
        elif engine == 'google':
            google_gen.configure(api_key=os.getenv('GOOGLE_LLM_API_KEY'))
            self.client = google_gen.GenerativeModel(model_name=model)
