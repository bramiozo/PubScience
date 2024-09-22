"""
Python class to translate texts using neural machine translation models.
"""
from pubscience import share

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import torch
from typing import List, Literal, Dict, Tuple, Any

import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

from typing import Dict, Tuple, Literal, Any, LiteralString

class TranslationNTM:
    def __init__(self,
                 model: Literal["facebook/nllb-200-3.3B",
                                "facebook/nllb-200-distilled-600M"
                                "facebook/m2m100_418M",
                                "facebook/mbart-large-50-many-to-many-mmt",
                                "vvn/en-to-dutch-marianmt"]= 'facebook/nllb-200-distilled-600M',
                 multilingual: bool=True,
                 use_gpu: bool=True,
                 provider: Literal['huggingface', 'local']='huggingface',
                 source_lang: str='eng_Latn',  # Source language code
                 target_lang: str='nld_Latn',  # Target language code
                 max_length: int=512
                 ):
        self.model = model
        self.multilingual = multilingual
        self.use_gpu = use_gpu
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

        if multilingual == False:
            logger.warning(f"""The model is not multilingual.
                Make sure the source language {source_lang} and target language {target_lang}
                are correct and coincide with the model {model}.""")
        else:
            logger.warning(f"""The model {model} is assumed to be multilingual.
                The source language {source_lang} and target language {target_lang}. Make sure that they
                coincide with the models language identifiers. For instance, nllb200 uses BCP-47 language codes.
                """)

        if self.provider not in ['huggingface', 'local']:
            raise ValueError("Unsupported provider. Choose either 'huggingface' or 'local'.")

        if self.provider == 'local':
            if not os.path.exists(self.model):
                raise FileNotFoundError(f"The specified model path {self.model} does not exist.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)

        if self.multilingual:
            self.tokenizer.src_lang = self.source_lang
            self.tokenizer.tgt_lang = self.target_lang

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model,
                                            device_map='auto' if self.use_gpu else 'cpu')
        except Exception as e:
            if "does not support `device_map" in str(e):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model)
            else:
                raise ValueError(e)

        self.config = self.model.config
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        self.device = "cuda:0" if (torch.cuda.is_available()) & (torch.cuda.device_count()==1) else "cpu"
        if torch.cuda.device_count()<=1:
            if self.device =='cuda:0':
                #ntm_model.half()
                self.model.to(self.device)
                self.model.eval()
            elif self.use_gpu == True:
                print("No GPU available. Using CPU.")
                self.model.to(self.device)
                self.model.eval()

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        if self.multilingual:
            outputs = self.model.generate(**inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=self.max_length)
        else:
            outputs = self.model.generate(**inputs,
                max_length=self.max_length)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def translate_batch(self, texts: str) -> List[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        if self.multilingual:
            outputs = self.model.generate(**inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=self.max_length)
        else:
            outputs = self.model.generate(**inputs, max_length=self.max_length)
        translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated_texts

if __name__ == "__main__":
    print("Multilingual test...")
    ntm = TranslationNTM(source_lang='eng_Latn', target_lang='nld_Latn')
    text = "Hello world, I am a test sentence for the neural machine translation model."
    translated_text = ntm.translate(text)
    print(translated_text)

    print("Monolingual test...")
    ntm = TranslationNTM(model='vvn/en-to-dutch-marianmt')
    text = "Hello world, I am a test sentence for the neural machine translation model."
    translated_text = ntm.translate(text)
    print(translated_text)
