"""
Python class to translate texts using neural machine translation models.
"""
from pubscience import share

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import torch
from typing import List, Literal, Dict, Tuple, Any
import nltk
nltk.download('punkt_tab')

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
                                "google/madlad400-3b-mt",
                                "facebook/mbart-large-50-many-to-many-mmt",
                                "vvn/en-to-dutch-marianmt"]= 'facebook/nllb-200-distilled-600M',
                 multilingual: bool=True,
                 use_gpu: bool=True,
                 provider: Literal['huggingface', 'local']='huggingface',
                 source_lang: str='eng_Latn',  # Source language code
                 target_lang: str='nld_Latn',  # Target language code
                 max_length: int=512,
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

        if use_gpu:
            self.device = "cuda:0" if (torch.cuda.is_available()) & (torch.cuda.device_count()==1) else "cpu"
        else:
            self.device = "cpu"

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

    def translate_long(self, text: str) -> str:
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        translated_text = ""
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            # Encode sentence to get its length in tokens
            sentence_length = len(self.tokenizer.encode(sentence))
            if current_length + sentence_length <= self.max_length:
                current_chunk += " " + sentence
                current_length += sentence_length
            else:
                # Translate the current chunk
                inputs = self.tokenizer([current_chunk.strip()], return_tensors="pt").to(self.device)
                translated = self.model.generate(**inputs)
                translated_chunk = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                translated_text += " " + translated_chunk

                # Start a new chunk with the current sentence
                current_chunk = sentence
                current_length = sentence_length

        # Translate any remaining text
        if current_chunk:
            inputs = self.tokenizer([current_chunk.strip()], return_tensors="pt").to(self.device)
            translated = self.model.generate(**inputs)
            translated_chunk = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            translated_text += " " + translated_chunk

        return translated_text.strip()

    def translate_long_batch(self, texts: List[str]) -> List[str]:
        # Split text into sentences

        output_texts = []
        for text in texts:
            sentences = nltk.sent_tokenize(text)
            translated_text = ""
            current_chunk = ""
            current_length = 0

            for sentence in sentences:
                # Encode sentence to get its length in tokens
                sentence_length = len(self.tokenizer.encode(sentence))
                if current_length + sentence_length <= self.max_length:
                    current_chunk += " " + sentence
                    current_length += sentence_length
                else:
                    # Translate the current chunk
                    inputs = self.tokenizer([current_chunk.strip()], return_tensors="pt").to(self.device)
                    translated = self.model.generate(**inputs)
                    translated_chunk = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    translated_text += " " + translated_chunk

                    # Start a new chunk with the current sentence
                    current_chunk = sentence
                    current_length = sentence_length

            # Translate any remaining text
            if current_chunk:
                inputs = self.tokenizer([current_chunk.strip()], return_tensors="pt").to(self.device)
                translated = self.model.generate(**inputs)
                translated_chunk = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                translated_text += " " + translated_chunk

            output_texts.append(translated_text.strip())
        return output_texts

    def translate_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
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
    ntm = TranslationNTM(source_lang='eng_Latn', target_lang='rus_Cyrl')
    text = "I love you my dearest Maria."
    translated_text = ntm.translate(text)
    print("*"*50)
    print(translated_text)
    print("*"*50)

    print("Monolingual test...")
    ntm = TranslationNTM(model='Helsinki-NLP/opus-mt-en-ru')
    text = "I love you my dearest Maria."
    translated_text = ntm.translate(text)
    print("*"*50)
    print(translated_text)
    print("*"*50)

    print("Monolingual test long")
    ntm = TranslationNTM(model='Helsinki-NLP/opus-mt-en-ru')
    text = """To be competitive on the free market there is little place for morality. Morally questioning your decisionmaking slows down the decision making, leads to economically sub-optimal results as morality is not rewarded in the short term, and unfortunately free-market capitalism is all about short term gains. This focus on short-term gains and predictable risk leads to risk aversion and an almost neurotic focus on the existing markets, i.e. free market capitalism does not lead to innovation because it is inherently conservative. The risks that entrepreneurs take are very real in an economic sense but trivial in an intellectual sense, because true radical innovation is unpredictable. That is why even huge conglomerates hardly produce radical innovation despite sitting on tens of billions of dollars of R&D budget. 
    Free market capitalism is not just amoral it is also inhumane: e.g. if the goal is maximisation of profit then working hours are increased until the productivity per labor cost no longer increases, regardless of the human cost.
    In the mean time labor markets are relatively opaque for the workers who are completely dependent on a job for their basic survival. Hence, in a tight labor market, the negotiation position of the worker is very weak and even non-existent if push comes to shove and there is no public safety net. This naturally drives the overall labor conditions, including wages, down.
    The only resolution for this is a safety net, democratically determined constraints on the market and a fully transparent labor market.
    It is easy to see why amorality works in a Darwinian reward system; besides the practical ease of being able to ignore personal moral liability and to instead, simply refer to the "market" - if I don't do it, someone else will - it opens up the possibility to dehumanise laborers, contract partners and customers.
    This is one aspect where effective globalisation neglects the importance of personal identity and cultural tradition, this forms a connection between international socialism and neoliberalism. One specific aspect is the divestment of national public shares in private enterprises, meaning that nationally owned companies were gradually sold on the international markets. Another aspect is the dehumanisation which under neoliberalism is the result of Darwinism and under international socialism is the result of the denial of individuality, the abstraction of the individual as an anonymous element of the collective. 
    If I don't do it, someone else will…
    The greatest excuse to perpetuate amoral behavior is also exemplary for one of the biggest flaws of free market capitalism, and one of the main drivers beyond the myriad of tragedies of commons that occur in every part of the economic system that is dependent on finite resources.
    If moral business conduct is a less effective business conduct, if doing the right thing is less profitable than what your competitors are doing then it follows that amorality itself and the willingness to do amoral things is a moat that you can leverage to beat the market.
    How does the liberalist justify amorality as a virtue?"""
    print(1.5*len(text.split(" ")), "tokens, roughly..")
    translated_text = ntm.translate_long(text)
    print("*"*50)
    print(translated_text)
    print("*"*50)
