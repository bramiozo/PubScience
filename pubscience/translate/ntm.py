"""
Python class to translate texts using neural machine translation models.
"""
from transformers.utils.quantization_config import BitsAndBytesConfig
from pubscience import share

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, BitsAndBytesConfig
import torch
from typing import List, Literal, Dict, Tuple, Any
import nltk
import pysbd
import re

from accelerate import Accelerator
from accelerate.utils import gather_object
import torch.distributed as dist
import torch.multiprocessing as mp

nltk.download('punkt_tab')

import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
# https://medium.com/@rakeshrajpurohit/model-quantization-with-hugging-face-transformers-and-bitsandbytes-integration-b4c9983e8996
class TranslationNTM:
    def __init__(self,
                 model_name: Literal["facebook/nllb-200-3.3B",
                                "facebook/nllb-200-distilled-600M",
                                "facebook/m2m100_418M",
                                "google/madlad400-3b-mt",
                                "facebook/mbart-large-50-many-to-many-mmt",
                                "vvn/en-to-dutch-marianmt"]= 'facebook/nllb-200-distilled-600M',
                 multilingual: bool=True,
                 use_gpu: bool=True,
                 provider: Literal['huggingface', 'local']='huggingface',
                 source_lang: str='eng_Latn',  # Source language code
                 target_lang: str='nld_Latn',  # Target language code
                 max_length: int=228,
                 sentence_splitter: Literal['nltk', 'pysbd']='pysbd',
                 num_beams: int=5,
                 use_quantisation: bool=False,
                 temperature: float=0.49,
                 use_accelerate: bool=False
                 #max_new_tokens: int=256
                 ):


        logger.info(f"Initialising translation with {model_name} \n\n")

        # Initialize accelerator first
        if use_accelerate and torch.cuda.device_count() > 1:
            logger.info("Multiple GPUs found, using Accelerator for inference")
            self.accelerator = Accelerator()
            self.use_accelerate = True
        else:
            if torch.cuda.device_count() > 1:
                logger.info("Multiple GPUs found, consider using Accelerator")
            self.accelerator = None
            self.use_accelerate = False

        self.sentence_splitter = sentence_splitter
        self.model_name = model_name
        self.multilingual = multilingual
        self.use_gpu = use_gpu
        self.provider = provider
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length # this is for the tokenizer/chunker
        self.num_pos_embeddings = AutoTokenizer.from_pretrained(model_name).model_max_length
        self.num_beams = num_beams
        self.use_quantisation = use_quantisation
        # Ensure max_context_length + max_new_tokens doesn't exceed num_pos_embeddings
        self.max_new_tokens = min(self.num_pos_embeddings - self.max_length, int(self.max_length*1.75))
        # set min_new_tokens ?
        self.gen_kwargs = {
            'max_new_tokens' : self.max_new_tokens,
            'num_beams' : self.num_beams,
            'early_stopping': True,
            'top_k': 50,
            'do_sample': False,
            #'min_p': 0.05,
            'repetition_penalty': 1.5,
            'encoder_repetition_penalty': 1.5,
            'length_penalty': 0.25,
            'no_repeat_ngram_size': 0,
            'num_return_sequences': 1,
        }
        if model_name not in ["vvn/en-to-dutch-marianmt","facebook/nllb-200-3.3B",
                                "facebook/nllb-200-distilled-600M"]:
            self.gen_kwargs.update({
                'top_p': 0.95,
                'temperature': temperature
            })

        if self.use_quantisation:
            self.quant_config = BitsAndBytesConfig(**{
                "_load_in_4bit": False,
                "_load_in_8bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_storage": "uint8",
                "bnb_4bit_quant_type": "fp4",
                "bnb_4bit_use_double_quant": False,
                "llm_int8_enable_fp32_cpu_offload": False,
                "llm_int8_has_fp16_weight": False,
                "llm_int8_skip_modules": None,
                "llm_int8_threshold": 6.0,
                "load_in_4bit": False,
                "load_in_8bit": True,
                "quant_method": "bitsandbytes"
            })
        else:
            self.quant_config = None

        print(f"Max length of input: {self.max_length}, Max length of output: {self.max_new_tokens}")
        if self.max_new_tokens < self.max_length:
            raise Warning(f"The maximum number of tokens that can be generated is {self.max_new_tokens}, "
                          f"the input length is {self.max_length}. The budget is {self.num_pos_embeddings}.\n\n"
                          f"We strongly advise that the max_length is set to be <1/2 the modelcapacity")

        if multilingual == False:
            logger.warning(f"""The model is not multilingual.
                Make sure the source language {source_lang} and target language {target_lang}
                are correct and coincide with the model {model_name}.""")
        else:
            logger.warning(f"""The model {model_name} is assumed to be multilingual.
                The source language {source_lang} and target language {target_lang}. Make sure that they
                coincide with the models language identifiers. For instance, nllb200 uses BCP-47 language codes.
                """)

        if self.provider not in ['huggingface', 'local']:
            raise ValueError("Unsupported provider. Choose either 'huggingface' or 'local'.")

        if self.provider == 'local':
            if not os.path.exists(self.model_name):
                raise FileNotFoundError(f"The specified model path {self.model_name} does not exist.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.multilingual:
            self.tokenizer.src_lang = self.source_lang
            self.tokenizer.tgt_lang = self.target_lang
            self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        else:
            self.forced_bos_token_id = None

        self.load_model()
        max_model_length = getattr(self.model.config, 'max_position_embeddings', None) or \
                            getattr(self.model.config, 'max_length', None)  \
                            or getattr(self.model.config, 'n_positions', None)
        assert(max_model_length is not None), "Model does not have max_position_embeddings, max_length or n_positions attribute."
        assert(self.max_length <= max_model_length), f"max_length {self.max_length} is greater than the model's max_position_embeddings {max_model_length}."
        logger.info(f"Model {self.model_name} loaded successfully.")
        logger.info(f"Model configuration: {self.model.config}")
        self.config = self.model.config

    def _input_device(self):
       # Grab the device of the first model parameter
       return next(self.model.parameters()).device

    def reset(self):
        """
            Empty GPU memory/cache
        """
        if self.use_gpu:
            torch.cuda.empty_cache()
        return True


    def load_model(self):
        if  self.use_accelerate and self.accelerator:
            # Let accelerate handle device placement
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                quantization_config=self.quant_config
            )
            # Prepare model with accelerate
            self.model = self.accelerator.prepare(self.model)
            self.device = self.accelerator.device
        else:
            if self.use_gpu:
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    if device_count == 1:
                        self.device = "cuda:0"
                    else:
                        # If multiple devices available, use the first one but could be configured
                        self.device = f"cuda:{torch.cuda.current_device()}"

                else:
                    self.device = "cpu"
                _device = self.device
                logger.info("MODEL LOADED ON GPU")
            else:
                self.device = "cpu"
                _device = self.device
                logger.info("MODEL LOADED ON CPU")

            used_device_map = False
            try:
                # First, try loading the model with device_map
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    device_map='auto' if self.use_gpu else None,
                    torch_dtype=torch.float32, # Explicitly set dtype to avoid meta tensors
                    quantization_config=self.quant_config
                )
                used_device_map = True
            except Exception as e:
                if "does not support `device_map" in str(e):
                    # If device_map is not supported, load without it
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        quantization_config=self.quant_config
                    )
                    # Manually move the model to the correct device
                    self.model = self.model.to(_device)
                elif "Cannot copy out of meta tensor" in str(e):
                    # Handle meta tensors
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        device_map='auto' if self.use_gpu else None,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        quantization_config=self.quant_config
                    )
                    used_device_map = True
                else:
                    raise ValueError(f"Unexpected error while loading model: {e}")
            self.model.eval()

        return self.model

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt",
            padding=True, max_length=self.max_length, truncation=True).to(self._input_device)
        if self.multilingual:
            outputs = self.model.generate(**inputs,
                **self.gen_kwargs,
                forced_bos_token_id=self.forced_bos_token_id)
        else:
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def translate_batch(self, texts: List[str]) -> List[str]:
            inputs = self.tokenizer(texts,
                                    return_tensors="pt",
                                    max_length=self.max_length,
                                    padding='longest',
                                    truncation=True).to(self._input_device)
            if self.multilingual:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs,
                        **self.gen_kwargs,
                        forced_bos_token_id=self.forced_bos_token_id,
                        )
            else:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs,
                        **self.gen_kwargs)
            translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translated_texts

    def translate_batch_distributed(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Multi-GPU batch translation using Accelerate
        """
        if not self.use_accelerate:
            return self.translate_batch(texts)

        # Split texts across processes
        with self.accelerator.split_between_processes(texts) as local_texts:
            local_results = []

            for i in range(0, len(local_texts), batch_size):
                batch_texts = local_texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding='longest',
                    truncation=True
                )

                # Use accelerator's send_to_device method
                inputs = self.accelerator.send_to_device(inputs)

                with torch.no_grad():
                    if self.multilingual:
                        outputs = self.model.generate(
                            **inputs,
                            **self.gen_kwargs,
                            forced_bos_token_id=self.forced_bos_token_id
                        )
                    else:
                        outputs = self.model.generate(**inputs, **self.gen_kwargs)

                    batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    local_results.extend(batch_translations)

        # Gather results from all processes
        all_results = gather_object(local_results)
        return all_results


    def translate_long(self, text: str) -> str:
        if self.sentence_splitter == 'nltk':
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text, language='english')
        elif self.sentence_splitter == 'pysbd':
            seg = pysbd.Segmenter(language="en")
            sentences = seg.segment(text)
        else:
            raise ValueError("Unsupported sentence splitter. Choose either 'nltk' or 'pysbd'.")
        translated_sentences = []
        current_chunk = ""

        for sentence in sentences:
            sentence_length = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            if sentence_length > self.max_length:
                # Split the long sentence into smaller parts
                sub_sentences = self._split_long_sentence(sentence)
                for sub_sentence in sub_sentences:
                    current_chunk_length = len(self.tokenizer.encode(current_chunk + " " + sub_sentence,
                                                                     add_special_tokens=False))
                    if current_chunk_length <= self.max_length:
                        current_chunk = f"{current_chunk} {sub_sentence}".strip()

                    else:
                        if current_chunk:
                            translated_chunk = self._translate_chunk(current_chunk.strip())
                            translated_sentences.append(translated_chunk)
                        current_chunk = sub_sentence
            else:
                current_chunk_length = len(self.tokenizer.encode(current_chunk + " " + sentence, add_special_tokens=False))
                if current_chunk_length <= self.max_length:
                    current_chunk = f"{current_chunk} {sentence}".strip()

                else:
                    if current_chunk:
                        translated_chunk = self._translate_chunk(current_chunk.strip())
                        translated_sentences.append(translated_chunk)
                    current_chunk = sentence

        if current_chunk:
            current_chunk_length = len(self.tokenizer.encode(current_chunk, add_special_tokens=False))
            translated_chunk = self._translate_chunk(current_chunk.strip())
            translated_sentences.append(translated_chunk)

        translated_paragraph = "\n".join(translated_sentences)

        return translated_paragraph

    def _split_long_sentence(self, sentence: str) -> list:
        """
        Splits a long sentence into smaller chunks that fit within max_length.
        """
        words = sentence.split()
        sub_sentences = []
        current_sub_sentence = ""
        for word in words:
            sub_sentence_length = len(self.tokenizer.encode(current_sub_sentence + " " + word, add_special_tokens=False))
            if sub_sentence_length <= self.max_length:
                current_sub_sentence += " " + word
            else:
                if current_sub_sentence:
                    sub_sentences.append(current_sub_sentence.strip())
                current_sub_sentence = word
        if current_sub_sentence:
            sub_sentences.append(current_sub_sentence.strip())
        return sub_sentences

    def _translate_chunk(self, chunk: str) -> str:
        # Tokenize the input without truncation
        inputs = self.tokenizer(
            [chunk],
            return_tensors="pt",
            truncation=False,  # Disable truncation to prevent input loss
            max_length=None,  # Ensure max_length does not enforce truncation
            padding='longest'
        ).to(self._input_device)

        input_token_length = inputs['input_ids'].shape[1]
        model_max_length = self.model.config.max_position_embeddings

        # Check if input exceeds model's maximum position embeddings
        if input_token_length > model_max_length:
            print(
                f"Input length ({input_token_length}) exceeds model's maximum length ({model_max_length}). Truncating input.")
            inputs = self.tokenizer(
                [chunk],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding='longest'
            ).to(self._input_device)

        # Generate translation with specified max_new_tokens
        with torch.no_grad():
            translated = self.model.generate(**inputs, **self.gen_kwargs)

        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def translate_long_batch(self, texts: List[str], batch_size: int = 5) -> List[str]:
        '''
        This assumes the batching of chunks in large documents. I.e. the batching is INTERNAL

        texts: List of strings to translate
        batch_size: Number of texts to translate in parallel; the higher the faster but more memory-intensive, adapt this depending on your GPU memory.
        '''
        all_translated_texts = []
        re_splitter = re.compile(r'(\n+)')
        for text in texts:
            sentences = re_splitter.split(text)
            translated_paragraphs = []

            chunks = self._prepare_chunks(sentences)
            if self.use_accelerate:
                translated_chunks = self.translate_batch_distributed(chunks, batch_size)
            else:
                translated_chunks = self._translate_chunks_batch(chunks, batch_size)
            translated_paragraph = " ".join(translated_chunks)
            translated_paragraphs.append(translated_paragraph)

            translated_text = "\n".join(translated_paragraphs)
            all_translated_texts.append(translated_text)

        return all_translated_texts

    def _prepare_chunks(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Accurate token length calculation
            sentence_length = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            if sentence_length > self.max_length:
                # Split the sentence further
                words = sentence.split()
                temp_sentence = ""
                for word in words:
                    temp_length = len(self.tokenizer.encode(temp_sentence + " " + word, add_special_tokens=False))
                    if temp_length <= self.max_length:
                        temp_sentence += " " + word
                    else:
                        chunks.append(temp_sentence.strip())
                        temp_sentence = word
                if temp_sentence:
                    chunks.append(temp_sentence.strip())
            else:
                temp_length = len(self.tokenizer.encode(current_chunk + " " + sentence, add_special_tokens=False))
                if temp_length <= self.max_length:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


    def _translate_chunks_batch(self, chunks: List[str], batch_size: int) -> List[str]:
        translated_chunks = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            inputs = self.tokenizer(batch_chunks, return_tensors="pt", truncation=True, max_length=self.max_length, padding='longest').to(self._input_device)
            with torch.no_grad():
                translated = self.model.generate(**inputs, **self.gen_kwargs,
                    forced_bos_token_id=self.forced_bos_token_id)
                batch_translations = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            translated_chunks.extend(batch_translations)

        return translated_chunks

    def translate_long_batch_v2(self, texts: List[str], batch_size: int = 8) -> List[str]:
        if self.sentence_splitter == 'nltk':
            nltk.download('punkt', quiet=True)
        elif self.sentence_splitter == 'pysbd':
            seg = pysbd.Segmenter(language="en")
        else:
            raise ValueError("Unsupported sentence splitter. Choose either 'nltk' or 'pysbd'.")

        output_texts = []
        for text in texts:
            if self.sentence_splitter == 'nltk':
                sentences = nltk.sent_tokenize(text, language='english')
            elif self.sentence_splitter == 'pysbd':
                sentences = seg.segment(text)
            chunks = []
            current_chunk_tokens = []
            current_length = 0

            # Prepare chunks
            for sentence in sentences:
                sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False, padding='longest')
                sentence_length = len(sentence_tokens)

                if sentence_length > self.max_length:
                    for i in range(0, sentence_length, self.max_length):
                        chunk_tokens = sentence_tokens[i:i + self.max_length]
                        chunks.append(chunk_tokens)
                else:
                    if current_length + sentence_length <= self.max_length:
                        current_chunk_tokens.extend(sentence_tokens)
                        current_length += sentence_length
                    else:
                        chunks.append(current_chunk_tokens)
                        current_chunk_tokens = sentence_tokens
                        current_length = sentence_length
            if current_chunk_tokens:
                chunks.append(current_chunk_tokens)

            # Translate chunks in batches
            # Translate chunks in batches
            if self.use_accelerate:
                # Convert chunk tokens back to text for accelerate processing
                chunk_texts = [self.tokenizer.decode(chunk_tokens, skip_special_tokens=True) for chunk_tokens in chunks]
                translated_chunks = self.translate_batch_distributed(chunk_texts, batch_size)
            else:
                translated_chunks = []
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_texts = [self.tokenizer.decode(chunk_tokens, skip_special_tokens=True) for chunk_tokens in batch_chunks]
                    inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=self.max_length, padding='longest').to(self._input_device)
                    with torch.no_grad():
                        translated = self.model.generate(**inputs, **self.gen_kwargs,
                        forced_bos_token_id=self.forced_bos_token_id)
                        batch_translations = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                    translated_chunks.extend(batch_translations)


            # Assemble translated text
            translated_text = " ".join(translated_chunks)
            output_texts.append(translated_text.strip())
        return output_texts



if __name__ == "__main__":
    print("Multilingual test...")
    ntm = TranslationNTM(source_lang='eng_Latn', target_lang='rus_Cyrl')
    text = "I love you my dearest Maria."
    translated_text = ntm.translate(text)
    print("*"*50)
    print(translated_text)
    print("*"*50)

    print("Monolingual test...")
    ntm = TranslationNTM(model_name='Helsinki-NLP/opus-mt-en-ru')
    text = "I love you my dearest Maria."
    translated_text = ntm.translate(text)
    print("*"*50)
    print(translated_text)
    print("*"*50)

    print("Monolingual test long")
    ntm = TranslationNTM(model_name='Helsinki-NLP/opus-mt-en-ru')
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
