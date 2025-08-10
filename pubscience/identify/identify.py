"""
Identify relevant files in a directory, or documents in a corpus.

Based on a model from huggingface.co and/or based on term inclusion/exclusion lists
"""
import re
import os
import sys
import json
import pandas as pd
import xml.etree.ElementTree as ET
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Union, Literal
import gc
import benedict
from dotenv import load_dotenv
import argparse

load_dotenv(".env")

class Identify:
    def __init__(self, model_path: str|None=None, inclusion_terms: List|None=None, exclusion_terms: List|None=None,
        model_threshold: float=0.5, model_outcome: bool=True, device: str|int='cuda', id_col: str|None='id', text_col: str|None='text', textfile_readlines: bool=False, file_splitter_regex: str|None=None,
        max_write_size: int|None=None, max_read_size: int=512*1024, max_chunk_length: int=256,
        write_interval: int=1024, batch_size: int=64, output_file: str='./output/output.jsonl',
        streaming: bool=False, min_length: int=64, start_from_previous: bool=True):
        """
        Initialize the Identify class for identifying relevant documents.

        Args:
            model_path: Path to the huggingface model for text classification. If None, only term-based filtering is used.
            inclusion_terms: List of terms that should be included in relevant documents. If None, all documents pass term inclusion.
            exclusion_terms: List of terms that should NOT be in relevant documents. If None, no documents are excluded by terms.
            model_threshold: Threshold for the model's confidence score (0.0-1.0) to consider a document relevant.
            model_outcome: If True, use the model's positive class score; if False, use the negative class score.
            device: Device to run the model on ('cpu', 0, 1, etc. for GPU).
            id_col: Column name containing document IDs in structured data files.
            text_col: Column name containing document text in structured data files.
            textfile_readlines: If True, process text files line by line instead of as a whole file.
            file_splitter_regex: Regex pattern to split large text files into documents.
            max_write_size: Maximum number of bytes to write in a single batch.
            max_read_size: Maximum number of bytes to read in a single batch.
        """
        settings_loc = os.getenv('SETTINGS_YAML')
        settings = benedict.benedict.from_yaml(settings_loc)

        if model_path is not None:
            self.model = self.load_model(model_path, device)
        else:
            self.model = None
        self.inclusion_terms = inclusion_terms or []
        self.exclusion_terms = exclusion_terms or []

        if self.inclusion_terms is []:
            self.inclusion_terms = settings['inclusion_terms']
        if self.exclusion_terms is []:
            self.exclusion_terms = settings['exclusion_terms']

        self.included_documents = []
        self.model_outcome = model_outcome
        self.model_threshold = model_threshold
        self.device = device
        self.id_col = id_col
        self.text_col = text_col
        self.text_file_readlines = textfile_readlines
        self.file_splitter_regex = re.compile(file_splitter_regex) if file_splitter_regex else None
        self.max_write_size = max_write_size if max_write_size is not None else max_read_size
        self.max_read_size = max_read_size
        self.output_file = output_file
        self.streaming = streaming
        self.max_chunk_length = max_chunk_length
        self.batch_size = batch_size
        self.write_interval = write_interval
        self.min_length = min_length

        # if output file already exists, collect a set with the id's based on the self.id_col
        self.existing_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        doc_id = doc.get(self.id_col)
                        if doc_id is not None:
                            self.existing_ids.add(doc_id)
                    except Exception:
                        continue
        # if set empty set to None
        if len(self.existing_ids) == 0:
            self.existing_ids = None
        else:
            print(f"Found: {len(self.existing_ids)} prior processed ids, skipping these..", flush=True)

        self.start_from_previous = start_from_previous
    @staticmethod
    def load_model(model_path, device):
        # Load the model from the given path
        return pipeline("text-classification",
            model=model_path, device=device, truncation=True)

    def _term_inclusion(self, text):
        # Check if the text contains any of the inclusion terms
        #if not self.inclusion_terms:  # If no inclusion terms, consider all texts
        #    return not any(term in text for term in self.exclusion_terms)
        return any(term in text for term in self.inclusion_terms) and (not any(term in text for term in self.exclusion_terms))

    def _model_inclusion(self, text: str)->bool:
        # Check if the model predicts inclusion
        if not self.model:
            return True

        result = self.model(text)
        if result[0]['label'] == 'LABEL_0':
           score = 1 - result[0]['score']
        elif result[0]['label'] == 'LABEL_1':
            score = result[0]['score']
        else:
            raise ValueError(f"Did not recognize the label: {result[0]}")

        if self.model_outcome:
            return score > self.model_threshold
        else:
            return 1-score > self.model_threshold

    def _model_inclusion_batch(self, texts: List[str])->List[bool]:
        # Check if the model predicts inclusion
        if not self.model:
            return [True]

        results = self.model(texts)

        scores = []
        for result in results:
            if result['label'] == 'LABEL_0':
                score = 1 - result['score']
            elif result['label'] == 'LABEL_1':
                score = result['score']
            else:
                raise ValueError(f"Did not recognize the label: {result}")

            if self.model_outcome:
                 scores.append(score > self.model_threshold)
            else:
                scores.append(1-score > self.model_threshold)
        return scores

    def is_relevant(self, text) -> bool:
        # Determine if a text is relevant based on terms and/or model
        term_relevant = self._term_inclusion(text)

        if self.model is None:
            return term_relevant

        model_relevant = self._model_inclusion(text)
        return term_relevant or model_relevant

    def is_relevant_batch(self, texts: List[str])->List[bool]:
        # Determine if a text is relevant based on terms and/or model

        term_relevant = [self._term_inclusion(text) for text in texts]

        if self.model is None:
            return term_relevant

        model_relevant = self._model_inclusion_batch(texts)
        return [(t[0]==True) or (t[1]==True) for t in zip(term_relevant, model_relevant)]

    def parse_directory(self, directory_path):
        """Parse the directory and identify relevant files"""
        for filename in tqdm(os.listdir(directory_path), desc="Processing files"):
            file_path = os.path.join(directory_path, filename)

            if not os.path.isfile(file_path):
                continue

            if filename.endswith(".txt"):
                self.parse_txt(file_path)
            elif filename.endswith(".json"):
                self.parse_json(file_path)
            elif filename.endswith(".jsonl"):
                self.parse_jsonl(file_path)
            elif filename.endswith(".parquet"):
                self.parse_parquet(file_path)
            elif filename.endswith(".xml"):
                self.parse_xml(file_path)

    def parse_txt(self, document_path):
        """Parse a text file and identify relevant content

        Args:
            document_path: Path to the text file
            streaming: If True, read the file in chunks to handle very large files
        """

        try:
            filename = os.path.basename(document_path)

            if self.streaming:
                # Process file in chunks for large files
                with open(document_path, 'r', encoding='utf-8') as file:
                    chunk_index = 0
                    accumulated_text = ""
                    bytes_read = 0
                    bytes_write = 0
                    for line in file:
                        len_in_bytes = len(line.encode('utf-8'))
                        bytes_read += len_in_bytes
                        bytes_write += len_in_bytes
                        # Check if this line contains a document boundary
                        split_match = None
                        if self.file_splitter_regex is not None:
                            match = self.file_splitter_regex.search(line)
                            if match:
                                split_match = match.group()  # Get the actual text that matched

                        # Split if we found a boundary or exceeded size
                        if split_match or (bytes_read >= self.max_read_size):
                            # Add current line to accumulated text before processing
                            accumulated_text += line

                            # Create a doc_id based on the matching text if available
                            if split_match:
                                # Clean up the matching text to use as an ID
                                # Remove non-alphanumeric chars and limit length
                                clean_match = re.sub(r'[^a-zA-Z0-9]', '', split_match)[:20]
                                doc_id = f"{filename}_{clean_match}_{chunk_index}"
                            else:
                                doc_id = f"{filename}_chunk{chunk_index}"

                            if self.is_relevant(accumulated_text):
                                self.included_documents.append({'id': doc_id, 'text': accumulated_text})

                            if bytes_write >= self.max_write_size:
                                # we write out self.included_documents to disk and reset
                                self.write_documents_to_disk()
                                self.included_documents = []
                                gc.collect()
                                bytes_write = 0

                            # Reset for next document
                            accumulated_text = ""
                            bytes_read = 0
                            chunk_index += 1
                        else:
                            # No boundary, just accumulate the line
                            accumulated_text += line

                    # Handle the last document if needed
                    if accumulated_text:
                        doc_id = f"{filename}_chunk{chunk_index}"
                        if self.is_relevant(accumulated_text):
                            self.included_documents.append({'id': doc_id, 'text': accumulated_text})
                        self.write_documents_to_disk()
                        self.included_documents = []
                        gc.collect()
            else:
                with open(document_path, 'r', encoding='utf-8') as file:
                    if self.text_file_readlines:
                        lines = file.readlines()
                        for i, line in enumerate(lines):
                            doc_id = f"{filename}_{i}"
                            if self.is_relevant(line):
                                self.included_documents.append({'id': doc_id, 'text': line})
                    else:
                        text = file.read()
                        doc_id = filename

                        # Check if this is being called from parse_directory
                        calling_frame = sys._getframe(1)
                        if calling_frame.f_code.co_name == 'parse_directory':
                            # Already using the filename as ID, so no change needed
                            pass

                        if self.is_relevant(text):
                            self.included_documents.append({'id': doc_id, 'text': text})

                self.write_documents_to_disk()
                self.included_documents = []
                gc.collect()

        except Exception as e:
            print(f"Error processing {document_path}: {e}")

    def parse_json(self, document_path):
        """Parse a JSON document and identify relevant content"""
        try:
            with open(document_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            filename = os.path.basename(document_path)

            # Handle both single document and collection of documents
            if isinstance(data, dict):
                self._process_json_item(data, filename)
            elif isinstance(data, list):
                for item in data:
                    self._process_json_item(item, filename)

        except Exception as e:
            print(f"Error processing {document_path}: {e}")

    def _process_json_item(self, item, filename=None):
        """Process a single JSON item"""
        if self.text_col in item:
            text = item[self.text_col]
            doc_id = item.get(self.id_col, str(hash(text)))

            # Prepend filename if called from parse_directory
            if filename is not None:
                doc_id = f"{filename}_{doc_id}"

            if self.is_relevant(text):
                self.included_documents.append({"id": doc_id, "text": text})

    def parse_jsonl(self, document_path):
        """Parse a JSONL document and identify relevant content"""
        try:
            filename = os.path.basename(document_path)

            with open(document_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    # Check if this is being called from parse_directory
                    calling_frame = sys._getframe(1)
                    if calling_frame.f_code.co_name == 'parse_directory':
                        self._process_json_item(item, filename)
                    else:
                        self._process_json_item(item)

        except Exception as e:
            print(f"Error processing {document_path}: {e}")

    def parse_parquet(self, document_path):
        """Parse a Parquet file and identify relevant content"""
        try:
            df = pd.read_parquet(document_path)
            filename = os.path.basename(document_path)

            # Check if this is being called from parse_directory
            calling_frame = sys._getframe(1)
            called_from_directory = calling_frame.f_code.co_name == 'parse_directory'

            if self.text_col in df.columns:
                for _, row in df.iterrows():
                    text = row[self.text_col]

                    # Use the ID column if available, otherwise use text hash
                    if self.id_col in df.columns:
                        doc_id = row[self.id_col]
                    else:
                        doc_id = str(hash(text))

                    # Prepend filename if called from parse_directory
                    if called_from_directory:
                        doc_id = f"{filename}_{doc_id}"

                    if self.is_relevant(text):
                        self.included_documents.append({"id": doc_id, "text": text})

        except Exception as e:
            print(f"Error processing {document_path}: {e}")

    def parse_xml(self, document_path):
        """Parse an XML document and identify relevant content"""
        try:
            tree = ET.parse(document_path)
            root = tree.getroot()
            filename = os.path.basename(document_path)

            # Check if this is being called from parse_directory
            calling_frame = sys._getframe(1)
            called_from_directory = calling_frame.f_code.co_name == 'parse_directory'

            # Look for elements that might contain text
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text = elem.text.strip()

                    # Try to find an ID attribute or generate one
                    # Handle the case when self.id_col is None
                    if self.id_col is not None:
                        doc_id = elem.get(self.id_col) if elem.get(self.id_col) else str(hash(text))
                    else:
                        doc_id = str(hash(text))

                    # Prepend filename if called from parse_directory
                    if called_from_directory:
                        doc_id = f"{filename}_{doc_id}"

                    if self.is_relevant(text):
                        self.included_documents.append({"id": doc_id, "text": text})

        except Exception as e:
            print(f"Error processing {document_path}: {e}")

    def write_documents_to_disk(self):
        # make sure output_file is not None
        if self.output_file is None:
            raise ValueError("output_file must be specified")
        # make sure directory exists
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))

        with open(self.output_file, 'a') as f:
            for doc in self.included_documents:
                f.write(json.dumps(doc) + '\n')

    def parse_dataset(self, dataset: str|None, split="train", textcol="text", idcol="id", subset="nld_Latn",
        aggregation: Literal['simple', 'majority', 'selective']='simple'):
        """
        Parses Huggingface dataset; uses model and inclusion/exclusion list to identify relevant texts and writes them iteratively to parquet
        """
        if dataset is None:
            raise ValueError("dataset must be specified")

        dataset = load_dataset(dataset, split=split, name=subset, streaming=True)
        texts_batch = []
        texts_full = []
        doc_ids = []
        text_lens = []
        seen_previous = set()
        for item in tqdm(dataset):
            text = item[textcol]
            doc_id = item[idcol]

            text_len = len(text.split())
            text_lens.append(text_len)

            ### This is logic to start from previous
            #######################################
            if self.start_from_previous:
                if self.existing_ids is not None:
                    if doc_id in self.existing_ids:
                        seen_previous.add(doc_id)
                        continue

                if (len(seen_previous)<len(self.existing_ids)):
                    continue
            ########################################

            if text_len >=self.min_length:
                # TODO: majority vote and selective (use pysbd or paragraph delimiter)
                doc_ids += [doc_id]
                if len(self.included_documents) > self.write_interval:
                    print(f"Writing {self.write_interval} documents to disk...average text_len: {sum(text_lens)/len(text_lens)}")
                    self.write_documents_to_disk()
                    self.included_documents = []

                if aggregation == 'simple':
                    if len(text) > self.max_chunk_length:
                        max_len_text = " ".join(text.split(" ")[:self.max_chunk_length])
                    else:
                        max_len_text = text

                    texts_batch += [max_len_text]
                    texts_full += [text]

                    if len(texts_batch)>=self.batch_size:
                        scores = self.is_relevant_batch(texts_batch)
                        for k, score in enumerate(scores):
                            if score == True:
                                self.included_documents.append({"id": doc_ids[k], "text": texts_full[k]})
                        texts_batch = []
                        texts_full = []
                        doc_ids = []
                elif aggregation == 'majority':
                    raise NotImplementedError("Majority aggregation not yet implemented")
                    votes = []
                    word_list = text.split(" ")
                    for i in range(len(word_list)//self.max_chunk_length):
                        chunk = " ".join(word_list[i*self.max_chunk_length:(i+1)*self.max_chunk_length])
                        if self.is_relevant(chunk):
                            votes.append(1)
                        else:
                            votes.append(0)
                    if sum(votes) > len(votes)//2:
                        self.included_documents.append({"id": doc_id, "text": text})
                elif aggregation == 'selective':
                    # not yet available
                    # TODO: implement selective aggregation
                    raise NotImplementedError("Selective aggregation not yet implemented")
                else:
                    raise ValueError(f"Invalid aggregation type: {aggregation}")

        self.write_documents_to_disk()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify relevant content in documents")
    parser.add_argument("--directory", help="Directory containing documents")
    parser.add_argument("--dataset", help="Huggingface dataset")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--model", help="Name of the HF model..")
    parser.add_argument("--min_length", default=32, help="Minimum number of words of document..")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    args = parser.parse_args()

    identifier = Identify(output_file=args.output,
                          streaming=args.streaming,
                          model_path=args.model,
                          min_length=args.min_length)

    if args.directory:
        print("Parsing directory...")
        identifier.parse_directory(args.directory)

    if args.dataset:
        print("Parsing dataset...")
        identifier.parse_dataset(args.dataset)
