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
from tqdm import tqdm
from typing import List, Dict, Any, Union
import gc
import benedict
from dotenv import load_dotenv
import argparse

load_dotenv(".env")

class Identify:
    def __init__(self, model_path: str|None=None, inclusion_terms: List|None=None, exclusion_terms: List|None=None,
        model_threshold: float=0.5, model_outcome: bool=True, device: str|int='cpu', id_col: str|None='id', text_col: str|None='text', textfile_readlines: bool=False, file_splitter_regex: str|None=None,
        max_write_size: int|None=None, max_read_size: int=512*1024,
        output_file: str='./output/output.jsonl', streaming: bool=False):
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

    @staticmethod
    def load_model(model_path, device):
        # Load the model from the given path
        return pipeline("text-classification", model=model_path, device=device)

    def _term_inclusion(self, text):
        # Check if the text contains any of the inclusion terms
        if not self.inclusion_terms:  # If no inclusion terms, consider all texts
            return not any(term in text for term in self.exclusion_terms)
        return any(term in text for term in self.inclusion_terms) and not any(term in text for term in self.exclusion_terms)

    def _model_inclusion(self, text):
        # Check if the model predicts inclusion
        if not self.model:
            return True

        result = self.model(text)
        print(f"Model outcome: {result}")
        score = result['score']

        if self.model_outcome:
            return score > self.model_threshold
        else:
            return 1-score > self.model_threshold

    def is_relevant(self, text):
        # Determine if a text is relevant based on terms and/or model
        term_relevant = self._term_inclusion(text)

        if self.model is None:
            return term_relevant

        model_relevant = self._model_inclusion(text)
        return term_relevant and model_relevant

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
                self.included_documents.append(doc_id)

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
                        self.included_documents.append(doc_id)

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
                        self.included_documents.append(doc_id)

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
            json.dump(self.included_documents, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify relevant content in XML documents")
    parser.add_argument("--directory", help="Directory containing documents")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    args = parser.parse_args()

    identifier = Identify(output_file=args.output, streaming=args.streaming)
    identifier.parse_directory(args.directory)
