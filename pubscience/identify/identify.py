"""
Identify relevant files in a directory, or documents in a corpus.

Based on a model from huggingface.co and/or based on term inclusion/exclusion lists
"""
import regex
import os
import sys
import json
import pandas as pd
import xml.etree.ElementTree as ET
from transformers import pipeline
from tqdm import tqdm
from typing import List, Dict, Any, Union

class Identify:
    def __init__(self, model_path: str|None, inclusion_terms: List|None, exclusion_terms: List|None,
        model_threshold: float=0.5, model_outcome: bool=True, device: str|int='cpu', id_col: str|None='id', text_col: str|None='text', textfile_readlines: bool=False):
        if model_path is not None:
            self.model = self.load_model(model_path, device)
        else:
            self.model = None
        self.inclusion_terms = inclusion_terms or []
        self.exclusion_terms = exclusion_terms or []
        self.included_documents = []
        self.model_outcome = model_outcome
        self.model_threshold = model_threshold
        self.device = device
        self.id_col = id_col
        self.text_col = text_col
        self.text_file_readlines = textfile_readlines

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
        """Parse a text file and identify relevant content"""
        try:
            filename = os.path.basename(document_path)

            with open(document_path, 'r', encoding='utf-8') as file:
                if self.text_file_readlines:
                    lines = file.readlines()
                    for i, line in enumerate(lines):
                        doc_id = f"{filename}_{i}"
                        if self.is_relevant(line):
                            self.included_documents.append(doc_id)
                else:
                    text = file.read()
                    doc_id = filename

                    # Check if this is being called from parse_directory
                    calling_frame = sys._getframe(1)
                    if calling_frame.f_code.co_name == 'parse_directory':
                        # Already using the filename as ID, so no change needed
                        pass

                    if self.is_relevant(text):
                        self.included_documents.append(doc_id)

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
