import os
import zipfile
import gzip
import py7zr
import pandas as pd
import deepl
from google.cloud import translate_v2 as google_translate
from google.cloud import storage
import io

class TranslationService:
    def __init__(self, source_lang, target_lang, glossary=None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.glossary = glossary
        self.progress = 0
        self.cost = 0
        self.deepl_client = deepl.Translator(os.getenv("DEEPL_API_KEY"))
        self.google_client = google_translate.Client()

    def translate_text(self, text, service="deepl"):
        if service == "deepl":
            result = self.deepl_client.translate_text(text, source_lang=self.source_lang, target_lang=self.target_lang, glossary=self.glossary)
            self.cost += len(text) * 0.00002  # Assuming €20 per 1M characters
            return result.text
        elif service == "google":
            result = self.google_client.translate(text, source_language=self.source_lang, target_language=self.target_lang)
            self.cost += len(text) * 0.00002  # Assuming $20 per 1M characters
            return result["translatedText"]
        else:
            raise ValueError("Invalid translation service specified")

    def update_progress(self, increment):
        self.progress += increment

    def get_progress(self):
        return self.progress

    def get_cost(self):
        return self.cost

class CorpusReader:
    @staticmethod
    def read_file(file_path, file_type):
        if file_type == "txt.zip":
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                return zip_ref.read(zip_ref.namelist()[0]).decode('utf-8')
        elif file_type == "txt.gz":
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return f.read()
        elif file_type == "txt.7z":
            with py7zr.SevenZipFile(file_path, mode='r') as z:
                return z.readall()[z.getnames()[0]].decode('utf-8')
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)
            return df.to_dict(orient='records')
        else:
            raise ValueError("Unsupported file type")

    @staticmethod
    def read_from_gcs(bucket_name, blob_name, file_type):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if file_type in ["txt.zip", "txt.gz", "txt.7z"]:
            in_memory_file = io.BytesIO()
            blob.download_to_file(in_memory_file)
            in_memory_file.seek(0)
            return CorpusReader.read_file(in_memory_file, file_type)
        elif file_type == "parquet":
            df = pd.read_parquet(f"gs://{bucket_name}/{blob_name}")
            return df.to_dict(orient='records')
        else:
            raise ValueError("Unsupported file type")

class TranslationManager:
    def __init__(self, translation_service):
        self.translation_service = translation_service

    def translate_corpus(self, input_path, output_path, file_type, is_gcs=False):
        if is_gcs:
            bucket_name, blob_name = input_path.replace("gs://", "").split("/", 1)
            corpus = CorpusReader.read_from_gcs(bucket_name, blob_name, file_type)
        else:
            corpus = CorpusReader.read_file(input_path, file_type)

        if isinstance(corpus, list):  # For structured formats like parquet
            translated_corpus = []
            total_items = len(corpus)
            for i, item in enumerate(corpus):
                translated_item = {k: self.translation_service.translate_text(v) for k, v in item.items()}
                translated_corpus.append(translated_item)
                self.translation_service.update_progress((i + 1) / total_items * 100)
            
            pd.DataFrame(translated_corpus).to_parquet(output_path)
        else:  # For text files
            translated_text = self.translation_service.translate_text(corpus)
            self.translation_service.update_progress(100)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)

        print(f"Translation completed. Progress: {self.translation_service.get_progress()}%")
        print(f"Total cost: ${self.translation_service.get_cost():.2f}")

# Usage example:
# translation_service = TranslationService("EN", "DE", glossary="my_glossary")
# manager = TranslationManager(translation_service)
# manager.translate_corpus("input.txt.gz", "output.txt", "txt.gz")
# manager.translate_corpus("gs://my-bucket/input.parquet", "output.parquet", "parquet", is_gcs=True)
