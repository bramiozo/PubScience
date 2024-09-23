import datasets
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict, Dataset

from typing import Optional

class HuggingFaceDatasetManager:
    def __init__(self, dataset: Dataset):
        self.dataset= dataset

    def save_to_disk(self, path):
        self.dataset.save_to_disk(path)

    def push_to_hub(self, repo_name, token):
        self.dataset.push_to_hub(repo_name, token=token)
