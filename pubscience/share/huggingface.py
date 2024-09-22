import datasets
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict, Dataset

class HuggingFaceDatasetManager:
    def __init__(self, dataset_name, split='train'):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split)

    def save_to_disk(self, path):
        self.dataset.save_to_disk(path)

    def load_from_disk(self, path):
        self.dataset = Dataset.load_from_disk(path)

    def push_to_hub(self, repo_name, token):
        self.dataset.push_to_hub(repo_name, token=token)

    def load_from_hub(self, repo_name, split='train'):
        self.dataset = load_dataset(repo_name, split=split)
