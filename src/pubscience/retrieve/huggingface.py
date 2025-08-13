import datasets
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict, Dataset
import argparse

class HuggingFaceDatasetManager:
    def __init__(self, dataset_name, split='train'):
        self.dataset_name = dataset_name
        self.split = split

    def load_from_disk(self, path):
        self.dataset = Dataset.load_from_disk(path)

    def load_from_hub(self, repo_name, split='train'):
        self.dataset = load_dataset(repo_name, split=split)

    def save_to_disk(self, path):
        self.dataset.save_to_disk(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Dataset Manager")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--textcol', type=str, required=True, help='Name of the text column')
    parser.add_argument('--dest', type=str, required=True, help='Absolute output location')
    args = parser.parse_args()
    print(f"Loading {args.dataset}")
    print("-"*30)
    print(f"Writing to {args.dest}")
