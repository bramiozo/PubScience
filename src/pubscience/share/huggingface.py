from datasets import DatasetDict, Dataset
import argparse
from huggingface_hub import HfApi, DatasetCard, errors
from requests.exceptions import HTTPError
from huggingface_hub import add_collection_item, get_collection
import os
from typing import Optional

import config

# Example usage:
# python hf_dataset.py --organization "DT4H" --name "Example dataset"\
# --dataset_path data --name example_api --description 'This is an example dataset'\
#  --language es --license mit --token YOUR_TOKEN

def create_dataset_card(name, description, language, license, tags):
    """
    Gets main information and creates a dataset card using the template in config.py
    """
    text = config.description_text(name, description, language, license, tags)
    # Using the Template
    card = DatasetCard(content=text)

    return card

def push_to_huggingface(repo_id, dataset_path, card, token, private):
    api = HfApi(token=token)
    print(f"Attempting to push to Repository {repo_id}. \nRepo type {config.repo_type}\n Token {token}")
    try:
        # Check if repository exists by trying to fetch its info
        api.repo_info(repo_id=repo_id, repo_type=config.repo_type)  # You can adjust repo_type if it's a dataset or space
        print(f"Repository '{repo_id}' already exists.")
    except HTTPError as e:
        if e.response.status_code == 404:
            # If repository does not exist, create it
            print(f"Repository '{repo_id}' does not exist. Creating...")
            api.create_repo(token=token, repo_id=repo_id, private=private, repo_type=config.repo_type)
        else:
            if e.response.status_code == 409:
                print(f"Repository '{repo_id}' already exists. Continuing")
            else:
                raise e

    # Upload dataset files
    if dataset_path.endswith(".jsonl") | dataset_path.endswith(".json"):
        file_path = dataset_path
        dataset_path = os.path.dirname(dataset_path)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.relpath(file_path, dataset_path),
            repo_id=repo_id,
            repo_type=config.repo_type,
            token=token,
        )
    else:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.relpath(file_path, dataset_path),
                    repo_id=repo_id,
                    repo_type=config.repo_type,
                    token=token,
                )

    # Push dataset card
    card.push_to_hub(
                        repo_id,
                        token=token,
                        repo_type=config.repo_type,
        )

class HuggingFaceDatasetManager:
    def __init__(self, dataset: Dataset):
        self.dataset= dataset

    def save_to_disk(self, path):
        self.dataset.save_to_disk(path)

    def push_to_hub(self, repo_name, token):
        self.dataset.push_to_hub(repo_name, token=token)

def main():

    parser = argparse.ArgumentParser(description="Push dataset and dataset card to Hugging Face")
    parser.add_argument("--data_organization", default="DT4H", help="Organization to push the dataset to")
    parser.add_argument("--collection_organization", default="DT4H", help="Organization that owns the collection")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    # parser.add_argument("--repo_id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset files")
    parser.add_argument("--name", required=True, help="Name of the dataset")
    parser.add_argument("--description", required=True, help="Description of the dataset")
    parser.add_argument("--language", required=True, help="Language of the dataset")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--license", default="mit", choices=config.licenses, help="License of the dataset")
    parser.add_argument("--tags", nargs="+", default=[], help="Tags for the dataset")

    args = parser.parse_args()

    repo_id = args.name.replace(" ", "_").lower()
    repo_id = f"{args.data_organization}/{repo_id}" if args.data_organization else repo_id

    # Create dataset card
    card = create_dataset_card(args.name, args.description, args.language, args.license, args.tags)

    # Push dataset and card to Hugging Face
    push_to_huggingface(repo_id, args.dataset_path, card, args.token, private=args.private)

    # Add dataset to collection
    collection_id = f"{args.collection_organization}/{config.collections[args.language]}"
    print(f"Adding to collection {collection_id}")
    add_collection_item(collection_id, item_id=repo_id, item_type=config.repo_type, token=args.token, exists_ok=True)

    if config.repo_type == "dataset":
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        coll_url = f"https://huggingface.co/datasets/{collection_id}"

        print(f"Dataset and card successfully pushed to {repo_url}")
        print(f"Dataset successfully added to {coll_url}")

if __name__ == "__main__":
    main()
