# gcloud auth login
# gcloud config set project "clinicalnlp-308710"
# gcloud compute tpus tpu-vm ssh --zone "europe-west4-a" "clinlp-tpu-1" --project "clinicalnlp-308710"

# !# Install PyTorch and PyTorch/XLA
# pip install torch torchvision
# pip install cloud-tpu-client
# pip install transformers sentencepiece
# pip install google-cloud-storage


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from google.cloud import storage
import json
import nltk
import torch_xla.core.xla_model as xm
import tqdm 

# Initialize storage client
storage_client = storage.Client()

# Specify bucket and file
bucket_name = 'apollo_corpus'
file_name = 'medicalGuideline_en_text.json'

# Access the bucket and blob
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_name)

# Download text data
text_data = blob.download_as_text()

# Split text data into individual documents
documents = text_data.strip().split('\n')
documents = [doc for doc in documents if doc.strip()]  # Remove empty strings

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to TPU
device = xm.xla_device()
print(str(device))
model.to(device)

# Set batch size
batch_size = 16  # Adjust based on TPU memory and document length

# Function to create batches
def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Create batches
batches = list(batchify(documents, batch_size))

# List to store translations
translated_texts = []

# Translate each batch
for batch in tqdm.tqdm(batches):
    # Tokenize the batch of documents
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate translations
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode the translations
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Collect the translations
    translated_texts.extend(translations)

# Combine translations
final_translation = '\n'.join(translated_texts)

# Output translations
print("Translated Documents:")
print(final_translation)
