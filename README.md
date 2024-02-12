# PubScience
Repository for public-article extraction and mining.

Three components:
1. **Retrieve** text data from Arxiv/Biorxiv/Medrxiv or Pubmed/PMC
2. **Identify** relevant text from generic corpora
3. **Clean** the XML/JSON/.. etc. from the previous step and output cleaned text
4. **Translate** the pruned/cleaned text to Dutch
5. **Share** make shareable through e.g. Huggingface

Tools
https://camelot-py.readthedocs.io/en/master/
https://pdfminersix.readthedocs.io/en/latest/
https://pypi.org/project/pdftotext/
beautifulsoup
scrapy

# Retrieve 

* Use the API's to pull .pdf's, .xml's or .json's.
* Pull directly from ```http``` of ```ftp```.
* Parse from local files (parquet/csv.gzip).

# Identify 

Based on 
* relevant document embedders (bi-encoders/cross-encoders) or
* topic models, or
* supervised models, trained to distinguish between domain specific texts and generic/other texts

# Clean

Fix broken XML/JSON, and select text-sections using Beautifulsoup and other Python libraries, clean for non-word characters and e.g. formatting spans.

# Translate 

Use Bulk google Translate/DeepL/LLM's(GPT4/Gemini/etc) or open source translation models in combination with UMLS-based glossaries to translate the
cleaned text to Dutch. 

Key features:
* A domain specific glossary, and related,
* a domain specific vocabulary.
* A ```cache``` functionality to reduce translation cost, i.e. a dynamically programmed wrapper

# Share


Text extraction pipelines:
* download pdf, extract body text, translate, clean, store
* download XML, fix broken XML, extract body text, translate, clean, store
* download pdf, extract Dutch section, clean, store
