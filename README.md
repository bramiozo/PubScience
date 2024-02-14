 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/PubScience.png" alt="image" width="300" height="auto" >
 </p>
<hr width=100%>
 
# PubScience
Repository for public-article extraction and mining.

Three components:
1. **Select** using API's to connect with 3rd party data
1. **Retrieve** text data from Arxiv/Biorxiv/Medrxiv or Pubmed/PMC
2. **Identify** relevant text from generic corpora
3. **Clean** the XML/JSON/.. etc. from the previous step and output cleaned text
5. **Deduplicate**
6. **Translate** the pruned/cleaned text to Dutch
7. **Anonymise** 
8. **Share** make shareable through e.g. Huggingface

 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/PubScience.png" alt="image" width="300" height="auto" >
 </p>



Tools
https://camelot-py.readthedocs.io/en/master/
https://pdfminersix.readthedocs.io/en/latest/
https://pypi.org/project/pdftotext/
beautifulsoup
scrapy

## Retrieve 

* Use the API's to pull .pdf's, .xml's or .json's.
* Pull directly from ```http``` of ```ftp```.
* Parse from local files (parquet/csv.gzip).

## Identify 

Based on 
* keyword lookup, using e.g. [FlashText](https://arxiv.org/abs/1711.00046)
* relevant document embedders (bi-encoders/cross-encoders) or
* topic models, or
* supervised models, trained to distinguish between domain specific texts and generic/other texts

## Clean

Fix broken XML/JSON, and select text-sections using Beautifulsoup and other Python libraries, clean for non-word characters and e.g. formatting spans.

## Deduplicate

## Translate 

Use Bulk google Translate/DeepL/LLM's(GPT4/Gemini/etc) or open source translation models in combination with UMLS-based glossaries to translate the
cleaned text to Dutch. 

Key features:
* A domain specific glossary, and related,
* a domain specific vocabulary.
* A ```cache``` functionality to reduce translation cost, i.e. a dynamically programmed wrapper

## Share


Text extraction pipelines:
* download pdf, extract body text, translate, clean, store
* download XML, fix broken XML, extract body text, translate, clean, store
* download pdf, extract Dutch section, clean, store

# Sources

## Dutch

As part of multi-lingual corpora
* [Aya collection](https://huggingface.co/datasets/CohereForAI/aya_collection)


As part of Dutch generic corpora
* SoNaR. Raw: $~$ **5GB**
* OSCAR. Raw: **41.5GB**
* COW14. Raw: **5.3GB**
* TnwC: ask permission to share with AMC. Raw: **3.1GB**
* CC100. Raw: **31.5GB**
* [mC4](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned). Raw: **151GB**
* [Gigacorpus](http://gigacorpus.nl/). Raw: **234GB**
* [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400), see [paper](https://arxiv.org/abs/2309.04662). Raw: **118.2GB**

As part of English corpora that we can filter, clean, then translate
* MIMIC III. **3.4GB**
* MIMIC III CXR: **0.421GB**
* MIMIC IV: **2GB**
* eICU: **0.32GB**
* [PMC OA COMM](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/):  **54GB** compressed,  **150GB** uncompressed
* [PMC OA NON COMM](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/): **16GB** compressed,  **50GB** uncompressed, PMC OA represent more than _3M_ articles
* [Pubmed abstracts](https://github.com/thoppe/The-Pile-PubMed?tab=readme-ov-file)
* [S2ORC](https://github.com/allenai/s2orc): _81M_ abstracts, _8.1M_ fulltext, estimated **500GB**
* [Biorxiv/Medrxiv](https://connect.biorxiv.org/news/2020/04/18/tdm), [also](https://github.com/BlueBrain/Search/issues/459): _0.22M_ fulltext documents, estimated **20GB** 
* [Clinical guidelines](https://huggingface.co/datasets/epfl-llm/guidelines)
* Medical PhD-theses

As part of Dutch clinical texts
* NtvG journals
* Dutch medical protocols

