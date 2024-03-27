 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/logo.webp" alt="image" width="300" height="auto" >
 </p>
<hr width=100%>
 
# PubScience
Repository for public-article extraction and mining.

Multiple components:
* **Select** using API's to connect with 3rd party data
* **Retrieve** text data from Arxiv/Biorxiv/Medrxiv or Pubmed/PMC
* **Parse** process XML/JSON/HTML/PDF/CSV in lists of texts 
* **Identify** relevant text from generic corpora
* **Deduplicate**
* **Clean** the XML/JSON/.. etc. from the previous step and output cleaned text
* **Translate** the pruned/cleaned text to Dutch
* **Anonymise** 
* **Share** make shareable through e.g. Huggingface

 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/PubScience.png" alt="image" width="300" height="auto" >
 </p>



Tools
* [camelot](https://camelot-py.readthedocs.io/en/master/)
* [pdfminer](https://pdfminersix.readthedocs.io/en/latest/)
* [pdftotext](https://pypi.org/project/pdftotext/)
* [fitz](https://github.com/pymupdf/PyMuPDF)
* [beautifulsoup](https://www.crummy.com/software/BeautifulSoup/)
* [scrapy](https://docs.scrapy.org/en/latest/)

## Select

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

## Deduplicate

## Clean

Fix broken XML/JSON, and select text-sections using Beautifulsoup and other Python libraries, clean for non-word characters and e.g. formatting spans.


## Translate 

Use Bulk google Translate/DeepL/LLM's(GPT4/Gemini/etc) or open source translation models in combination with UMLS-based glossaries to translate the
cleaned text to Dutch. 

Key features:
* A domain specific glossary, and related,
* a domain specific vocabulary.
* A ```cache``` functionality to reduce translation cost, i.e. a dynamically programmed wrapper
* Medical span alignment

 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/Aligned_translation.png" alt="image" width="900" height="auto" >
 </p>

When we translate _annotated corpora_ we need to make sure that the labeled spans are correctly translated **and** spanned.
We identify three approaches: (1) span-preserving translation, (2) span-inference of translation, (3) translate-then-align
### Span preserving translation
An example approach is given by [Seinen et al.](https://github.com/mi-erasmusmc/DutchClinicalCorpora); Seinen et al inject the span-information directly
in the original text prior to translation. Even though this might, arguably, negatively effect the translation quality the resulting models trained on the 
translated corpora showed similar accuracy to the model trained on the original English corpora. 

### Span-inference of translation
In principle we are able to create a training set with span-to-span information, e.g. as part of existing collective translation efforts (such as [datatools4heart](https://www.datatools4heart.eu/). 

### Translate-then-align
We translate a text _as is_: ```the fox jumps over the fence``` -> ```de vos springt over het hek```, then we identify the spans in the translated sentence.
One possible solution is to perform semantic similarity matching using multi-lingual (or at least bilingual) bi- or cross-encoders.

A more lexical/syntactic approach is followed by [Soares and Krallinger](https://arxiv.org/pdf/1905.01712.pdf), who use the [Aligner](https://sourceforge.net/projects/aligner/) tool.

## Anonymise

[DEDUCE](https://github.com/vmenger/deduce), [Presidio](https://github.com/microsoft/presidio)

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
* [PleIAs, common corpus](https://huggingface.co/datasets/PleIAs/Dutch-PD). Raw: **180GB**

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
* [Apollo corpora](https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus).

As part of Dutch clinical texts
* NtvG journals
* Dutch medical protocols

## Translation of majority language sources

In principle all the English corpora can be used given an appropriate translation method.
