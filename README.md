 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/logo.webp" alt="image" width="300" height="auto" >
 </p>
<hr width=100%>
 
# PubScience
Repository for public-article extraction and mining.

Multiple components:
* **Select** using API's to connect with 3rd party data
* **Retrieve** text data from Arxiv/Biorxiv/Medrxiv or Pubmed/PMC
* **Parse** process ingress XML/JSON/HTML/PDF/CSV into desired format
* **Identify** relevant text from generic corpora
* **Deduplicate** remove exact and mark approximate duplicates
* **Clean** the XML/JSON/.. etc. from the previous step and output cleaned text
* **Translate** the pruned/cleaned text to any target language
* **Anonymize** replace PII-information by placeholder terms
* **Share** make shareable through e.g. Huggingface
* **Augment** Add paraphrasing
* **Synonimize** identify and replace typos
* **Deabbreviate** identify and deabbreviate abbreviations
* **Stats** extract corpus statistics

**Status** (minimum working example):
| Task          | In progress    | Completed  |
|---------------|----------------|------------|
| Select & Retrieve    |   [ ]   | [ ]        |
| Parse         |   [ ]          | [ ]        |
| Identify      |   [ ]          | [ ]        |
| Deduplicate   |   [ ]          | [ ]        |
| Clean         |   [x]          | [ ]        |
| Translate     |   [ ]          | [ ]        |
| Anonymise     |   [x]          | [ ]        |
| Share         |   [ ]          | [ ]        |
| Augment       |   [ ]          | [ ]        |
| Synonimize    |   [ ]          | [ ]        |
| Deabbreviate  |   [ ]          | [ ]        |
| Stats         |   [x]          | [ ]        |

## Project descriptions
Here we can a bit more detail on the projects.

**Select & Retrieve**: interfaces with APIs for S2ORC/Pubmed/PMC/arxiv/medxriv/biorxiv/OAI

**Parse**: parser to normalise incoming data in JSON/YAML or HuggingFace dataset formats

**Identify**: functionality to identify medical texts in general corpora using supervised and self-supervised models

**Deduplicate**: remove exact duplicates and mark approximate duplicates.
Following the Llama3.1 recipe we use
* MinHash (see [Broder](https://ieeexplore.ieee.org/document/666900))
* Line-level deduplication; line-level frequency determination with cut-off, and selective removal

**Clean**: remove noise, code/format artifacts, escape/remove quotes
* duplicated n-gram coverage ratio (see [Rao et al.](https://arxiv.org/pdf/2112.11446)) to identify error logs
* Encoding degarbling
* file-format headers/endings

The core function here is the extract the _text intended to be read_.

**Translate**: using NMT and translation APIs optionally in combination with glossaries translate corporate to a target language.

**Anonymize**: replace PII-information by placeholder terms using deidentification libraries and optional custom patterns.

**Share**: turn translated dataset into shared datasets including a dataset-card, license, etc.

**Augment**:  code to use paraphrasing for text generation

**Synonimize**: identify and replace typos, normalise variations of the same word

**Deabbreviate**: identify and deabbreviate abbreviations to reduce the ambiguity

**Stats**: extract stats from corpora, specifically; number of tokens, number of sentences, number of documents, vocab size


Basic operation:
```python

from pubscience import clean, deduplicate, anonymise
from pubscience.utils import Pipeline

Cleaner = clean(**clean_kwargs)
Deduplicate = deduplicate(**dedup_kwargs)
Deid = anonymise(**deid_kwargs) 


TextPipe = Pipeline([('Cleaner', Cleaner), 
                     ('Deduplicate',  Deduplicate), 
                     ('Deid', Deid)], 
                    n_jobs=16)

df['processed_text'] = TextPipe.fit_transform(df['raw_text']) 

# here Deduplicate adds a column to indicate the duplication degree
```


 <p "center">
<img src="https://github.com/bramiozo/PubScience/blob/main/PubScience.png" alt="image" width="300" height="auto" >
 </p>


Tools
* [camelot](https://camelot-py.readthedocs.io/en/master/)
* [pdfminer](https://pdfminersix.readthedocs.io/en/latest/)
* [pdftotext](https://pypi.org/project/pdftotext/)
* [fitz](https://github.com/pymupdf/PyMuPDF)
* [beautifulsoup](https://www.crummy.com/software/BeautifulSoup/)
* [scrapy](https://docs.scrapy.org/en/latest/)

Language:
This is primarily interesting because large scale text-processing can in principle be parallelized in
an embarassingly simple way, that means we should prefer natively heteregenous languages such as
* [Bend (GPU/CPU)](https://higherorderco.com/)
* [Julia (CPU)]()
* [Mojo (CPU)]()


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

A simple recipe could be (1) use command line string manipulation tools such as `grep`, `awk` and `cat` for the initial pruning
so for instance `grep "cardiale\|hartziekte\|vasculair\|tachycardie\|hartritme\|angina pectoris\|vaatlijden"  nl_clean_0000.jsonl > nl_clean_cardiale.jsonl`,
this is then followed by (2) a bi-encoder to check whether documents are 'near' medical texts or (3) a supervised model to identify medical texts.

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

And in principle we are able to identify medical texts in non-Dutch generic corpora followed by a 
translation.

As part of Dutch clinical texts
* NtvG journals
* Dutch medical protocols
* medical health records from participating medical centers.

## Spanish


## Translation of majority language sources

In principle all the English corpora can be used given an appropriate translation method.
