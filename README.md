 <p align="center">
<img src="https://github.com/bramiozo/PubScience/blob/main/logo.webp" alt="image" width="200" height="auto" >
 </p>

<div align="center" style="background-color: white">
  <a href="https://www.datatools4heart.eu/">
    <img height="60px" src="dt4h_logo_color.svg" alt="DataTools4Heart Project"/>
  </a>
  &nbsp; &nbsp; &nbsp; &nbsp;
  <a href="https://www.ai4hf.com/">
    <img height="60px" src="ai4hf_logo.svg" alt="AI4HF Project"/>
  </a>
</div>

# PubScience

>[!TIP]
> This library comes with **zero** guarantees. If you want to see improvements, please add a gitissue, or contribute with a pull-request.
> If you want collaborate scientifically, please send [an email](mailto:b.vanes-3@umcutrecht.nl?subject=pubscience)

Repository for public-article extraction and mining.

Multiple components:
* **Select** (S) using API's to connect with 3rd party data
* **Retrieve** (S) text data from Arxiv/Biorxiv/Medrxiv or Pubmed/PMC
* **Parse** (S) process ingress XML/JSON/HTML/PDF/CSV into desired format
* **Identify** (S) relevant text from generic corpora
* **Deduplicate** (S) remove exact and mark approximate duplicates
* **Clean** the XML/JSON/.. etc. from the previous step and output cleaned text
* **Translate** the pruned/cleaned text to any target language
* **Transform** 
* **Anonymize** replace PII-information by placeholder terms
* **Share** make shareable through e.g. Huggingface
* **Augment** Add paraphrasing
* **Synonimize** identify and replace typos
* **Deabbreviate** identify and deabbreviate abbreviations
* **Stats** extract corpus statistics

Here the (S) indicates that these functions should be calleable in streaming mode. Especially to for smaller
domains, with limited storage capacity, we may not want to download Terabytes of corpora before we start our higher level
processing functions.




**Status** (minimum working example):
| Task          | In progress    | Completed  | Current Task  |
|---------------|----------------|------------|---------------|
| Select & Retrieve    |   [ ]   | [ ]        |               |
| Parse         |   [x]          | [ ]        |               |
| Identify      |   [ ]          | [ ]        |               |
| Deduplicate   |   [x]          | [ ]        |               |
| Clean         |   [x]          | [ ]        |               |
| Translate     |   [x]          | [ ]        |   *batch translation* |
| Transform     |   [ ]          | [ ]        |               |
| Label         |   [ ]          | [ ]        |  *first version based on Transform* |
| Anonymise     |   [x]          | [ ]        |               |
| Share         |   [ ]          | [ ]        |               |
| Augment       |   [ ]          | [ ]        |  *paraphrase* |
| Synonimize    |   [ ]          | [ ]        |               |
| Deabbreviate  |   [ ]          | [ ]        |               |
| Stats         |   [ ]          | [ ]        |               |

## Project descriptions
Here we can a bit more detail on the projects.

**Select & Retrieve**: interfaces with APIs for S2ORC/Pubmed/PMC/arxiv/medxriv/biorxiv/OAI and Huggingface.

The select function must be able to pull in data in streaming mode.

For Huggingface datasets this might be easy:
```python
from datasets import load_dataset

datasets = load_dataset('some/dataset', *params, streaming=True)
```

**Parse**: parser to normalise incoming data in JSON/YAML or HuggingFace dataset formats

**Identify**: functionality to identify medical texts in general corpora using supervised and self-supervised models

* Use pre-trained supervised models to identify relevant documents or text sections
* Use LLMs to identify relevant texts using in-context-learning
* Use seed-texts in combination with bi-encoder and cross-encoder models to find texts that are _near_

The core function _ab initio_ is to ease the creation and dissemination of Dutch clinical NLP work (including corpora) but
in principle this code is not limited to the Dutch language or the medical domain.

**Deduplicate**: remove exact duplicates and mark approximate duplicates.
Following the Llama3.1 recipe we use
* MinHash (see [Broder](https://ieeexplore.ieee.org/document/666900))
* Line-level deduplication; line-level frequency determination with cut-off, and selective removal

**Clean**: remove noise, code/format artifacts, escape/remove quotes
* duplicated n-gram coverage ratio (see [Rao et al.](https://arxiv.org/pdf/2112.11446)) to identify error logs
* Encoding degarbling
* file-format headers/endings
* using fasttext-based language detectors remove text-sections that exceed a pre-set fraction being _other lingual_ based on a per-line basis. e.g. if >50% of the paragraph or document is non-English we remove that paragraph

The core function here is the extract the _text intended to be read_.

**Translate**: using NMT and translation APIs optionally in combination with glossaries translate corporate to a target language.

**Transform**: on and off-premise LLM APIs to transform texts with multiple steps.

**Label**: on and off-premise LLM APIs to extract labels from texts.

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

Translation
```python

from pubscience.translate import llm

dotenv.load_dotenv('../../.env')
json_example = os.getenv('PMC_Patients')
OUTPUT_LOC = os.getenv('PMC_Patients_output')

BATCH_SIZE = 4
TEXT_IDS = ['title', 'patient']
ID_COLS = ['patient_id', 'patient_uid', 'PMID', 'file_path', 'pub_date']
META_COLS = ['age', 'gender']
MAX_LENGTH = 10_000
MAX_NUM_LINES = 250_293
SLEEP = 3
SYSTEM_PROMPT = """You are a faithful and truthful translator in the medical/clinical domain.
The user query is formatted as a dictionary {'source_language':..,'target_language':.., 'text_to_translate':..},
your response should ONLY consist of your translation"""

vars = {
    'model': 'gemini-1.5-flash',
    'provider': 'google',
    'source_lang': 'english',
    'target_lang': 'dutch',
    'max_tokens': MAX_LENGTH,
    'system_prompt': SYSTEM_PROMPT,
    'temperature': 0.15,
    'env_loc': '../../.run.env',
}

translator = llm.TranslationLLM(**vars)
batches = get_batches(json_example)
for batch in batches:
 translated_batch = translator.translate_batch(batch)
 ...
```

Transformation
```python
from pubscience.transform import text
transformer = text.transform(
    system_prompt=args.system_prompt if args.system_prompt else None,
    instruction_list=args.instruction_list.split(",") if args.instruction_list else None,
    provider=args.provider,
    model=args.model,
    n=args.n,
    max_tokens=args.max_tokens
)
```

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
* [html2text](https://github.com/Alir3z4/html2text)
* [Compact language detector](https://github.com/CLD2Owners/cld2)
* [justText](https://pypi.org/project/jusText/)
* [Fixes Text For You](https://ftfy.readthedocs.io/en/latest/)

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

We want to be able to do this as part of the select process. E.g. in case of the PubMed fulltext articles we
can use the abstract for semantic search to identify the relevant PubMed identifiers, which we can then selectively parse from the fulltext.

## Deduplicate

## Clean

Fix broken XML/JSON, and select text-sections using Beautifulsoup and other Python libraries, clean for non-word characters and e.g. formatting spans.


## Translate

Use Bulk google Translate/DeepL/LLM's(GPT4/Gemini/etc) or open source translation models in combination with UMLS-based glossaries to translate the
cleaned text to Dutch.

* External LLM APIs:
  * Google Gemini
  * OpenAI GPT4
  * Anthropic Claude
  * Groq (Llama, Mistral etc.)
* External translation APIs:
  * Google Translate
  * DeepL
* pre-trained NLMs (in principle all models that are availabe through Huggingface):
  * Maria NMT
  * NLLB200
  * M2M100
  * MADLAD400
  * T5
* pre-trained local LLMs (assuming quantized models):
  * Llama
  * Mistral
  * DCLM

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

# Pre-training Sources

## Dutch

As part of Dutch generic corpora
* SoNaR. Raw: $~$ **5GB**
* OSCAR. Raw: **41.5GB**
* COW14. Raw: **5.3GB**
* TnwC: ask permission to share with AMC. Raw: **3.1GB**
* CC100. Raw: **31.5GB**
* [mC4](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned/tree/main/mc4_nl_cleaned/train). Raw: **151GB**
* [Gigacorpus](https://web.archive.org/web/20240414113716/http://gigacorpus.nl/). Raw: **234GB**
* [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400/tree/main/data/nl), see [paper](https://arxiv.org/abs/2309.04662). Raw: **118.2GB**
* [PleIAs, common corpus](https://huggingface.co/datasets/PleIAs/Dutch-PD/tree/main) Raw: **180GB**

Here we have to note that CC100, mC4, GigaCorpus and MADLAD-400 all consists primarily (if not solely) of CC text.
The mC4 corpus is "filtered" for profanities and is therefore unsuitable as a basis for medical corpora. If you use multiple extraction versions of CC, be aware of the considerable required effort to deduplicates the text.

## English
As part of English corpora that we can filter, clean, then translate
* MIMIC III. **3.4GB**
* MIMIC III CXR: **0.421GB**
* MIMIC IV: **2GB**
* eICU: **0.32GB**
* [PMC Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients): $160$k patient records
* [PMC OA COMM](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/txt/):  **54GB** compressed,  **150GB** uncompressed
* [PMC OA NON COMM](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/txt/): **16GB** compressed,  **50GB** uncompressed, PMC OA represent more than _3M_ articles
* [Pubmed abstracts](https://github.com/thoppe/The-Pile-PubMed?tab=readme-ov-file)
* [S2ORC](https://github.com/allenai/s2orc): _81M_ abstracts, _8.1M_ fulltext, estimated **500GB**
* [Biorxiv/Medrxiv](https://connect.biorxiv.org/news/2020/04/18/tdm), [also](https://github.com/BlueBrain/Search/issues/459): _0.22M_ fulltext documents, estimated **20GB**
* [Clinical guidelines](https://huggingface.co/datasets/epfl-llm/guidelines)
* Medical PhD-theses
* [Apollo corpora](https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus).
* [UFAL multilingual corpora](https://ufal.mff.cuni.cz/ufal_medical_corpus)
* [SCIELO](https://huggingface.co/datasets/bigbio/scielo)

We have Italian corpora:
* [BioBERT Italian](https://huggingface.co/datasets/IVN-RIN/BioBERT_Italian)
* [Medical mT5](https://huggingface.co/datasets/HiTZ/Multilingual-Medical-Corpus)


And in principle we are able to identify medical texts in non-Dutch generic corpora followed by a
translation.

As part of Dutch clinical texts
* NtvG journals
* Dutch medical protocols
* medical health records from participating medical centers.
* [EMEA](https://opus.nlpl.eu/EMEA/corpus/version/EMEA)

## Spanish
* [CARES](https://huggingface.co/datasets/chizhikchi/CARES)
* [SBCC](https://zenodo.org/records/5513237)
* [ScSpainCrawel](https://github.com/PlanTL-GOB-ES/SciELO-Spain-Crawler)
* [Biomedical crawled corpus](https://zenodo.org/records/5513237)

# Finetuning source

## English

**Sentence similarity**
* [WikiMedical sentence similarity](https://huggingface.co/datasets/nuvocare/WikiMedical_sentence_similarity)
* MedSTS
* [MedNLI](https://huggingface.co/datasets/bigbio/mednli)
* [SciTail](https://huggingface.co/datasets/bigbio/scitail)
* [Medical Question Pairs](https://huggingface.co/datasets/bigbio/mqp)
* [Mediqa RQE](https://huggingface.co/datasets/bigbio/mediqa_rqe)
* [Mediqa NLI](https://huggingface.co/datasets/bigbio/mediqa_nli)
* [EHR Rel](https://huggingface.co/datasets/bigbio/ehr_rel)
* [BIOSSES](https://huggingface.co/datasets/bigbio/biosses)

**Term similarity**
* [UMNSRS](https://huggingface.co/datasets/bigbio/umnsrs)

**NER**

* [BLURB](https://huggingface.co/datasets/bigbio/blurb)
* [MultiCardioNER](https://temu.bsc.es/multicardioner/)
* [DisTEMIST](https://zenodo.org/records/7614764)
* [SympTEMIST](https://zenodo.org/records/10635215)
* [PharmaCoNER](https://zenodo.org/records/4270158)
* [MEDDOPROF](https://zenodo.org/records/7116201)
* [MEDDOPLACE](https://zenodo.org/records/8403498)
* [MEDDOCAN](https://zenodo.org/records/4279323)
* [Cantemist](https://zenodo.org/records/3978041)
* [CodiEsp](https://zenodo.org/records/3837305)
* [LivingNER](https://zenodo.org/records/7684093)
* [BC2GM](https://huggingface.co/datasets/spyysalo/bc2gm_corpus)
* [BC4CHEMD](https://huggingface.co/datasets/chintagunta85/bc4chemd)
* [CHEMDNER](https://huggingface.co/datasets/bigbio/chemdner)
* [BC5CDR](https://huggingface.co/datasets/tner/bc5cdr)
* [MedMentions](https://github.com/chanzuckerberg/MedMentions)
* [NCBI disease](https://huggingface.co/datasets/ncbi/ncbi_disease), and [here](https://huggingface.co/datasets/ncbi/ncbi_disease)
* [JNLPBA](https://huggingface.co/datasets/jnlpba/jnlpba), [this](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/JNLPBA) seems to be larger
* [CRAFT](https://github.com/UCDenver-ccp/CRAFT)
* [CONECO](https://huggingface.co/datasets/bigbio/coneco)
* [Flambe](https://huggingface.co/datasets/bigbio/flambe)
* [S800](https://species.jensenlab.org/)
* [NLM GENE](https://huggingface.co/datasets/bigbio/nlm_gene)
* [tmVar](https://huggingface.co/datasets/bigbio/tmvar_v3)
* [GENIA term](https://nlp.stanford.edu/~mcclosky/biomedical.html), also [see](https://huggingface.co/datasets/bigbio/genia_term_corpus)
* [SCAI Disease](https://huggingface.co/datasets/bigbio/scai_disease)
* [SCAI Chemical](https://huggingface.co/datasets/bigbio/scai_chemical)
* [NLM WSD](https://huggingface.co/datasets/bigbio/nlm_wsd)
* [n2c2](https://n2c2.dbmi.hms.harvard.edu/data-sets)
* [MuchMore](https://huggingface.co/datasets/bigbio/muchmore)
* [MSH WSD](https://huggingface.co/datasets/bigbio/msh_wsd)
* [GeneTag](https://huggingface.co/datasets/bigbio/genetag)
* [EBM PICO](https://huggingface.co/datasets/bigbio/ebm_pico)
* [DDI ner](https://huggingface.co/datasets/bigbio/ddi_corpus)
* [Chia](https://huggingface.co/datasets/bigbio/chia)
* [cHEBI](https://huggingface.co/datasets/bigbio/chebi_nactem)
* [AnEm](https://huggingface.co/datasets/bigbio/an_em)
* [AnatEm](https://huggingface.co/datasets/bigbio/anat_em)
* [NERO](https://www.nature.com/articles/s41540-021-00200-x)

**Entity classification**
* [BioScope](https://huggingface.co/datasets/bigbio/bioscope)

**De-abbreviation**
* [MeDAL](https://huggingface.co/datasets/bigbio/medal)


**Relationship extraction**
* [DDI re](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/ddi)
* [ChemProt](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/chemprot)
* [GAD](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/GAD), also [see](https://huggingface.co/datasets/bigbio/gad)
* [DrugProt](https://huggingface.co/datasets/bigbio/drugprot)
* [BioRed](https://huggingface.co/datasets/bigbio/biored)
* [LLL](https://huggingface.co/datasets/bigbio/lll)
* [GENIA relation](https://huggingface.co/datasets/bigbio/genia_relation_corpus)
* [BioRelex](https://huggingface.co/datasets/bigbio/biorelex)
* [BioNLP BB](https://huggingface.co/datasets/bigbio/bionlp_st_2019_bb), [BioNLP PC](https://huggingface.co/datasets/bigbio/bionlp_st_2013_pc), [BioNLP GRO ](https://huggingface.co/datasets/bigbio/bionlp_st_2013_gro), [BioNLP CG](https://huggingface.co/datasets/bigbio/bionlp_st_2013_cg), [BioNLP REL](https://huggingface.co/datasets/bigbio/bionlp_st_2011_rel), [BioNLP EPI](https://huggingface.co/datasets/bigbio/bionlp_st_2011_epi), [BioNLP GE](https://huggingface.co/datasets/bigbio/bionlp_st_2011_ge)

**Document classification**
* [HoC](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/HoC), also [see](https://huggingface.co/datasets/bigbio/hallmarks_of_cancer)
* [NLMChem](https://huggingface.co/datasets/bigbio/nlmchem)
* [LitCovid](https://huggingface.co/datasets/bigbio/bc7_litcovid)

**Summarisation**
* [MeqSum](https://huggingface.co/datasets/bigbio/meqsum)

**Q/A**
* [BioASQ](http://participants-area.bioasq.org/datasets/)
* [BioInstructQA](https://huggingface.co/datasets/BioMistral/BioInstructQA)
* [PubMedQA](https://github.com/justinphan3110/SciFive/tree/main/biot5x/data/pubmedqa)
* [SCIq](https://huggingface.co/datasets/bigbio/sciq)
* [SciFact](https://huggingface.co/datasets/bigbio/scifact)
* [Mediqa QA](https://huggingface.co/datasets/bigbio/mediqa_qa)
* [MedHop](https://huggingface.co/datasets/bigbio/medhop)
* [MedDialog](https://huggingface.co/datasets/bigbio/meddialog)
* [Evidence Inference](https://huggingface.co/datasets/bigbio/evidence_inference)
* [BioMRC](https://huggingface.co/datasets/bigbio/biomrc)
* [BioHowWhy](https://huggingface.co/datasets/bigbio/biology_how_why_corpus)
* [BioInfer](https://huggingface.co/datasets/bigbio/bioinfer)
* [AskAPatient](https://huggingface.co/datasets/bigbio/ask_a_patient)

## German

**NER**
* [CARDIODE](https://huggingface.co/datasets/bigbio/cardiode)
* [GGPONC2](https://huggingface.co/datasets/bigbio/ggponc2)
* [Bronco](https://huggingface.co/datasets/bigbio/bronco)

## Spanish

**Entity Classification**
* [NUBes](https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus)

**Document Classification**
* [CODIESP](https://huggingface.co/datasets/bigbio/codiesp)
* [MESINES](https://huggingface.co/datasets/bigbio/bioasq_2021_mesinesp)

## Translation of majority language sources

In principle all the English corpora can be used given an appropriate translation method.
