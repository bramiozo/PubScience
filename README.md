# PubScience
Repository for public-article extraction and mining.

Three components:
1. **Retrieve** text data from Arxiv/Biorxiv/Medrxiv or Pubmed/PMC
2. **Clean** the XML/JSON/.. etc. from the previous step and output cleaned text
3. **Translate** the cleaned text to Dutch

# Retrieve 

Use the API's to pull .pdf's, .xml's or .json's.

# Clean

Fix broken XML/JSON, and select text-sections using Beautifulsoup and other Python libraries, clean for non-word characters.

# Translate 

Use Bulk google Translate/DeepL or open source translation models in combination with UMLS-based glossaries to translate the
cleaned text to Dutch.