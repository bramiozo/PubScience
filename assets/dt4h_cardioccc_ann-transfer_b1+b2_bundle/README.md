# DataTools4Heart CardioCCC Annotation Transfer Batch 1 + 2 Bundle

This folder contains the first and second batch of the multilingual Cardiology Clinical Case Corpus (CardioCCC) generated as part of the European project DataTools4Heart (DT4H).

Starting from a manually-annotated Gold Standard corpus in Spanish, the corpus texts were translated using machine translation and the Gold Standard annotations were projected into them using lexical annotation projection techniques. The result was then validated by clinical experts that are native speakers of each of the languages in the project's consortium. The experts used the annotation tool brat in side-by-side comparison mode, with the Gold Standard on one side of the screen and the projected files on the other. Given that the machine-translated texts are expected to contain translation errors, the annotators were encouraged to provide alternative translations for the annotated terms, which were then integrated in later versions of the corpus (more details below).

This first batch includes 258 (first batch) + 250 (second batch) documents in six languages: Czech, English, Spanish, Italian, Dutch, Romanian and Swedish. For the second batch, Romanian is missing since it is still in-the-works and will be added soon. Four labels are included: disease, medication, procedure and symptom. The Gold Standard annotations were created following dedicated guidelines which are available at [Zenodo](https://zenodo.org/records/10171647). For the projection process, the annotators used validation and correction guidelines which are also available at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.13151039).

Some basic statistics of the corpus are available upon request.

## Folder Structure

There is a folder for each batch, each in turn containing three versions of the data.

- `0_brat_originals`: These are the original documents as downloaded from brat. They have not been processed in any way (the only changes made were to delete invalid lines within the annotation files) and are only included for documentation purposes. It is strongly encouraged not to use them to train and evaluate models (or at all).
- `1_validated_without_sugs`: This is an intermediate version of the corpus with some preprocessing done. All labels have been separated (after homogenizing label names -- that is, removing any "_SUG" prefixes) and some of the annotations and comments have been cleaned. All labels share the same text, as no changes have been introduced yet to incorporate translation suggestions by the clinical experts.
- `2_validated_w_sugs`: This is the final version of the corpus that should be used to train and evaluate models. In this version, the texts have been changed to incorporate the translation suggestions provided by the clinical experts. This means that each label has its own texts.

In turn, within each corpus version folder the different languages are located in their own folder. These folders are named after the language's ISO code: `cz` (Czech), `en` (English), `es` (Spanish), `it` (Italian), `nl` (Dutch), `ro` (Romanian) and `sv` (Swedish).  

Next, within each language folder the annotations are separated label-wise. The code names for each label are `dis` (DISEASE), `med` (MEDICATION), `proc` (PROCEDURE) and `symp` (SYMPTOM). The annotations for version `0_brat_originals` follow a slightly different folder scheme, as they contain two labels merged together: `dis_proc` (diseases and procedures) and `symp_med` (symptoms and medications).

Finally, each label folder contains three different folders: `ann` (annotations in .ann format), `txt` (texts in .txt format and UTF-8 encoding) and `tsv` (annotations in .tsv format). The annotation formats are explained below.

Notably, the annotations in Spanish, being the original Gold Standard corpus, are the same in each version.

## Annotation Format

The annotations are presented in two standoff formats. On the one hand, they are available in .ann format, which is used by the annotation tool brat. In this format, each line represents an annotation with different parts separated by either tabulators or spaces. More information on this format is available on [brat's website](https://brat.nlplab.org/standoff.html). To process these files, you might use [brat-peek](https://github.com/s-lilo/brat-peek), a Python library dedicated to working specifically with .ann files (and created by the writer of this README file).

On the other hand, the annotations are also presented as a tab-separated value file (.tsv). There is one file for each set of annotations. It includes six columns (with headers): `name` (name of the document the annotation belongs to), `tag` (name of the label used for the annotation), `start_span` (starting point of the annotation in characters), `end_span` (ending point of the annotation in characters), `text` (text corresponding to the annotation) and `comment` (comment created by the annotator and associated to the annotation, if any).

## Contact

The final version of the corpus and this README document were prepared by Salvador Lima-López from the NLP for Biomedical Information Analysis (NLP4BIA) team at the Barcelona Supercomputing Center. For inquiries, please contact <salvador.limalopez@gmail.com>.
