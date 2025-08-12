# this scripts contains algorithms to perform actions/transformations on corpora
# Function SeinenCorpusPrep: take a NEL corpus, e.g. with txt/csv and turn it into an integrated corpus with "..tex text [[span][cui]]"

import pandas as pd
import json
from typing import List, Dict
import argparse
from pubscience.utils.pubtator_loader import PubTatorCorpusReader

#TODO: load semantic types mapping

def pubtator_to_corpus(pubtator_file: str, id_col: str='id', text_col: str='text', span_col: str='tags', 
                       add_semantic_group: bool=False, add_seinen_integration: bool=False) -> List[Dict]:
    """
    Converts a PubTator file into a corpus format.
    :param pubtator_file: Path to the PubTator file.
    :param id_col: The column containing the document ID.
    :param text_col: The column containing the text.
    :param cui_col: The column containing the CUI (Concept Unique Identifier).
    :param span_col: Optional; the column containing the span information.
    :return: A list of dictionaries representing the corpus.
    """
    reader = PubTatorCorpusReader(pubtator_file)
    documents = reader.load_corpus()
    corpus = []
    for document in documents:
        temp_dict ={}
        temp_dict[id_col] = document.id
        temp_dict['title'] = document.title_text
        temp_dict[text_col] = document.abstract_text

        # go through entities
        ent_list = []
        for entity in document.entities:
            if add_semantic_group:
                sem_group = add_semantic_types(entity.semantic_type_id, reader.tui_sem_map)
            else:
                sem_group = None
            ent_list.append({
                'tui': entity.semantic_type_id,
                'cui': entity.entity_id,
                'start': entity.start_index,
                'end': entity.end_index,
                'text': entity.text_segment,
                'tag': sem_group
            })
        temp_dict[span_col] = ent_list

        if add_seinen_integration:
            temp_dict[f"{text_col}_seinen"] = SeinenCorpusPrep(temp_dict[text_col], temp_dict['title'], ent_list)

        corpus.append(temp_dict)
    return corpus


def add_semantic_types(tui: str, tui_sem_map: Dict[str,str]) -> str:
    return tui_sem_map.get(tui, 'Unknown')

def SeinenCorpusPrep(text: str, title:str|None, entities: List[Dict], sep: str=".")->str:
    """
    Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11258409/
    Takes a NEL dictionary and transforms it into an integrated corpus format, according to Seinen et al. (2024).
    """
    offset = 0
    entities_sorted = sorted(entities, key=lambda e: (e['start'], e['end']), reverse=False)
    current_text = text if title is None else f"{title}{sep}{text}"
    for entity in entities_sorted:
        start, end = entity['start'], entity['end']
        surface = entity.get('text', text[start:end])
        if 'cui' in entity:
            replacement =  f"[[{surface}][{entity['cui']}]]"
        else:
            replacement =  f"[[{surface}][{entity['tui']}]]"
        
        prior_text = current_text[:start + offset]
        post_text = current_text[end + offset:]
        current_text = prior_text + replacement + post_text
        offset += len(replacement) - (end - start)

    return current_text.strip()
 
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert PubTator file to corpus format.")
    argparser.add_argument('--pubtator_file', type=str, required=True, help='Path to the PubTator file.')
    argparser.add_argument('--id_col', type=str, default='id', help='Column name for document ID.')
    argparser.add_argument('--text_col', type=str, default='text', help='Column name for text content.')
    argparser.add_argument('--span_col', type=str, default='tags', help='Column name for span information.')
    argparser.add_argument('--output_file', type=str, required=True, help='Path to save the JSONL output corpus file.')

    args = argparser.parse_args()
    # check if folder of output_file exists, if not create it
    import os
    from pathlib import Path
    output_path = Path(argparser.parse_args().output_file).parent
    if not output_path.exists():
        os.makedirs(output_path)


    # create the corpus
    new_corpus = pubtator_to_corpus(
        pubtator_file=args.pubtator_file,
        id_col=args.id_col,
        text_col=args.text_col,
        span_col=args.span_col,
        add_semantic_group=False,  # Assuming we want to add semantic groups
        add_seinen_integration=True  # Assuming we want to integrate Seinen format
    )

    # write list of dictionaries to jsonl file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in new_corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

