import os
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
from collections import defaultdict
import json
import argparse

class TAGS(BaseModel):
    start: int
    end: int
    tag: str

class NER(BaseModel):
    """
    {'tags': [{'start': 19, 'end': 32, 'tag': 'PER'},
              {'start': 6, 'end': 32, 'tag': 'LOC'}],
     'id': 'blaId',
     'text': "bladiebla"}
    """
    tags: List[TAGS]
    id: str
    text: str

class NERFormer():
    """
        Transform incoming formats in the NER format
        Incoming formats:
            (*.ann, *.txt) -> NER
            (db.tsv, *.txt) -> NER

        Outgoing formats:
            NER -> (*.ann, *.txt)
            NER -> (db.tsv, *.txt)
    """

    def __init__(self, ann_dir: str,
        txt_dir: str,
        db_path: str,
        out_path: str,
        name_map: Dict[str, str],
        write_to_file: bool = False):

        self.ann_dir = ann_dir
        self.txt_dir = txt_dir
        self.db_path = db_path
        self.out_path = out_path

        # check if (ann_dir exists and txt_dir) or (db_path exists and txt_dir)
        # if not raise ValueError
        if (ann_dir is None and db_path is None) or txt_dir is None:
            raise ValueError("Please provide a valid ann/db/txt directory")

        if out_path is not None:
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
        else:
            if write_to_file:
                raise ValueError("Please provide an output directory")

        self.write_to_file = write_to_file
        self.name_map = name_map

    def parse_db(self, db_path: str, text_list: List[str]) -> List[NER]:
        # read file
        # first line contains header with tab-separated names
        with open(db_path, 'r', encoding='utf-8') as fread:
            lines = fread.readlines()

        id_str = self.name_map["id"]
        tag_str = self.name_map["tag"]
        start_str = self.name_map["start"]
        end_str = self.name_map["end"]

        # get the header
        # get the index of the text column
        header = lines[0].strip().split("\t")

        res_dict = defaultdict(List[TAGS])
        for r in lines:
            rdict = dict(zip(header, r.strip().split("\t")))
            TAG = TAGS(start=rdict[start_str], end=rdict[end_str], tag=rdict[tag_str])
            res_dict[rdict[id_str]].append(TAG)

        # second iteration to add the text, we only need to to this once per id
        output_jsonl = []
        for k,v in res_dict.items():
            # get the text from the text file
            file_name = os.path.join(self.txt_dir, k, '.txt')
            with open(file_name, 'r', encoding='utf-8') as fread:
                text = fread.read()
            output_jsonl.append(NER(tags=v, id=k, text=text))
        return output_jsonl


    def parse_ann(self, text_list: List[str], ann_list: List[str]) -> List[NER]:
        pass

    def transform(self):
        # read the list of files in the directory for ann/txt
        #
        txt_list = os.listdir(self.txt_dir)

        if self.ann_dir is not None:
            ann_list = os.listdir(self.ann_dir)
            out_jsonl = self.parse_ann(txt_list, ann_list)
        elif self.db_path is not None:
            out_jsonl = self.parse_db(self.db_path, txt_list)

        if self.write_to_file:
            # write the output to the out_path
            with open(self.out_path, "w") as f:
                for line in out_jsonl:
                    f.write(json.dumps(line.dict()) + "\n")

        return out_jsonl


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Transform NER data")
    parser.add_argument("--ann_dir", type=str, help="Directory with ann files")
    parser.add_argument("--txt_dir", type=str, help="Directory with txt files")
    parser.add_argument("--db_path", type=str, help="Path to db file")
    parser.add_argument("--out_path", type=str, help="Path to output file")
    parser.add_argument("--name_map", type=str, help="Mapping of column names")
    parser.add_argument("--write_to_file", type=bool, help="Write to file")

    args = parser.parse_args()
    name_map = json.loads(args.name_map)
    ner = NERFormer(ann_dir=args.ann_dir,
                    txt_dir=args.txt_dir,
                    db_path=args.db_path,
                    out_path=args.out_path,
                    name_map=name_map,
                    write_to_file=args.write_to_file)
    ner.transform()
