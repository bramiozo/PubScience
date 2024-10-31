import os
import argparse
from typing import List
from collections import defaultdict
'''
function to collect .ann file with specific class and concatenate them into one .ann file...

e.g.
basefolder
    dis
        ann
            file1.ann
    proc
        ann
            file1.ann
    ..
        ann
            file1.ann

->

" ".join([file1_dis, file1_proc, ..])
'''

def join_anns(basefolder: str, classnames: List[str]):
    '''
        load .ann files from basefolder/ann per classname and concatenate them into one .ann file per
        file identifier
    '''

    # check if basefolder exists
    # if basefolder exists, then check if basefolder/ann exists, otherwise make it
    #
    if not os.path.exists(basefolder):
        raise ValueError("Please provide a valid basefolder")

    if not os.path.exists(os.path.join(basefolder, 'ann')):
        print(f"Creating ann folder in basefolder {basefolder}")
        os.mkdir(os.path.join(basefolder, 'ann'))

    ann_dict = defaultdict(str)
    for c in classnames:
        _anns = os.listdir(os.path.join(basefolder, c, 'ann'))
        _anns = [a for a in _anns if a.endswith('.ann')]
        _anns = sorted(_anns)
        anns = [os.path.join(basefolder, c, 'ann', a) for a in _anns]
        # load all ann files
        for i,a in enumerate(anns):
            with open(a, 'r', encoding='utf-8') as fread:
                file_str = fread.read()
                ann_dict[_anns[i]] += "\n" + file_str

        # write the concatenated anns to the basefolder
        for k,v in ann_dict.items():
            # remove leading newline
            #
            v = v.strip()
            # remove empty lines
            #
            v = "\n".join([l for l in v.split("\n") if l.strip() != ""])
            with open(os.path.join(basefolder, 'ann', k), 'w', encoding='utf-8') as fwrite:
                fwrite.write(v)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--basefolder", type=str, required=True, help="Base folder containing the class folders")
    argparser.add_argument("--classnames", type=str, nargs='+', required=True, help="List of class names")

    args = argparser.parse_args()

    # starting parser
    join_anns(args.basefolder, args.classnames)
