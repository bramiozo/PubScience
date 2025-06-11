#!/bin/bash

source /home/bvanes/.cache/pypoetry/virtualenvs/pubscience-F-Cpi52l-py3.10/bin/activate
python ex12_pubmed.py --output-dir=/home/bvanes/data/pubmed_translations --temp-dir=/home/bvanes/data/temp --batch-size=8 --model=vvn/en-to-dutch-marianmt --max-length=256 --multi_gpu
