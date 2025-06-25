#!/bin/bash

source /home/bvanes/.cache/pypoetry/virtualenvs/pubscience-F-Cpi52l-py3.10/bin/activate
accelerate launch --multi_gpu --num_processes=2 /home/bvanes/PubScience/examples/translate/ntm/ex12_pubmed.py --output-dir=/home/bvanes/data/pubmed_pmc/pubmed_translations --temp-dir=/home/bvanes/data/temp --batch-size=24 --model=vvn/en-to-dutch-marianmt --max-length=256 --multi_gpu
