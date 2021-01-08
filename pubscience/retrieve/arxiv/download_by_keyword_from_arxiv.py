import re
import requests
import json
import os
import sys
from collections import defaultdict
import time
from tqdm import tqdm
import logging
from tika import parser

logger = logging.Logger('Mr. Marvis')
os.chdir('/media/bramiozo/DATA-FAST/text_data/pubscience/arxiv')


search_key = sys.argv[1]
logger.warning(f'Keyword is: {search_key}')

arxiv_list= []
logger.warning('Opening JSON file')
with open('/media/bramiozo/DATA-FAST/text_data/pubscience/arxiv/arxiv-metadata-oai-snapshot.json') as json_file:
    json_file_readlines = json_file.readlines()

logger.warning('Putting dictionaries in list')
arxiv_list = [json.loads(l) for l in json_file_readlines]
    
logger.warning('Processing dictionaries and writing files')
cnt = 0
for _el in tqdm(arxiv_list):
    _id = _el['id']
    try:
        _abstract = _el['abstract']
        if search_key in _abstract:
            cnt += 1
            time.sleep(2)
            logger.warning(f"Writing {id} to disk")
            url = f'http://arxiv.org/pdf/{_id}'
            r = requests.get(url, stream=True)
    
            with open(f'pdfdump/{_id}.pdf', 'wb') as fd:
                fd.write(r.content)

            _full = parser.from_file(f'pdfdump/{_id}.pdf')

            with open(f'articledump/{_id}_full.txt', 'w') as ff:
                ff.write(_full['content'])

            with open(f'abstractdump/{_id}_abstract.txt', 'w') as fa:
                fa.write(_abstract)
            
    except:
        pass

logger.warning(f'Found {cnt} articles for the keyword {search_key}')
