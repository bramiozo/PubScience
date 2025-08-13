import urllib.request, sys
import urllib.parse
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import re
import feedparser
import math
import time
from tika import parser
import logging
logger = logging.Logger('Mr. Marvis')


# TODO: should be environmentvariable or config setting
os.chdir('/media/bramiozo/DATA-FAST/text_data/pubscience/arxiv')

query = urllib.parse.quote(sys.argv[1])
try:
    num_res = int(sys.argv[2])
except:
    num_res = 100
max_res = 1000

logger.warning(f'Max results: {num_res}')

rawout = "abstractdump/"+query+".abstract.raw"


wf = open(rawout, 'w', encoding='utf-8')
num_iter = math.ceil(num_res/max_res)

art_idx = []
chunk_res = min(num_res, max_res)
for i in tqdm(range(num_iter)):
    start = i*max_res
    url = f'http://export.arxiv.org/api/query?search_query={query}&start={start}&max_results={chunk_res}'
    newline_re = re.compile(r'[\r\n]')
    quotes_re = re.compile(r'[\"]')
    with urllib.request.urlopen(url) as reader:
        feed = feedparser.parse(reader.read())
        for idx, entry in tqdm(enumerate(feed.entries)):

            summary_cleaned = newline_re.sub("", entry.summary)
            summary_cleaned = quotes_re.sub("'", summary_cleaned)
            summary_cleaned = "\""+summary_cleaned+"\""
        
            article_id = entry.id
            article_id = article_id.split("/")[-1]
            art_idx.append(article_id)
            article_id = "\""+article_id+"\""
            txt = article_id+";"+summary_cleaned+"\n"
            wf.write(txt)
        logger.warning(f"Number of abstracts in chunk {i}: {idx}")
wf.close()


logger.warning("Extracting .pdf's")
for id in tqdm(art_idx):
    url = f'http://arxiv.org/pdf/{id}'
    r = requests.get(url, stream=True)

    with open(f'pdfdump/{id}.pdf', 'wb') as fd:
        fd.write(r.content)

    fulltext = parser.from_file(f'pdfdump/{id}.pdf')

    with open(f'articledump/{id}_full.txt', 'w') as ff:
        ff.write(fulltext['content'])

    time.sleep(1)
