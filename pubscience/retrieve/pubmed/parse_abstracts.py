import xml.etree.ElementTree as ET
import os
import sys
import re
import gc
from tqdm import tqdm
import argparse

# TODO: add --folder and --filename argument
# TODO: if --folder, loop through folder for .xml's
# TODO: use --folder/out/[filename].raw.txt or as output location unless specified otherwise with --outfolder
# TODO: use [root] of filename for [root]/out/[filename].raw.txt as output location if 
#       --folder and --outfolder not given 
filename = sys.argv[1]
rawout = filename+".abstract.raw"

with open(filename) as rf:
    xmls = [line.strip() for line in rf if len(line.strip())>0]
    xmls = "\r".join(xmls)
    
# split by </xml>
xmls_arr = xmls.split("<?xml version=\"1.0\" ?>")[1:]
del xmls
gc.collect()

print(f"There are {len(xmls_arr)} XML objects in {filename}")

with open(rawout, 'w', encoding='utf-8') as wf:
    while xmls_arr:
        _xml = xmls_arr.pop()
        xml_str = re.sub(r"<\?xml[^><]+\?>", "", _xml).strip()
        xml_str = re.sub(r"<\!DOC[^><]+>\r?<PubmedArticleSet>", "<PubmedArticleSet>", xml_str).strip()
        xml_str = "<xml>"+xml_str+"</xml>"
        xml_str = re.sub(r'<\/xml>{2,}', '</xml>', xml_str)
        xml_str = re.sub(r'<\/xml><\/xml>+', '</xml>', xml_str)
        xml_str = re.sub(r'>>', '>', xml_str)

        try:
            root = ET.fromstring(xml_str.encode('utf-8'))

            for idx, ab in tqdm(enumerate(root.iter('AbstractText'))):
                wf.write(ab.text+"\n")
        
            print(f"Number of abstracts in this chunk: {idx}")
        except Exception as e:
            print(f"XML processing failed with error; {e}")
            print("first characters:")
            print(xml_str[:100])
            print("last characters:")
            print(xml_str[-100:])
