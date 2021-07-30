import xml.etree.ElementTree as ET
import os
import sys
import re
import gc
from tqdm import tqdm

dir_name = sys.argv[1]
files = os.listdir(dir_name)
files = [f for f in files if '.xml' in f]

for filename in files:
    rawout = os.path.join(dir_name, 'out', filename+".full.raw")

    with open(os.path.join(dir_name, filename), encoding='latin1') as rf:
        xmls = [line.strip() for line in rf if len(line.strip())>0]
        xmls = "\r".join(xmls)


    # find LAST </article> BEFORE next <\?xml
    # remove/ignore everything after </article> and before <\?xml



    # append </pmc-articleset>

    # get all text within <body></body> with Beautifulsoup

    # remove all LaTex

    # continue, clean everything before <\?xml..

            

    xmls_arr = xmls.split("<?xml version=\"1.0\" ?>")[1:]
    del xmls
    gc.collect()

    print(f"There are {len(xmls_arr)} XML objects in {filename}")

    with open(rawout, 'w', encoding='utf-8') as wf:
        while xmls_arr:
            _xml = xmls_arr.pop()
            xml_str = re.sub(r"<\?xml[^><]+\?>", "", _xml).strip()
            xml_str = re.sub(r"<\!DOCTYPE.*>", "", xml_str).strip()
            xml_str = re.sub(r"<\!DOC[^><]+>\r?<PubmedArticleSet>", "<PubmedArticleSet>", xml_str).strip()
            xml_str = "<xml>"+xml_str+"</xml>"
            xml_str = re.sub(r'<\/xml>{2,}', '</xml>', xml_str)
            xml_str = re.sub(r'<\/xml><\/xml>+', '</xml>', xml_str)
            xml_str = re.sub(r'>>', '>', xml_str)

            try:
                root = ET.fromstring(xml_str.encode('latin1'))

                for idx, ab in tqdm(enumerate(root.iter('AbstractText'))):
                    wf.write(ab.text+"\n")
            
                print(f"Number of abstracts in this chunk: {idx}")
            except Exception as e:
                print(f"XML processing failed with error; {e}")
                print("first characters:")
                print(xml_str[:100])
                print("last characters:")
                print(xml_str[-100:])
