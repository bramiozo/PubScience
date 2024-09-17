from bs4 import BeautifulSoup, NavigableString
import re
import os
import json
import requests
import pprint
from time import sleep
import pandas
import dotenv
from tqdm import tqdm
dotenv.load_dotenv('.env')

def remove_p_tags(x):
    return x.replace("<p>", "").replace("</p>", "").replace("<br>", "")


def main(filename: str):

    print("Loading ATC data...")
    ATC_PATH = os.getenv('ATC_CODE_PATH')
    CUI_DEF_PATH = os.getenv('CUI_DEF_PATH')
    # read in the atc_codes
    ATC_CUI = pandas.read_parquet(r''+ATC_PATH)
    CUI_DEF = pandas.read_parquet(r''+CUI_DEF_PATH)
    ATC_STR = CUI_DEF.loc[CUI_DEF.LAT=='ENG'].merge(ATC_CUI, on='CUI', how='inner')
    ATC_STR = ATC_STR.assign(STR=ATC_STR.STR.str.lower())
    ATC_STR = ATC_STR[~ATC_STR.duplicated(subset=['CUI', 'ATC_code'])]


        
    '''
    <a href="./?code=A&showdescription=no">ALIMENTARY TRACT AND METABOLISM</a></b><br/>
    <p>Description</p><br/>
    <a href="./?code=A01&showdescription=no">STOMATOLOGICAL PREPARATIONS</a></b><br/>
    <p>Description</p><br/>
    <a href="./?code=A01A&showdescription=no">STOMATOLOGICAL PREPARATIONS</a></b><br/>
    <p>Description</p><br/>
    <a href="./?code=A01AA&showdescription=no">Caries prophylactic agents</a></b><br/>
    <p>Description</p><br/>
    '''

    pattern_string = r"\<a href\=\"\.\/\?code\=([A-Z0-9]{1,7})\&showdescription\=no\"\>([\w\s]+)\<\/a\>\<\/b\>\<br\/\>\n(\<p\>[A-z0-9\,\.\;\:\-\<\>\(\)\/\s]+\<\/p\>)?"
    atc_pattern = re.compile(pattern_string)

    list_of_final_dicts= [] # OMFG THIS IS UGLY
    errors = []
    COUNT = 0
    print("Start parsing")
    with open(filename, 'w') as file:
        for _atc_code, _atc_name in tqdm(ATC_STR[['ATC_code', 'STR']].values):
            sleep(3)
            url = f'https://atcddd.fhi.no/atc_ddd_index/?code={_atc_code}&showdescription=yes'

            response = requests.get(url)
            if response.status_code == 200:
                html_content = response.text

                results = re.findall(atc_pattern, html_content)

                list_of_dicts = []
                for res in results:
                    atc_code = res[0]
                    atc_name= res[1]
                    atc_desc = res[2]
                    list_of_dicts.append({
                        "code" : atc_code,
                        "name": atc_name,
                        "desc": remove_p_tags(atc_desc)
                    })

                final_dict = {}
                for level in range(len(list_of_dicts)):
                    if level<len(list_of_dicts)-1:
                        final_dict[list_of_dicts[level]['code']] = {
                            "name": list_of_dicts[level]['name'],
                            "desc": list_of_dicts[level]['desc'],
                            "child": {
                                "code": list_of_dicts[level+1]['code'],
                                "name": list_of_dicts[level+1]['name'],
                                "desc": list_of_dicts[level+1]['desc'],
                            }
                        }
                    else:
                        final_dict[list_of_dicts[level]['code']] = {
                                "name": list_of_dicts[level]['name'],
                                "desc": list_of_dicts[level]['desc'],
                                    "child": {
                                        'code': _atc_code,
                                        'desc': _atc_name,
                                        'name': atc_name
                                }
                        }
                list_of_final_dicts.append(final_dict)
                
                json_line = json.dumps(final_dict) + '\n'
                file.write(json_line)
                file.flush()                            
            else:
                errors.append((_atc_code, response.text))
            
            COUNT += 1
            
            if COUNT>5:
                break

if __name__ == "__main__":
    main("output.jsonl")
    print("Streaming write completed.")