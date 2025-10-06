from pathlib import Path
from wasabi import msg
from dspipe import Pipe
from tqdm import tqdm
import bs4
from utils import iterate_pubmed_xml
import jsonlines


def compute(f0, f1):

    data = []
    for pmid, text in tqdm(iterate_pubmed_xml(f0)):

        # This is the bottleneck operation
        soup = bs4.BeautifulSoup(text, "lxml")

        article = {}
        article["abstract"] = soup.find("abstract")
        article["title"] = soup.article.find("articletitle")
        article["pmid"] = soup.find("pmid")
        pubdate = soup.find("pubdate")

        # Skip if any missing fields
        if any((v is None for v in article.values())):
            continue

        # Remove copyright information from abstract text
        copy = article["abstract"].find("copyrightinformation")
        if copy is not None:
            copy.decompose()

        # Convert to text, remove extra spacing.
        for k, val in article.items():
            article[k] = " ".join(val.get_text().strip().split())


        if pubdate is not None:
            try:
                article["pubdate_year"] = pubdate.find("year").get_text().strip()
                article["pubdate_month"] = pubdate.find("month").get_text().strip()
            except Exception as e:
                article["pubdate_year"] = ""
                article["pubdate_month"] = ""
        else:
            article["pubdate_year"] = ""
            article["pubdate_month"] = ""
        mesh_heading_list = soup.find("meshheadinglist")
        if mesh_heading_list is not None:
            article["mesh_ids"] = [tag['ui'] for tag in mesh_heading_list.find_all(attrs={"ui": True})]
            article["mesh_terms"] = [tag.get_text() for tag in mesh_heading_list.find_all('descriptorname')]
        else:
            article["mesh_id"] = []
            article["mesh_terms"] = []


        # Check if article has a PMCID to filter later
        pmc = soup.find("articleid", idtype="pmc")

        if pmc is not None:
            article["pmc"] = pmc.get_text()
        else:
            article["pmc"] = None

        # Check if article has language tag to filter later
        lang = soup.find("language")

        if lang is not None:
            article["language"] = lang.get_text()
        else:
            article["language"] = None

        data.append(article)

    # Only write at the end to mark as success
    with jsonlines.open(f1, "w") as FOUT:
        FOUT.write_all(data)

    msg.good(f"Finished {f1}, saved {len(data)} articles")


def safe_compute(*args):
    try:
        compute(*args)
    except:
        print(f"Failed {args}")


P = Pipe(
    source="data/baseline/gz",
    dest="data/baseline/parsed",
    input_suffix=".gz",
    output_suffix=".jsonl",
    shuffle=False,
)

P(compute, -1)
