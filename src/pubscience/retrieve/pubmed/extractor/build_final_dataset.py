from pathlib import Path
from dspipe import Pipe
import jsonlines
from utils import iterate_pubmed_xml
from wasabi import msg
from tqdm import tqdm

# Keep track of what articles we keep for reporting
stats = {
    "not_pmc": 0,
    "pmc": 0,
}
no_mesh = 0

f_save = "data/PUBMED_title_abstracts_2020_baseline.jsonl"
FOUT = jsonlines.open(f_save, mode="w")

meta_columns = ["pmid", "language", "mesh_ids", "mesh_terms", "pubdate_year", "pubdate_month"]


def compute(f0):
    global stats
    global no_mesh

    with jsonlines.open(f0, "r") as FIN:
        for row in tqdm(FIN):

            # Remove entries where they overlap with PMC
            if row["pmc"] is not None:
                stats["pmc"] += 1
                

            stats["not_pmc"] += 1

            # Concatenate
            item = {"meta": {}}
            item["text"] = "\n".join([row["title"], row["abstract"]])

            for c in meta_columns:
                item["meta"][c] = None

            # Build the meta information
            for k in meta_columns:
                try:
                    item["meta"][k] = row[k]
                except Exception as e:
                    no_mesh += 1

            # PMID should always exist and be an integer
            item["meta"]["pmid"] = int(item["meta"]["pmid"])

            # Save to the master file
            FOUT.write(item)


P = Pipe(source="data/baseline/parsed/", input_suffix=".jsonl", shuffle=True)

P(compute, 1)
msg.good(f"Saved to {f_save}")
msg.info(
    f"Saved {stats['not_pmc']:,}, filtered {stats['pmc']:,} articles that overlapped in PMC"
)

filesize = Path(f_save).stat().st_size
msg.info(f"Compressed filesize {filesize:,}")
