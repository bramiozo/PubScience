# https://huggingface.co/docs/hub/repositories-licenses
licenses = [
            'apache-2.0', 'mit', 'openrail', 'bigscience-openrail-m', 'creativeml-openrail-m',
            'bigscience-bloom-rail-1.0', 'bigcode-openrail-m', 'afl-3.0', 'artistic-2.0', 'bsl-1.0',
            'bsd', 'bsd-2-clause', 'bsd-3-clause', 'bsd-3-clause-clear', 'c-uda', 'cc', 'cc0-1.0',
            'cc-by-2.0', 'cc-by-2.5', 'cc-by-3.0', 'cc-by-4.0', 'cc-by-sa-3.0', 'cc-by-sa-4.0',
            'cc-by-nc-2.0', 'cc-by-nc-3.0', 'cc-by-nc-4.0', 'cc-by-nd-4.0', 'cc-by-nc-nd-3.0',
            'cc-by-nc-nd-4.0', 'cc-by-nc-sa-2.0', 'cc-by-nc-sa-3.0', 'cc-by-nc-sa-4.0',
            'cdla-sharing-1.0', 'cdla-permissive-1.0', 'cdla-permissive-2.0', 'wtfpl', 'ecl-2.0',
            'epl-1.0', 'epl-2.0', 'etalab-2.0', 'eupl-1.1', 'agpl-3.0', 'gfdl', 'gpl', 'gpl-2.0',
            'gpl-3.0', 'lgpl', 'lgpl-2.1', 'lgpl-3.0', 'isc', 'lppl-1.3c', 'ms-pl', 'apple-ascl',
            'mpl-2.0', 'odc-by', 'odbl', 'openrail++', 'osl-3.0', 'postgresql', 'ofl-1.1', 'ncsa',
            'unlicense', 'zlib', 'pddl', 'lgpl-lr', 'deepfloyd-if-license', 'llama2', 'llama3',
            'llama3.1', 'llama3.2', 'gemma', 'unknown', 'other'
        ]

collections = {"es": "spanish-66f1460e7972f6224f479a17",
               "sv": "swedish-66f14687831eacfcad87bbb7",
               "ro": "romanian-66f14654c6592d7516edb1e9",
               "it": "italian-66f14649878c56e920b38fdb",
               "nl": "dutch-66f14641caf6968847a453a9",
               "cs": "czech-66f14639c46132b895c8ad55",
               "en": "english-66f14630ff0a35fed8e4c7de"}

repo_type = "dataset"

def description_text(name, description, language, license, tags):
    """
    Template for dataset card
    """
    metadata = f"""
---
id: {name}
name: {name}
description: {description}
license: {license}
language: {language}
tags: {tags}
---
"""

    text = f"""{metadata}
# Dataset Card for {" ".join(name.split("_")).title()}

This dataset was created by: [Author] \n\n

The source language: [Source Language]\n\n

The original data source: [Original Data Source]\n\n


# Data description

{description}


# Acknowledgement

This is part of the DT4H project with attribution [Cite the paper].

# Doi and reference

[DOI and reference to the source paper/s]

"""
    return text