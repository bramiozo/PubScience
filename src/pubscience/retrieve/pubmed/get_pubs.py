import biopython

# db : 'pubmed', 'pmc'
# retstart
# retmax (max 100.000)
# rettype: 'full'

max_returns = 100
rettype='full'
db='pmc'
query='cardiovascular disease[abstract]'

ids = biopython.Entrez.esearch(db=db,
                               retmax=max_returns,
                               term=query)

fetch = biopython.Entrez.efetch(db=db, 
                                resetmode='xml', 
                                id=ids,
                                rettype=rettype)