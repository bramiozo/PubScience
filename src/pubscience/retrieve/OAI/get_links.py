# opens OAI, extracts 
# the publication_name, 
# the publication_type,
# the publication_description
# the publication_date
# the pdf_link

# open OAI: 
    # if resumptionToken present, reload with resumptionToken=..
    #  until resumptionToken is not present
    #  execute as :
    #    https://oai.narcis.nl/oai?verb=ListRecords&resumptionToken=XXXX
    
import oai

