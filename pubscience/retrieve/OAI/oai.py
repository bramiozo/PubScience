import requests
import re
import xml

# Radboud - fulltext not available?
# 

YYYY="2010"
sources = {
    'Tilburg': {'link': f'https://pure.uvt.nl/ws/oai?metadataPrefix=oai_dc&verb=ListRecords&set=publications:year{YYYY}:withFiles', 
                'pdf': 'identifier',
                'follow': 'direct',
                'lang': 'language', 
                'date': 'date', 
                'type': 'type'},
    'TUE': {'link': f'https://pure.uvt.nl/ws/oai?metadataPrefix=oai_dc&verb=ListRecords&set=publications:year{YYYY}:withFiles', 
                'pdf': 'identifier',
                'follow': 'direct',
                'lang': 'language', 
                'date': 'date', 
                'subject': 'subject',
                'type': 'type'},
    'UTwente':{},
    'TUD':{},
    'RUG': {},
    'Maastricht':{'link': f'https://cris.maastrichtuniversity.nl/ws/oai?verb=ListRecords&metadataPrefix=oai_dc&set=publications:year{YYYY}:withFiles', 
                'pdf': 'identifier',
                'follow': 'direct',
                'lang': 'language', 
                'date': 'date', 
                'type': 'type'},
    'UVA': {'link': f'https://dare.uva.nl/oai?metadataPrefix=oai_dc&verb=ListRecords&set=publications:year{YYYY}:withFiles', 
                'pdf': 'identifier',
                'follow': 'direct',
                'lang': 'language', 
                'date': 'date', 
                'type': 'type'},
    'Utrecht': {'link': 'http://dspace.library.uu.nl/oai/dare?verb=ListRecords&metadataPrefix=oai_dc&set=com_1874_296827',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'language': 'dc:language',
                'type': 'dc:type'
              },
    'UMCU': {'link': 'http://dspace.library.uu.nl/oai/dare?verb=ListRecords&metadataPrefix=oai_dc&set=com_1874_298213',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'language': 'dc:language',
                'type': 'dc:type'
             },
    'DARE': {'link': 'http://dspace.library.uu.nl/oai/dare?verb=ListRecords&metadataPrefix=oai_dc&set=dare',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'language': 'dc:language',
                'type': 'dc:type'
             },
    'Narcis':
}