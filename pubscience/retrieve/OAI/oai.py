import requests
import re
import xml


YYYY="2010"

def source_dict(YYYY="2010", fromDate=None, untilData=None):
  # fromDate/untilDate : this format 2019-01-01T00:00:00Z

  sources = {
      'Tilburg': {'link': f'https://pure.uvt.nl/ws/oai?metadataPrefix=oai_dc', 
                  'pdf': 'identifier',
                  'follow': 'direct',
                  'lang': 'language', 
                  'date': 'date', 
                  'type': 'type'},
      'TUE': {'link': f'https://pure.uvt.nl/ws/oai?metadataPrefix=oai_dc', 
                  'pdf': 'identifier',
                  'follow': 'direct',
                  'lang': 'language', 
                  'date': 'date', 
                  'subject': 'subject',
                  'type': 'type'},
      'UTwente':{'link': f'https://ris.utwente.nl/ws/oai?metadataPrefix=oai_dc',
                'pdf': 'identifier',
                'follow': 'direct',
                'lang': 'language',
                'date': 'date',
                'subject': 'subject',
                'type': 'type'},
      'RUG': {'link': f'https://pure.rug.nl/ws/oai?metadataPrefix=oai_dc',
              'pdf': 'identifier',
              'follow': 'direct',
              'lang': 'language',
              'date': 'date', 
              'type': 'type'
              },
      'Maastricht':{'link': f'https://cris.maastrichtuniversity.nl/ws/oai?&metadataPrefix=oai_dc', 
                  'pdf': 'identifier',
                  'follow': 'direct',
                  'lang': 'language', 
                  'date': 'date', 
                  'type': 'type'},
      'UVA': {'link': f'https://dare.uva.nl/oai?metadataPrefix=oai_dc', 
                  'pdf': 'identifier',
                  'follow': 'direct',
                  'lang': 'language', 
                  'date': 'date', 
                  'type': 'type'},
      'UMCU': {'link': 'http://dspace.library.uu.nl/oai/dissertation',
                  'pdf': 'dc:identifier',
                  'follow': 'retrieve',
                  'language': 'dc:language',
                  'type': 'dc:type'
              },
      'DARE': {'link': 'http://dspace.library.uu.nl/oai/dare',
                  'pdf': 'dc:identifier',
                  'follow': 'retrieve',
                  'language': 'dc:language',
                  'type': 'dc:type'
              },
      'VU': {'link': f'https://research.vu.nl/ws/oai?metadataPrefix=oai_dc',
            'pdf': 'identifier',
            'follow': 'direct',
            'lang': 'language'
            },
      'Leiden': {'link': f'https://scholarlypublications.universiteitleiden.nl/oai2',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'language': 'dc:language',
                'type': 'dc:type'
      },
      'Radboud':{'link':f'https://repository.ubn.ru.nl/oai/openaire',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'type': 'dc:type',
                'sets': ['col_2066_119645', 'col_2066_119636', 'col_2066_69015']
              },
      'KNAW': {},
      'Erasmus':{},
      'TUDelft':{}
  }