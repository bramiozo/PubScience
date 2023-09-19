import requests
import re
import xml


YYYY="2010"

def source_dict(YYYY="2010", fromDate=None, untilData=None):
  # fromDate/untilDate : this format 2019-01-01T00:00:00Z

  sources = {
      'Narcis': {'link': 'https://oai.narcis.nl/oai2',
                'pdf': ['dc:relationship', 'dc:identifier'], 
                'lang': 'dc:language',
                'date': 'dc:date',
                'type': 'dc:type',
                'resumptionToken': 'resumptionToken',
                'set': ['oa_publication', 'publication', 'thesis']
      },
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
      'VU': {'link': f'https://research.vu.nl/ws/oai?metadataPrefix=oai_dc&verb=ListRecords&set=publications:year{YYYY}:withFiles',
            'pdf': 'identifier',
            'follow': 'direct',
            'lang': 'language',
            'type': 'type',
            'date': 'date',
            'thesis_set': f'studenttheses:year{YYYY}:withFiles'
            },
      'Leiden': {'link': f'https://scholarlypublications.universiteitleiden.nl/oai2',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'language': 'dc:language',
                'type': 'dc:type',
                'set': ['hdl_1887_55785', 'hdl_1887_4540', 'hdl_1887_9744', 'hdl_1887_85175', 'hdl_1887_20801', 'open_access'],
                'inclusion': {'language': 'nl', 'type': 'Doctoral Thesis'}
      },
      'Erasmus':{},
      'TUDelft':{},
      'Radboud':{'link':f'https://repository.ubn.ru.nl/oai/request?verb=ListRecords&metadataPrefix=oai_dc&set={SET}',
                'pdf': 'dc:identifier',
                'follow': 'retrieve',
                'type': 'dc:type',
                'set': ['col_2066_119636', 'col_2066_119645', 'com_2066_13798']
              },
      'KNAW': {'link': f'https://pure.knaw.nl/ws/oai?metadataPrefix=oai_dc&verb=ListRecords&set={SET}',
              'pdf': 'identifier',
              'follow': 'direct',
              'language': 'language',
              'type': 'type',
              'set': ['openaire_cris_publications', f'publications:year{YYYY}:withFiles'],
              'inclusion': {'language': 'nl', 'type': 'Doctoral Thesis'}
              }
  }