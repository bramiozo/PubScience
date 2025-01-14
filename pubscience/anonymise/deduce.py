import deduce
import rapidfuzz
import re
import sys
import os
import argparse
import collections
from multiprocessing import Pool, cpu_count, Lock, set_start_method, get_context
from joblib import Parallel, delayed
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import numpy as np
import pandas as pd
import phonenumbers

weblink_re = re.compile(r'(https?\:\/\/[A-z0-9.\/\?\-\=]+)|(www\.[A-z0-9.\/\?\-\=]+)', re.IGNORECASE)

doi_re = re.compile(r'DOI\s[0-9\.\/\-\_]+', re.IGNORECASE)

bsn_re = re.compile(r'[^0-9]{1,}([0-9]{9})[^0-9]{1,}')

date_res = [re.compile(r'[12][0-9]{3}[\-\/\\]?[0-9]{1,2}[\-\/\\]?[0-9]{1,2}'),    
            re.compile(r'[0-9]{1,2}[\-\/\\]?[0-9]{1,2}[\-\/\\]?[12][0-9]{3}')]

phone_res = [
            re.compile(r'((\+31|0|0031)[\s\-]?[1-9]{1}[0-9]{8})(?![^<]*>)'),
            re.compile(r'((0)[1-9]{2}[0-9]{1,2}[\s\-]?[1-9][0-9]{5})(?![^<]*>)'),
            re.compile(r'((\\+31|0|0031)[1-9][0-9][\s\-]?[1-9][0-9]{6})(?![^<]*>)'), 
            re.compile(r'((\(\d{3}\)|\d{3})\s?\d{3}\s?\d{2}\s?\d{2})(?![^<]*>)'),
            re.compile(r'[0\+][0-9]{2,3}[\-\s]\d{4,8}')
            ]

patid_re = re.compile(r'((verwijzersnummer|verwijsnummer|pati[eë]ntnummer|patientnr|patnummer|patid|pat\.?num\.?)[\s\:\;]\s?([0-9]{5,12}))', re.IGNORECASE)

import spacy

class Deidentify(BaseEstimator, TransformerMixin):
    def __init__(self, method='deduce',
                 n_jobs=1, 
                 to_dataframe=False,
                 bsn_check=True,
                 date_check=False,
                 phone_check=False,
                 url_check=False,
                 doi_check=False,
                 pid_check=False,
                 number_replace=True,
                 backend='joblib', 
                 custom_list = None,
                 clear_brackets=False,
                 kwargs=None,
                 data_index=None, 
                 text_cols=None
                 ):
        self.method = method
        self.n_jobs = n_jobs
        self.to_dataframe = to_dataframe
        self.simplify_deduce = re.compile(r'\-[0-9]\]')
        self.trim_deduce = re.compile(r'[\[\]]')
        self.bsn_check = bsn_check
        self.date_check = date_check
        self.phone_check = phone_check
        self.url_check = url_check
        self.doi_check = doi_check
        self.pid_check = pid_check
        self.number_replace = number_replace
        self.clear_brackets = clear_brackets
        self.backend='joblib'
        self.data_index = data_index
        self.text_cols = text_cols

        if (custom_list is not None) and \
            (isinstance(custom_list, list)):
            if len(custom_list) > 0:
                self.custom_list = custom_list
            else:
                self.custom_list = False
        else:
            self.custom_list = False


        if kwargs==None:
            self.kwargs = dict()  
            self.kwargs['names'] = True             # Person names, including initials
            self.kwargs['locations'] = True         # Geographical locations
            self.kwargs['institutions'] = True      # Institutions
            self.kwargs['dates'] = True             # Dates
            self.kwargs['ages'] = True             # Ages
            self.kwargs['patient_numbers'] = False  # Patient numbers
            self.kwargs['phone_numbers'] = False    # Phone numbers
            self.kwargs['email_addresses'] = False  # E-mail addresses
            self.kwargs['urls'] = False,            # Urls and e-mail addresses
            self.kwargs['flatten'] = True           # Debug option 
        else: # the error handling should be done by deduce.
            assert (isinstance(kwargs, dict)), 'kwargs should be a dictionary'
            self.kwargs = kwargs

        if method == 'deduce':
            self.deducer = deduce.Deduce(config=kwargs)
        elif method == 'spacy':
            nlp = spacy.load("nl_core_news_sm")

    def _to_dataframe(self, X):
        '''
            Assume that X is a list of lists with text -> pd.DataFrame
        '''
        output = pd.DataFrame()
        if self.data_index is None:
            self.data_index = list(range(len(X[0])))
        if self.text_cols is None:
            self.text_cols = [f'column_{i}' for i in range(len(X))]
        for idx, c in enumerate(X):
            _X = np.array(c)
            temp = pd.DataFrame(
                         data=_X,
                         columns=[self.text_cols[idx]],
                         index=self.data_index
                         )
            output = pd.concat([output, temp], axis=1)
        return output                      

    def _deidentify(self, string):
        '''
        Deidentify a string using the specified method.
        '''
        if self.method == 'deduce':
            # DEDUCE by Menger et al.
            return self._deduce(string)
        elif self.method == 'Spacy':
            # SpaCy is NOT suitable for Dutch clinical text
            doc = nlp(string)
            replaced_tokens = []
            for token in doc:
                if token.ent_type_ == "PERSON":
                    replaced_tokens.append("[PERSOON]")
                else:
                    replaced_tokens.append(token.text)

            # Join tokens back into a single string
            return " ".join(replaced_tokens)
        else:
            raise NotImplementedError('Method not implemented.')

    def _bsn_check(self, txt):
        '''
            Checks if number in text is a valid BSN.
            Apply the following rule:
            - BSN must be 9 digits long
            - take the product sum of index with the digits
            - the first digit digit should be multiplied with -1
            - the final sum should be a multiple of 11

            Examples are 111222333 and 123456782
        '''
        if len(txt) not in [9, 10]:
            return False
        if (len(txt)==8) & (txt[0]!='0'):
            return False
        if len(txt)==8:
            txt = '0'+txt

        bsn_sum = 0
        for c,i in enumerate(range(len(txt), 0, -1),1):
            if i == 1:
                bsn_sum += int(txt[c-1]) * -1
            else:
                bsn_sum += int(txt[c-1]) * i
        if bsn_sum % 11 == 0:
            return True
        else:
            return False

    def _bsn_remove(self, txt):
        '''
            Remove BSN numbers from text:
                - first extract all number sequences of length 9 surrounded by non-numeric characters
                - then apply the function _bsn_check
                - if it passes the function as True, replace the number sequence with "<PID>"
        '''
        bsn_list = bsn_re.findall(txt)
        for bsn in bsn_list:
            if self._bsn_check(bsn):
                txt = txt.replace(bsn, '[BSN]')

    def _patient_id_remove(self, txt):
        for (pid,_,_) in patid_re.findall(txt):
            txt = txt.replace(pid, '[PATIENTNUMMER]')
        return txt
        
    def url_remove(self, txt):
        '''
            Replace URL with "URL"
        ''' 
        url_list = weblink_re.findall(txt)
        for date in url_list:
            txt = txt.replace(date, '[URL]')
        return txt

    def doi_remove(self, txt):
        '''
            Replace DOI with "DOI"
        ''' 
        doi_list = doi_re.findall(txt)
        for date in doi_list:
            txt = txt.replace(date, '[DOI]')
        return txt        


    def _date_remove(self, txt):
        '''
            Remove dates from text:
                - first extract all potential data sequences
                - then apply the function _date_check
                - if it passes the function as True, replace the number sequence with "<DATUM>"
        '''
        for date_re in date_res:
            date_list = date_re.findall(txt)
            for date in date_list:
                txt = txt.replace(date, '[DATUM]')
        return txt

    def _phone_check(self, txt):
        try:
            phonenumbers.parse(txt.replace(" ",""), "NL")
            return True
        except:
            return False

    def _phonenumber_remove(self, txt):
        # alternative: pip  install phonenumbers; try: phonenumbers.parse(txt,None)..
        for re_idx, phone_re in enumerate(phone_res):
            phone_list = phone_re.findall(txt)
            for phone in phone_list:               
                phone = phone[0].strip()                
                if self._phone_check(phone):
                    txt = txt.replace(phone, '[TELEFOONNUMMER]')
        return txt

    def _deduce(self, string):
        '''
        Deidentify a string using the Deduce method.
        '''
        if self.bsn_check:
            string = self._bsn_remove(string)
        
        if self.pid_check:
            string = self._patient_id_remove(string)

        if self.phone_check:
            string = self._phonenumber_remove(string)

        if self.date_check:
            string = self._date_remove(string)

        if self.custom_list is not False:
            for t in self.custom_list:
                string = t[0].sub(t[1], string)
        
        if self.clear_brackets:
            deid_text = self.deducer.deidentify(string).deidentified_text
            string = self.trim_deduce.sub('', self.simplify_deduce.sub(']', deid_text))
        else:
            deid_text = self.deducer.deidentify(string).deidentified_text
            string = self.simplify_deduce.sub(']', deid_text)

        if self.number_replace:
            # replace floating point numbers with <FLOAT>
            string = re.sub(r'\d+\.\d+', '[FLOAT]', string)
            # replace integers with <INT>
            string = re.sub(r'\d+', '[INT]', string)

        return string

    def _deid_series(self, series):
        #series = pd.Series({'text': series})
        return [self._deidentify(doc) for doc in series]
    
    def _par_deduce(self, txts):
        if self.n_jobs==-1:
            self.n_jobs = cpu_count()

        if self.backend=='multiprocessing':
            with get_context("spawn").Pool(self.n_jobs) as pool:
                results_list = pool.map(self._deduce,  txts, chunksize=100)
        elif self.backend=='joblib':
            results_list = Parallel(n_jobs=self.n_jobs, backend='loky')\
                           (delayed(self._deduce)(row) for row in txts)
        else:
            raise NotImplementedError('Backend not implemented.')
        return results_list

    def _data_check(self, X):
        err_str = 'X should be a pandas.Series, a pandas.DataFrame, a np.array or a list of strings'
        if isinstance(X, list):
            # check if all items are lists, in which all items are text
            # of if all items is text
            all_list = True
            all_text_1 = True
            all_text_2 = True

            for el in X:
                all_list &= isinstance(el, list)
                all_text_1 &= isinstance(el, str)
                if all_list:
                    for _el in el:
                        all_text_2 &= isinstance(_el, str)
            if all_list & all_text_2:
                return X
            elif all_text_1:
                return [X]
            else:
                raise ValueError(err_str)
        elif isinstance(X, pd.DataFrame):
            self.data_index = X.index
            assert all([dt==object for dt in X.dtypes]), \
                'X should be pd.DataFrame with only object dtype'
            self.text_cols = X.columns
            return X.T.values.tolist()
        elif isinstance(X, pd.Series):
            assert X.dtype==object, 'X should be pd.Series with object dtype'
            return [X.T.values.tolist()]
        elif isinstance(X, np.ndarray):
            assert X.dtype == np.dtype('object'), \
                'X should be np.array of strings'
            if len(X.shape)>1:
                return X.T.tolist()
            else:
                return [X.T.tolist()]

    def fit(self, X, y=None):
        '''
            First we assert that the input is a pandas.Series of type str, 
            a pandas.DataFrame containing only strings, or a list of strings.
        '''
        assert (isinstance(X, list) | 
                isinstance(X, pd.DataFrame) | 
                isinstance(X, pd.Series) |
                isinstance(X, np.ndarray)), "X should be lists of lists, \
                a list of text, pd.DataFrame or np.array"   

        _X = self._data_check(X)
        self.deid = []
        for _col in _X:
            if self.n_jobs==1:
                self.deid.append(self._deid_series(_col))
            else:
                self.deid.append(self._par_deduce(_col))
        return self

    def transform(self, X, y=None):
        if self.to_dataframe:
            return self._to_dataframe(self.deid)
        else:
            if len(self.deid)==1:
                return self.deid[0]
            else:
                return self.deid
            
def process_parquet_files(InputFolder: str, OutputFolder: str, deidentifier):
    '''
        Process all parquet files in a folder
    '''
    if not os.path.exists(OutputFolder):
        os.makedirs(OutputFolder)
    files = [f for f in os.listdir(InputFolder) if f.endswith('.parquet')]
    for f in files:
        data = pd.read_parquet(os.path.join(InputFolder, f))
        deidentifier.fit_transform(data)\
                    .to_parquet(os.path.join(OutputFolder, f.replace('.parquet', '_DEID.parquet')), 
                                index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deidentify text using Deduce')
    parser.add_argument('--input', type=str, help='input file')
    parser.add_argument('--input_folder', type=str, help='input folder')
    parser.add_argument('--output', type=str, help='output file')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--method', type=str, default='deduce', help='method to use')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
    parser.add_argument('--to_dataframe', action='store_true', help='output as dataframe')
    parser.add_argument('--bsn_check', action='store_true', help='check BSN numbers')
    parser.add_argument('--date_check', action='store_true', help='check dates')
    parser.add_argument('--phone_check', action='store_true', help='check phone numbers')
    parser.add_argument('--pid_check', action='store_true', help='check patient numbers')
    parser.add_argument('--backend', type=str, default='joblib', help='backend to use')
    parser.add_argument('--custom_list', type=str, nargs='+', help='custom list of regex replacements')
    parser.add_argument('--clear_brackets', action='store_true', help='clear brackets')
    parser.add_argument('--data_index', type=str, nargs='+', help='data index')
    parser.add_argument('--text_cols', type=str, nargs='+', help='text columns')
    args = parser.parse_args()

    # either the input or input_folder should be specified
    if (args.input is None and args.input_folder is None) or (args.input is not None or args.input_folder is not None) :
        raise ValueError('Either input or input_folder should be specified')

    # either the output or output_folder should be specified
    if args.input is not None and args.output is None:
        raise ValueError('Output should be specified if input is specified')

    if args.input_folder is not None and args.output_folder is None:
        raise ValueError('Output folder should be specified if input folder is specified')
    
    deid = Deidentify(method=args.method,
                      n_jobs=args.n_jobs,
                      to_dataframe=args.to_dataframe,
                      bsn_check=args.bsn_check,
                      date_check=args.date_check,
                      phone_check=args.phone_check,
                      pid_check=args.pid_check,
                      backend=args.backend,
                      custom_list=args.custom_list,
                      clear_brackets=args.clear_brackets,
                      data_index=args.data_index,
                      text_cols=args.text_cols
                      )

    if args.input is not None:
        if args.output is not None:
            if args.input.endswith('.csv'):
                data = pd.read_csv(args.input)
            elif args.input.endswith('.tsv'):
                data = pd.read_csv(args.input, sep='\t')
            elif args.input.endswith('.xlsx'):
                data = pd.read_excel(args.input)
            elif args.input.endswith('.parquet'):
                data = pd.read_parquet(args.input)
            else:
                raise ValueError('Input file should be csv, tsv or xlsx')
            deid.fit(data)
            deid.transform(data).to_csv(args.output, index=False)
        else:
            raise ValueError('Output file should be specified')
        
    if args.input_folder is not None:
        process_parquet_files(args.input_folder, args.output_folder, deid)
