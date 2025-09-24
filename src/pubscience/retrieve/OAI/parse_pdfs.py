# read in pdf and extract text
# We want to ignore all the decorum and only extract the text (i.e. no page number etc.)
from __future__ import print_function


import pytesseract
#from PyPDF2 import PdfReader
from pypdf import PdfReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import fitz
from PIL import Image
import ftfy

import matplotlib.pyplot as plt

from io import StringIO
import re
import os

from tqdm import tqdm
from typing import List, Tuple, Literal

import signal

import sys
import threading
from time import sleep
import _thread as thread

"""
Page segmentation modes (PSM):
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
"""

# We want to ignore 
# Title pages
# Table of contents
# Reference lists 
# Acknowledgements
# List of abbreviations
# List of figures

# Remove
# all empty lines or lines that only have numbers
# all pages with less than K words

# We want to extract
# body, summary(english), summary(dutch)

# TODO: make env settable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

re_numbers_at_start_of_sentence_plus = re.compile(r'(\d+\n\d*)')  # Matches numbers at the start of a sentence
re_numbers_at_start_of_sentence = re.compile(r'(\d+)\n')  # Matches numbers at the start of a sentence

re_numbers_at_start_of_string = re.compile(r'^(\d+)')  # Matches numbers at the start of a sentence
re_lines_with_only_numbers = re.compile(r'^\s*\d+\s*$', re.MULTILINE)  # Matches lines that contain only numbers
re_multiple_newlines = re.compile(r'\n+')
re_empty_lines = re.compile(r'\n\s*\n')
re_empty_lines_start = re.compile(r'^\s*\n')
re_empty_lines_end = re.compile(r'\n\s*$')
re_multiple_spaces = re.compile(r'\s+')

re_section_num = re.compile(r'^\d+\n(\d*)') 



#############################################################################

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    #raise SystemExit 
    raise ValueError(f'{fn_name} took too long')

def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(240)
def pdf_to_text(path, 
                backends: List[str]=['pypdf', 'fitz', 'pdfminer'],
                ocr_backends:List[str]=['pytesseract','dotocr', 'paddleocr']) -> Tuple[List[str], str]:
    '''Extract text from pdf documents
        Source: https://towardsdatascience.com/pdf-preprocessing-with-python-19829752af9f
    '''

    def pdfminer(_path):
        manager = PDFResourceManager()
        retstr = StringIO()
        layout = LAParams(all_texts=False, detect_vertical=True)
        device = TextConverter(manager, retstr, laparams=layout)
        interpreter = PDFPageInterpreter(manager, device)
        with open(_path, 'rb') as filepath:
            for page in PDFPage.get_pages(filepath, check_extractable=True):
                interpreter.process_page(page)
        text = retstr.getvalue()
        device.close()
        retstr.close()
        return text.split('\n')

    def pypdfer(_path):
        bytes_stream = open(_path, 'rb')
        reader = PdfReader(bytes_stream, strict=True)
        return [p.extract_text(0) for p in reader.pages]
    
    def check_pdf_scan(_path):
        bytes_stream = open(_path, 'rb')
        reader = PdfReader(bytes_stream, strict=True)    
        meta = reader.metadata
        producer = str(meta.producer).lower()

        if any([s in producer for s in ['scanner', 'scan', 'image', 
                                        'finereader', 'tesseract']]):
            return True, producer
        
        for page_num, p in enumerate(reader.pages):
            text = p.extract_text()
            scan_count = 0
            if text is None or len(text.strip())==0:
                scan_count += 1
            if (scan_count / (page_num+1) > 0.75) and (page_num>15):
                return True, producer
            elif (scan_count / (page_num+1) < 0.25) and (page_num>15):
                return False, producer
        return False, producer

    
    def pdf2imagelist(_path, backend: Literal['fitz', 'pdf2image']='fitz') -> List[Image.Image]:
        '''Convert each page of a PDF into a list of PIL Image objects.
        '''
        if backend == 'fitz':
            pdf_document = fitz.open(_path)
            images = []
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        elif backend == 'pdf2image':
            from pdf2image import convert_from_path
            images = convert_from_path(_path)
        return images
    
    pages = []
    error = ""

    scanned, pdf_producer = check_pdf_scan(path)
    if scanned:  
        print(f"Document {os.path.basename(path)} is detected as a scanned document.")    
        ###########################################
        # OCR for scanned documents
        ###########################################
        if not any([m in ocr_backends for m in ['pytesseract', 'paddleocr', 'dotocr']]):
            return None, "It is a scanned document, but no OCR backend is selected", scanned, pdf_producer
        
        image_list = pdf2imagelist(path, backend='fitz')

        # dotocr
        # TODO: implement

        # paddleocr
        # TODO: implement

        # LAST RESORT
        if 'pytesseract' in ocr_backends:
            try:
                psm = 1  # Fully automatic page segmentation, but no OSD. (Default)
                config_str = f'--psm {psm}' 
                return [pytesseract.image_to_string(img, config=config_str) 
                         for img in image_list], None, scanned, pdf_producer
            except Exception as e_4:
                error = error + f"\n PyTesseract failed: {e_4}" 
            
        return pages, error, scanned, pdf_producer
    else:
        ###########################################
        # Normal text extraction
        ###############################
        if 'pypdf' in backends:
            try:
                pages, error = pypdfer(path, ), None
            except Exception as e_1:
                error = error + f"\n PyPDF failed: {e_1}"

        if pages and len(pages)>0:
            return pages, error, scanned, pdf_producer
        ###############################
        if 'fitz' in backends:
            try:
                pdf_file = fitz.open(path)
                pages = []
                for _p in pdf_file:
                    pages.append(_p.get_text())
                return pages, error, scanned
            except Exception as e_2:
                error = error + f"\n PyMuPDF failed: {e_2}"

        if pages and len(pages)>0:
            return pages, error, scanned, pdf_producer
        ###############################
        
        if 'pdfminer' in backends:
            try:
                return pdfminer(path), error, scanned, pdf_producer
            except Exception as e_3:
                error = error + f"pdfminer failed: {e_3}"
    
    return pages, error, scanned, pdf_producer


def extract_summary(Texts: List[str], max_scount: int=20) -> str:
    samenvatting = []
    capture = False
    scount = 0

    init_section_num =""
    for page in Texts:
        section_num = re_section_num.findall(page)        
        page = re_numbers_at_start_of_sentence_plus.sub(r'', page)  # Remove numbers at the start of a sentence
        if any((x in page.lower()[:60]) | (x in page.lower()[-60:]) for x in ['s amenvatting', 'samenvatting', 
                                                'nederlandse samenvatting', 
                                                'samenvatting in het nederlands',
                                                's amenvatting in het nederlands',     
                                                    'd utch summary',
                                                    'dutch summary',
                                                    'n ederlandse samenvatting']):
            capture = True
            init_section_num =  section_num
            scount += 1
        elif any((x in page.lower()[:60]) | (x in page.lower()[-60:])for x in ['d ankwoord', 
                                              'na woord',
                                              'a cknowledgment',
                                              'c ontents', 
                                              't able of contents', 
                                              'l ist of figures', 
                                              'l ist of abbreviations', 
                                              'a cknowledgements', 
                                              'r eferences',
                                              'dankwoord',
                                              'nawoord', 
                                              'acknowledgment',
                                              'contents', 
                                              'table of contents', 
                                              'list of figures', 
                                              'list of abbreviations', 
                                              'acknowledgements', 
                                              'references',
                                              's ummary', 
                                              'summary',
                                              'english summary']):
            capture = False
        elif section_num != init_section_num:
            capture = False
        
        if capture:
            scount += 1
            samenvatting.append(page)
        if scount >= max_scount:
            break
    summary = []
    capture = False
    scount = 0
    for page in Texts:
        page = re_numbers_at_start_of_sentence_plus.sub(r'', page)
        if any((x in page.lower()[:60]) | (x in page.lower()[-60:]) for x in ['s ummary', 'summary', 'english summary', 'summery']):
            capture = True
            init_section_num =  section_num
            scount += 1
        elif any((x in page.lower()[:60]) | (x in page.lower()[-60:]) for x in ['d ankwoord', 
                                              'na woord',
                                              'a cknowledgment',
                                              'c ontents', 
                                              't able of contents', 
                                              'l ist of figures', 
                                              'l ist of abbreviations', 
                                              'a cknowledgements', 
                                              'r eferences',
                                              'dankwoord',
                                              'nawoord', 
                                              'acknowledgment',
                                              'contents', 
                                              'table of contents', 
                                              'list of figures', 
                                              'list of abbreviations', 
                                              'acknowledgements', 
                                              'references',
                                              's amenvatting', 'samenvatting', 
                                              'nederlandse samenvatting', 
                                              'd utch summary',
                                              'dutch summary',
                                              'n ederlandse samenvatting']):
            capture = False
        elif section_num != init_section_num:
            capture = False
            
        if capture:
            scount += 1
            summary.append(page)
        if scount >= max_scount:
            break
    # remove numbers of the start of sentences
    # remove multiple newlines
    # remove empty lines
    # remove multiple spaces
    # remove lines with only numbers

    summary = [re_numbers_at_start_of_sentence.sub(r'', s) for s in summary]
    summary = [re_empty_lines_start.sub(r'', s) for s in summary]
    summary = [re_empty_lines_end.sub(r'', s) for s in summary]
    summary = [re_empty_lines.sub(r'', s) for s in summary]
    summary = [re_lines_with_only_numbers.sub(r'', s) for s in summary]
    summary = [re_numbers_at_start_of_string.sub(r'\n', s) for s in summary]

    samenvatting = [re_numbers_at_start_of_sentence.sub(r'', s) for s in samenvatting]
    samenvatting = [re_empty_lines_start.sub(r'', s) for s in samenvatting]
    samenvatting = [re_empty_lines_end.sub(r'', s) for s in samenvatting]
    samenvatting = [re_empty_lines.sub(r'', s) for s in samenvatting]
    samenvatting = [re_lines_with_only_numbers.sub(r'', s) for s in samenvatting]
    samenvatting = [re_numbers_at_start_of_string.sub(r'\n', s) for s in samenvatting]
    return '\n'.join(summary), '\n'.join(samenvatting)


def text_extractor(Text: List[str], min_words: int=100) -> List[str]:
    Text = [ftfy.fix_text(t) for t in Text]

    Text = [t for t in Text if len(t.split())>50]
    Text = [re_numbers_at_start_of_sentence.sub('', t) for t in Text]
    Text = [re_numbers_at_start_of_string.sub('', t) for t in Text]
    Text = [re_lines_with_only_numbers.sub('', t) for t in Text]
    Text = [re_multiple_newlines.sub('\n', t) for t in Text]
    Text = [re_empty_lines.sub('\n', t) for t in Text]
    Text = [re_empty_lines_start.sub('', t) for t in Text]
    Text = [re_empty_lines_end.sub('', t) for t in Text]
    Text = [re_multiple_spaces.sub(' ', t) for t in Text]
    Text = [t for t in Text if len(t.split())>50]

    # ignore references
    reference_phrases = ['references', 'literature', 'bibliography', 'referenties', 'literatuurlijst']
    Text = [t for t in Text if not any(reference_phrase in t.lower() for reference_phrase in reference_phrases)]
    # ignore lime that start with numbers after the linebreak or have "doi:10" in them
    # scan the page for lines that start with numbers after a linebreak
    _TEXT = []
    for page in Text:
        lines = page.split('\n')
        __page= []
        for line in lines:
            if (not re.search(r'^\d+', line)) and ('doi:10' not in line.lower()):
                __page.append(line)
        _TEXT.append('\n'.join(__page))
    Text = _TEXT

    # ignore list of figures
    figure_phrases = ['list of figures', 'lijst van figuren']
    Text = [t for t in Text if not any(figure_phrase in t.lower() for figure_phrase in figure_phrases)]

    # ignore list of abbreviations
    abbreviation_phrases = ['list of abbreviations', 'lijst van afkortingen']
    Text = [t for t in Text if not any(abbreviation_phrase in t.lower() for abbreviation_phrase in abbreviation_phrases)]

    # ignore copyright page
    copyright_phrases = ['all rights reserved', 'no part of this publication may be reproduced', 'copyright', 'uitgeverij']
    Text = [t for t in Text if not any(copyright_phrase in t.lower() for copyright_phrase in copyright_phrases)]

    phd_phrases = ['volgens besluit van het college voor promoties', 'de graad van doctor aan']
    Text = [t for t in Text if not any(phd_phrase in t.lower() for phd_phrase in phd_phrases)]

    # ignore table of contents
    toc_phrases = ['inhoudsopgave', 'table of contents']
    Text = [t for t in Text if not any(toc_phrase in t.lower() for toc_phrase in toc_phrases)]
    # ignore if multiple sentences in a page start with "chapter \d"
    chapter_phrases = ['chapter ', 'hoofdstuk ']
    Text = [t for t in Text if sum(t.lower().count(chapter_phrase) for chapter_phrase in chapter_phrases)<2]

    # ignore acknowledgements
    acknowledgement_phrases = ['acknowledgements', 'acknowledgements', 'dankwoord', 'dankbetuiging']
    Text = [t for t in Text if not any(acknowledgement_phrase in t.lower() for acknowledgement_phrase in acknowledgement_phrases)]

    # ignore list of publications
    publication_phrases = ['list of publications', 'lijst van publicaties', 'bibliography', 'bibliografie']
    Text = [t for t in Text if not any(publication_phrase in t.lower() for publication_phrase in publication_phrases)]


    Text = [t for t in Text if len(t.split())>25]

    TextNumWords = [len(t.split()) for t in Text]
    

    return Text, TextNumWords