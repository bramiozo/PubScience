'''
Parser for pdf's.

Base technology
- OCR libraries
- vision-text models: ColPali, QwenPali using the transformers library

Input options
- location of pdf's
- selection_filter
'''

from typing import Literal

class OCR():
    def __init__(self, ocr_model: Literal['mupdf2', 'tesseract']):


        pass


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class VisionText():
    def __init__():
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

def parser(location: str | None):
    pass
