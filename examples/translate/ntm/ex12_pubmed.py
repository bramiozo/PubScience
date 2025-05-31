#!/usr/bin/env python3
"""
PubMed XML Processing and Translation Script

This script downloads, processes, and translates PubMed XML files from NCBI FTP servers.
It extracts text content from XML files, cleans it, and translates it using neural machine translation.

Usage:
    python ex12_pubmed.py
"""

import os
import sys
import json
import tarfile
import ftplib
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import tempfile
import shutil
import logging
import argparse
from typing import List, Dict, Any, Optional, Literal
import ftfy

# Import the translation module
from pubscience.translate import ntm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PubMedProcessor:
    """Process PubMed XML files from NCBI FTP servers."""
    
    FTP_SERVERS = [
        'ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/',
        'ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/',
        'ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/xml/'
    ]
    
    def __init__(self, 
                 model_name: Literal["facebook/nllb-200-3.3B",
                                   "facebook/nllb-200-distilled-600M",
                                   "facebook/m2m100_418M",
                                   "google/madlad400-3b-mt",
                                   "facebook/mbart-large-50-many-to-many-mmt",
                                   "vvn/en-to-dutch-marianmt"] = 'vvn/en-to-dutch-marianmt',
                 max_length: int = 496,
                 target_lang: str = 'nld_Latn',
                 output_dir: str = './output', 
                 temp_dir: str = './temp',
                 batch_size: int = 16):
        """Initialize the processor."""
        self.model_name = model_name
        self.max_length = max_length
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize translator
        self.translator = ntm.TranslationNTM(
            model_name=model_name,
            multilingual=False,
            max_length=max_length,
            use_gpu=True,
            target_lang=target_lang
        )
        
    def get_tar_gz_files(self, server_path: str) -> List[str]:
        """Get list of .tar.gz files from FTP server."""
        try:
            ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
            ftp.login()
            ftp.cwd(server_path.replace('ftp.ncbi.nlm.nih.gov/', ''))
            
            files = []
            ftp.retrlines('LIST', lambda x: files.append(x.split()[-1]))
            tar_gz_files = [f for f in files if f.endswith('.tar.gz')]
            
            ftp.quit()
            logger.info(f"Found {len(tar_gz_files)} .tar.gz files on {server_path}")
            return tar_gz_files
            
        except Exception as e:
            logger.error(f"Error accessing FTP server {server_path}: {e}")
            return []
    
    def download_file(self, server_path: str, filename: str) -> Optional[Path]:
        """Download a file from FTP server."""
        try:
            ftp_url = f"ftp://{server_path}/{filename}"
            local_path = self.temp_dir / filename
            
            logger.info(f"Downloading {filename}...")
            urllib.request.urlretrieve(ftp_url, local_path)
            logger.info(f"Downloaded {filename} successfully")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
    
    def extract_tar_gz(self, tar_path: Path) -> Optional[Path]:
        """Extract .tar.gz file and return extraction directory."""
        try:
            extract_dir = self.temp_dir / tar_path.stem.replace('.tar', '')
            extract_dir.mkdir(exist_ok=True)
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            logger.info(f"Extracted {tar_path.name} to {extract_dir}")
            return extract_dir
            
        except Exception as e:
            logger.error(f"Error extracting {tar_path}: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean text using ftfy and remove unwanted XML tags."""
        if not text:
            return ""
        
        # Use ftfy to fix text encoding issues
        text = ftfy.fix_text(text)
        
        # Remove xref tags and their content
        text = re.sub(r'<xref[^>]*>.*?</xref>', '', text, flags=re.DOTALL)
        
        # Remove label tags and their content
        text = re.sub(r'<label[^>]*>.*?</label>', '', text, flags=re.DOTALL)
        
        # Remove mml tags and their content (MathML)
        text = re.sub(r'<mml[^>]*>.*?</mml[^>]*>', '', text, flags=re.DOTALL)
        
        # Remove any remaining XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_text_from_xml(self, xml_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata and text content from XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract filename
            filename = xml_path.name
            
            # Extract journal title
            journal_title = ""
            journal_elements = root.findall(".//journal-title")
            if journal_elements:
                journal_title = journal_elements[0].text or ""
            
            # Extract text from all <p> elements
            text_parts = []
            for p_element in root.findall(".//p"):
                if p_element.text:
                    text_parts.append(p_element.text)
                # Also get text from nested elements
                for child in p_element.iter():
                    if child.text and child.tag != 'xref' and child.tag != 'label' and not child.tag.startswith('mml'):
                        text_parts.append(child.text)
                    if child.tail:
                        text_parts.append(child.tail)
            
            # Combine and clean text
            raw_text = ' '.join(text_parts)
            cleaned_text = self.clean_text(raw_text)
            
            if not cleaned_text.strip():
                return None
            
            return {
                'meta_filename': filename,
                'meta_journal': journal_title.strip(),
                'text': cleaned_text
            }
            
        except Exception as e:
            logger.error(f"Error processing XML file {xml_path}: {e}")
            return None
    
    def remove_repetitions(self, text: str) -> str:
        """Remove spurious repetitive phrases from translated text."""
        if not text:
            return text
        
        # Split into sentences
        sentences = text.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for repetitive patterns (3+ consecutive repeated phrases)
            words = sentence.split()
            if len(words) < 6:  # Short sentences are unlikely to have meaningful repetitions
                cleaned_sentences.append(sentence)
                continue
            
            # Look for repeated sequences of 2-5 words
            cleaned = True
            for seq_len in range(2, 6):
                for i in range(len(words) - seq_len * 2):
                    sequence = words[i:i+seq_len]
                    # Check if this sequence is repeated immediately after
                    next_sequence = words[i+seq_len:i+seq_len*2] if i+seq_len*2 <= len(words) else []
                    if sequence == next_sequence:
                        # Found repetition, remove it
                        words = words[:i+seq_len] + words[i+seq_len*2:]
                        cleaned = False
                        break
                if not cleaned:
                    break
            
            cleaned_sentences.append(' '.join(words))
        
        return '. '.join(cleaned_sentences)
    
    def process_xml_files(self, extract_dir: Path, output_file: Path):
        """Process all XML files in the extracted directory."""
        xml_files = list(extract_dir.rglob('*.xml'))
        logger.info(f"Found {len(xml_files)} XML files to process")
        
        processed_count = 0
        batch = []
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for xml_file in xml_files:
                try:
                    # Extract text and metadata
                    data = self.extract_text_from_xml(xml_file)
                    if not data:
                        continue
                    
                    batch.append(data)
                    
                    # Process batch when it reaches batch_size
                    if len(batch) >= self.batch_size:
                        self.translate_and_write_batch(batch, f)
                        processed_count += len(batch)
                        batch = []
                        logger.info(f"Processed {processed_count} files...")
                
                except Exception as e:
                    logger.error(f"Error processing {xml_file}: {e}")
                    continue
            
            # Process remaining batch
            if batch:
                self.translate_and_write_batch(batch, f)
                processed_count += len(batch)
        
        logger.info(f"Processed {processed_count} XML files total")
    
    def translate_and_write_batch(self, batch: List[Dict[str, Any]], file_handle):
        """Translate a batch of texts and write to output file."""
        try:
            # Extract texts for translation
            texts = [item['text'] for item in batch]
            
            # Translate batch
            translated_texts = self.translator.translate_long_batch(texts, batch_size=self.batch_size)
            
            # Write results
            for i, translated_text in enumerate(translated_texts):
                # Remove repetitions from translated text
                cleaned_translation = self.remove_repetitions(translated_text)
                
                # Create output record
                output_record = {
                    'meta_filename': batch[i]['meta_filename'],
                    'meta_journal': batch[i]['meta_journal'],
                    'text': batch[i]['text'],
                    'translated_text': cleaned_translation
                }
                
                # Write to JSONL file
                file_handle.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Error translating batch: {e}")
            # Fallback: write original texts without translation
            for item in batch:
                output_record = {
                    'meta_filename': item['meta_filename'],
                    'meta_journal': item['meta_journal'],
                    'text': item['text'],
                    'translated_text': item['text']  # Fallback to original text
                }
                file_handle.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    
    def process_server(self, server_path: str):
        """Process all tar.gz files from a single FTP server."""
        logger.info(f"Processing server: {server_path}")
        
        # Get list of tar.gz files
        tar_gz_files = self.get_tar_gz_files(server_path)
        
        for tar_gz_file in tar_gz_files:
            try:
                logger.info(f"Processing {tar_gz_file}...")
                
                # Download tar.gz file
                downloaded_path = self.download_file(server_path, tar_gz_file)
                if not downloaded_path:
                    continue
                
                # Extract tar.gz file
                extract_dir = self.extract_tar_gz(downloaded_path)
                if not extract_dir:
                    # Clean up downloaded file
                    downloaded_path.unlink(missing_ok=True)
                    continue
                
                # Process XML files
                output_filename = f"output_{tar_gz_file.replace('.tar.gz', '')}.jsonl"
                output_path = self.output_dir / output_filename
                
                self.process_xml_files(extract_dir, output_path)
                
                # Clean up extracted directory and downloaded file
                shutil.rmtree(extract_dir, ignore_errors=True)
                downloaded_path.unlink(missing_ok=True)
                
                logger.info(f"Completed processing {tar_gz_file}")
                
            except Exception as e:
                logger.error(f"Error processing {tar_gz_file}: {e}")
                continue
    
    def run(self):
        """Run the complete processing pipeline."""
        logger.info("Starting PubMed XML processing and translation...")
        
        try:
            # Process each FTP server
            for server_path in self.FTP_SERVERS:
                self.process_server(server_path)
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
        finally:
            # Clean up temp directory
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PubMed XML Processing and Translation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ex12_pubmed.py
  python ex12_pubmed.py --model-name facebook/nllb-200-distilled-600M --max-length 1024
  python ex12_pubmed.py --target-lang fra_Latn --batch-size 8 --output-dir ./french_output
        """
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='vvn/en-to-dutch-marianmt',
        choices=[
            "facebook/nllb-200-3.3B",
            "facebook/nllb-200-distilled-600M",
            "facebook/m2m100_418M",
            "google/madlad400-3b-mt",
            "facebook/mbart-large-50-many-to-many-mmt",
            "vvn/en-to-dutch-marianmt"
        ],
        help='Translation model to use (default: vvn/en-to-dutch-marianmt)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=496,
        help='Maximum sequence length for translation (default: 496)'
    )
    
    parser.add_argument(
        '--target-lang',
        type=str,
        default='nld_Latn',
        help='Target language code (default: nld_Latn for Dutch)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for JSONL files (default: ./output)'
    )
    
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='./temp',
        help='Temporary directory for downloads and extraction (default: ./temp)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for translation processing (default: 16)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Log the configuration
    logger.info(f"Starting PubMed processor with configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Target language: {args.target_lang}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Temp directory: {args.temp_dir}")
    
    processor = PubMedProcessor(
        model_name=args.model_name,
        max_length=args.max_length,
        target_lang=args.target_lang,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        batch_size=args.batch_size
    )
    processor.run()


if __name__ == "__main__":
    main()