import regex as re
import hashlib
import os
import sys
import argparse
import fuzzywuzzy
import shutil
from collections import defaultdict
from tqdm import tqdm
import benedict

class DeduplicaterHash:
    '''
        Deduplicate a list of strings/files using hash functions.
        
        Args: similarity_threshold (float): The threshold for similarity.
        Args: metric (str): The similarity metric to use.
        Args: folder (str): source folder for the text-data
        Args: store (bool): keep text in memory

        Returns: list: A list of strings with duplicates removed.
    '''
    def __init__(self, 
                threshold=0.8,
                metric='cosine', 
                in_folder=None, 
                in_memory=True, 
                out_folder=None,
                prefix='dedup_'):
        assert(threshold >= 0 and threshold <= 1), 'Similarity threshold must be between 0 and 1'
        assert(metric in ['cosine', 'jaro', 'jaro_winkler', 'levenshtein']), 'Invalid similarity metric'
        assert(in_folder is None or os.path.isdir(in_folder)), 'Invalid folder'
        assert(out_folder is None or os.path.isdir(out_folder)), 'Invalid folder'
        assert(isinstance(prefix)), 'Invalid prefix'
        assert(isinstance(in_memory, bool)), 'store should be a boolean'

        self.threshold = threshold
        self.metric = metric
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.in_memory = in_memory
        self.prefix = prefix
        self.files = self._get_files(in_folder) if in_folder is not None else None
        self.contents = defaultdict(str)
        self.remaining_files = []

    def deduplicate_hash_documents(self):
        '''
            Deduplicate a list of strings using hash functions.
        '''
        hashes = set()
        remaining_strings = []
        for k, blob in tqdm(self.contents.items()):
            docs = self._get_documents(k, blob)
            for doc in docs:
                hash = self._get_hash(doc)
                if hash not in hashes:
                    remaining_strings.append(doc)
                    hashes.add(hash)
        return len(hashes)

    def deduplicate_hash_files(self):
        '''
            Deduplicate a list of files using hash functions.            
        '''
        if self.files == None:
            raise ValueError('No files to deduplicate, please set a correct folder as attribute')

        hashes = set()        
        for root, filepath in tqdm(self.files.items()):
            hash, txt = self._get_file_hash(filepath)
            if hash not in hashes:
                self.remaining_files.append(filepath)                
                hashes.add(hash)
                if self.in_memory:
                    self.contents[root]=txt
        return self.remaining_files
    
    def write(self):
        re_split = re.compile(r'[\\\/]')
        for k,v in tqdm(self.contents.items()):
            outfile = os.path.join(self.out_folder, re_split.split(k)[-1], self.prefix)
            with open(outfile, 'w') as f:
                f.write(v)

    ##################################################
    ################### Helpers  #####################
    ##################################################
    def _get_documents(self, key, blob):
        '''
            split the blob based on the splits defined in the remove_list
        '''
        split = patterns_to_split[key]
        docs = []
        for split in splits:
            docs.extend(re.split(split, blob))
        return docs
        

    def _get_file_contents(self, filepath):
        '''
            Get the contents of a file
        '''
        with open(filepath, 'r') as f:
            return f.readlines()

    def _get_hash(self, string):
        '''
            Get the hash of a string
        '''
        return hashlib.sha256(string.encode('utf-8')).hexdigest()

    def _get_file_hash(self, filepath):
        '''
            Get the hash of a file
        '''
        with open(filepath, 'rb') as f:
            txt = f.read().strip()            
            if self.in_memory:
                return self._get_hash(txt).hexdigest(), None
            else:
                return self._get_hash(txt).hexdigest(), txt

    def _get_files(self, folder):
        '''
            Recursively get a list of files in a folder and it's subfolders
        '''
        import os
        files = defaultdict(list)
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                files[root].append(os.path.join(root, filename))
        return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deduplicate a list of strings using similarity metrics or hash functions.')
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help='The threshold for similarity')
    parser.add_argument('-if', '--infolder', type=str, default=None, help='The folder to deduplicate')
    parser.add_argument('-of', '--outfolder', type=str, default=None, help='Where to write the deduplicated files')
    parser.add_argument('-m', '--inmemory', type=bool, default=True, help='Store the text in-memory')
    parser.add_argument('-p', '--prefix', type=str, default='dedup_', help='Prefix for output files')
    args = parser.parse_args()

    # first deduplicate the entire files
    deduplicater = DeduplicaterHash(threshold = args.threshold, 
                                metric = args.metric, 
                                in_folder = args.infolder, 
                                out_folder = args.outfolder, 
                                in_memory = args.inmemory,
                                prefix = args.prefix)

    if args.folder is not None:
        original_file_count = len(deduplicater.files)
        files = deduplicater.deduplicate_hash_files()
        print(f'Original: {original_file_count} files \t Deduplicated :{len(files)} files')

    # then deduplicate the documents
    if args.store:
        True

    # write the files
    deduplicater.write()


