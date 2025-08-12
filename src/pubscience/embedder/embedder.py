'''Main module for extracting aggregated sentence/document embeddings'''

import os, sys, argparse, logging, json, pickle, time, re
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import KeyedVectors
from gensim.similarities import fastss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

import torch
import spacy

from spacy.language import Language
from typing import List, Dict, Tuple, Union, Optional, Any, Callable

from umap import UMAP
from tqdm import tqdm

# add logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

'''
V.0.1 Supports:
models
* word2vec/fasttext 

aggregation:
* mean, max, median

TO ADD:
SBERT
Universal Sentence Encoder
transformers
aggregation; doc2vec

aggregation: SE3, https://github.com/BinWang28/Sentence-Embedding-S3E
aggregation: SIF, https://github.com/smujjiga/SIF

aggregation; 
    weights based on dictionaries wrt PoS/dependencies, TF-IDF, cluster-attribution, LDA/LSA attributions (ETM/FuzzyTM) etc.
    weights based on negations of terms, combine with PoS-tagger
spacy pipes; exploit things like lemma, POS, dependency, sentence splitting, etc.
add: Cui2vec/Snomed2Vec --> assumes list of lists with CUIS or list of list of tuples with (polarity, CUI)
add: Wikipedia2Vec
add: Sense2Vec -> 
'''

'''
example

from utils.embedder import DocEmbedder

doc_embedder = DocEmbedder(method='word2vec', 
                           aggregation='mean',
                           embedding_path='path/path.bin')	

embedded_docs = doc_embedder.fit_transform(X)
'''

'''
class EmbeddingConstrastor(BaseEstimator, TransformerMixin):
    class to enable contrastive embedding, taking existing embeddings as input, 
    using contrastive learning to maximize constrast between clusters, and allows for the 
    input of labels.
'''


class SpacyEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model: Union[Language, str]='nl_core_news_lg', emb_dim=300, n_jobs=-1, verbose=False):
        self.model = model
        self.n_jobs = n_jobs
        self.nlp = None
        self.emb_dim = emb_dim
        self.verbose = verbose

    def _arrayify(self, Embs):
        if self.verbose:
            logger.info(f"Embeddings shape: {np.array(Embs).shape}")
        text_embeddings = pd.DataFrame(data=np.array(Embs), 
                                       columns=['dim_'+str(c) 
                                       for c in range(self.emb_dim)])
        
        text_embeddings.index = self.index
        return text_embeddings
    
    def _indexify(self, X: Union[List[str], pd.DataFrame]):
        self.index = X.index if isinstance(X, pd.DataFrame) else range(len(X))

    def fit(self, X: Union[List[str], pd.DataFrame], y=None):
        if type(self.model) == str:
            self.nlp = spacy.load(self.model, disable=['parser', 'ner', 'lemmatizer'])
        else:
            self.nlp = self.model

        self.emb_dim = self.nlp.vocab.vectors_length
        return self

    def _tqdm(self, iterable):
        if self.verbose:
            return tqdm(iterable)
        else:
            return iterable

    def transform(self, X: Union[List[str], pd.DataFrame], y=None) -> pd.DataFrame:
        docs = self.nlp.pipe(X, n_process=self.n_jobs)
        mean_embeddings = []
        for doc in self._tqdm(docs):
            try:
                mean_embeddings.append(doc.vector)
            except Exception as e:
                # TODO: add logger
                if self.verbose:
                    logger.info(f"Mean embedding extraction failed, taking the mean: {e}")
                mean_embeddings.append(np.mean(mean_embeddings, axis=0))

        self._indexify(X)
        return self._arrayify(mean_embeddings)

    def fit_transform(self, X: Union[List[str], pd.DataFrame], y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

class SBERTEmbedder(BaseEstimator, TransformerMixin):
    '''
        Class to embed sentences using Sentence-BERT following the Sklearn api
    '''
    
    def __init__(self, 
                model: str='distiluse-base-multilingual-cased-v1', 
                normalise_embeddings=False,
                verbose=False):
        self.verbose = verbose
        self.normalise_embeddings = normalise_embeddings

        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        else:
            # TODO: check type!
            self.model = model

        self.emb_dim = self.model.get_sentence_embedding_dimension()

    def _arrayify(self, Embs: np.ndarray) -> pd.DataFrame:
        if self.verbose:
            logger.info(f"Embeddings shape: {np.array(Embs).shape}")
        text_embeddings = pd.DataFrame(data=Embs, 
                                       columns=['dim_'+str(c) 
                                       for c in range(self.emb_dim)])
        
        text_embeddings.index = self.index
        return text_embeddings
    
    def _indexify(self, X: Union[List[str], pd.DataFrame]) -> None:
        self.index = X.index if isinstance(X, pd.DataFrame) else range(len(X))
    
    def fit(self, X: Union[List[str], pd.DataFrame], y=None):
        return self

    def transform(self, X: Union[List[str], pd.DataFrame], y=None) -> pd.DataFrame:
        if self.verbose:
            logger.info(f"Embedding {len(X)} sentences")
            
        embeddings = self.model.encode(X, 
                                       show_progress_bar=self.verbose, 
                                       convert_to_numpy=True,
                                       batch_size=32,
                                       normalize_embeddings=self.normalise_embeddings)

        self._indexify(X)
        return self._arrayify(embeddings)

class RoBERTaEmbedder(BaseEstimator, TransformerMixin):
    '''
        Class to embed sentences using BERT following the Sklearn api
        TODO: add batching
        TODO: add alternative embedding aggregation methods
        TODO: add check for token_length
        TODO: add check for hiddenstate_length
        TODO: persist device throughout class
    '''
    
    def __init__(self, 
                model: str='UMCU/MedRoBERTa.nl_NegationDetection', 
                tokenizer: str='UMCU/MedRoBERTa.nl_NegationDetection',
                token_length=512,
                hiddenstate_length=768,
                device='cpu',
                batch_size=1,
                verbose=False):
        self.verbose = verbose
        self.token_length = token_length
        self.hiddenstate_length = hiddenstate_length
        self.batch_size = batch_size

        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
        else:
            self.model = model


        if self.verbose==True:
            logger.info(f"Using {model} to embed sentences")
            logger.info(f"Model info:\n {self.model.eval()}")

        if (device!='cpu') | (device is None):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
        self.device = device        

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def _arrayify(self, Embs: np.ndarray) -> pd.DataFrame:
        if self.verbose:
            logger.info(f"Embeddings shape: {np.array(Embs).shape}")
        text_embeddings = pd.DataFrame(data=Embs, 
                                       columns=['dim_'+str(c) 
                                       for c in range(self.emb_dim)])
        
        text_embeddings.index = self.index
        return text_embeddings

    def _tqdm(self, iterable):
        if self.verbose:
            return tqdm(iterable, 
                        total=len(iterable)//self.batch_size, 
                        desc=f'Going through the data with a batchsize of {self.batch_size}')
        else:
            return iterable

    def _indexify(self, X: Union[List[str], pd.DataFrame]) -> None:
        self.index = X.index if isinstance(X, pd.DataFrame) else range(len(X))
    
    def _chunkify(self, X: Union[List[str], pd.DataFrame], stepsize: int)-> List[List[str]]:
        num_steps = len(X) // stepsize 
        last_range = len(X) % stepsize
        chunks = []
        for i in range(num_steps):
            index = i * stepsize
            if index >= len(X):
                break
            chunks.append(list(X[index:index + stepsize]))
        if last_range > 0:
            chunks.append(list(X[-last_range:]))
        return chunks

    def fit(self, X: Union[List[str], pd.DataFrame], y=None):
        return self
    
    def transform(self, X: Union[List[str], pd.DataFrame], y=None) -> pd.DataFrame:
        if self.batch_size==1:
            with torch.no_grad():
                mean_hidden_states = []
                for doc in self._tqdm(X):
                    inputs = self.tokenizer.encode(doc, return_tensors='pt')
                    inputs = inputs.to(self.device)

                    # if the input is longer than self.token_length tokens the model should convolve over the input
                    # and take the mean of the last hidden state
                    LEN = inputs.shape[1]
                    if LEN < self.token_length:
                        output = self.model(input_ids=inputs)
                        mean_vector = torch.mean(output.last_hidden_state, dim=1).squeeze().detach().numpy()
                    else:            
                        mean_vector = np.zeros(self.hiddenstate_length)
                        for num, i in enumerate(range(0, LEN, self.token_length)):
                            output = self.model(input_ids=inputs[:,i:min([LEN,i+self.token_length])])
                            mean_vector += torch.mean(output.last_hidden_state, dim=1).squeeze().detach().numpy()
                        mean_vector /= num
                    mean_hidden_states.append(mean_vector)
        else:            
            chunks = self._chunkify(X, self.batch_size)
            assert sum([len(chunk) for chunk in chunks])==len(X), f"Something went wrong with the chunking: {sum([len(chunk) for chunk in chunks])} versus {len(X)}"
            with torch.no_grad():
                mean_hidden_states = []
                for chunk in self._tqdm(chunks):
                    inputs = self.tokenizer(chunk, return_tensors='pt', padding=True, max_length=self.token_length, truncation=True)
                    inputs = inputs.to(self.device)
                    output = self.model(**inputs)
                    mean_vector = torch.mean(output.last_hidden_state, dim=1).squeeze().detach().numpy()
                    mean_hidden_states.extend(list(mean_vector))

        self._indexify(X)
        return pd.DataFrame(data=np.array(mean_hidden_states),
                            columns=['dim_'+str(c) for c in range(self.hiddenstate_length)],
                            index=self.index)

class BERTEmbedder(BaseEstimator, TransformerMixin):
    '''
        Class to embed sentences using BERT following the Sklearn api
        TODO: add batching
        TODO: add alternative embedding aggregation methods
        TODO: add check for token_length
        TODO: add check for hiddenstate_length
        TODO: persist device throughout class
    '''
    
    def __init__(self, 
                model: str='GroNLP/bert-base-dutch-cased', 
                tokenizer: str='GroNLP/bert-base-dutch-cased',
                token_length=512,
                hiddenstate_length=768,
                NSP=True,
                device='cpu',
                verbose=False):
        self.verbose = verbose
        self.token_length = token_length
        self.hiddenstate_length = hiddenstate_length
        self.NSP = NSP

        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
        else:
            self.model = model

        if self.verbose==True:
            logger.info(f"Using {model} to embed sentences")
            logger.info(f"Model info:\n {self.model.eval()}")
        
        if (device!='cpu') | (device is None):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def _arrayify(self, Embs: np.ndarray) -> pd.DataFrame:
        if self.verbose:
            logger.info(f"Embeddings shape: {np.array(Embs).shape}")
        text_embeddings = pd.DataFrame(data=Embs, 
                                       columns=['dim_'+str(c) 
                                       for c in range(self.emb_dim)])
        
        text_embeddings.index = self.index
        return text_embeddings

    def _tqdm(self, iterable):
        if self.verbose:
            return tqdm(iterable)
        else:
            return iterable

    def _indexify(self, X: Union[List[str], pd.DataFrame]) -> None:
        self.index = X.index if isinstance(X, pd.DataFrame) else range(len(X))
    
    def fit(self, X: Union[List[str], pd.DataFrame], y=None):
        return self
    
    def transform(self, X: Union[List[str], pd.DataFrame], y=None) -> pd.DataFrame:
        with torch.no_grad():
            mean_hidden_states = []
            doc_lengths = []
            for doc in self._tqdm(X):
                inputs = self.tokenizer.encode(doc, return_tensors='pt')
                # if the input is longer than self.token_length tokens the model should convolve over the input
                # and take the mean of the last hidden state
                LEN = inputs.shape[1]
                doc_lengths.append(LEN)
                if LEN < self.token_length:
                    output = self.model(input_ids=inputs)

                    if self.NSP:
                        hidden_states = torch.stack(output.hidden_states)
                        mean_vector = hidden_states[-1][:,0,:].squeeze().detach().numpy()
                    else:
                        mean_vector = torch.mean(output.last_hidden_state, dim=1).squeeze().detach().numpy()
                else:            
                    mean_vector = np.zeros(self.hiddenstate_length)
                    for num, i in enumerate(range(0, LEN, self.token_length)):
                        output = self.model(input_ids=inputs[:,i:min([LEN,i+self.token_length])])

                        if self.NSP:
                            hidden_states = torch.stack(output.hidden_states)
                            mean_vector += hidden_states[-1][:,0,:].squeeze().detach().numpy()
                        else:
                            mean_vector += torch.mean(output.last_hidden_state, dim=1).squeeze().detach().numpy()
                    mean_vector /= num
                mean_hidden_states.append(mean_vector)

        self._indexify(X)
        return pd.DataFrame(data=np.array(mean_hidden_states),
                            columns=['dim_'+str(c) for c in range(self.hiddenstate_length)],
                            index=self.index)

#class Doc2Vec(BaseEstimator, TransformerMixin):

#class USEEmbedder(BaseEstimator, TransformerMixin):
# https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1

#class LaBSE(BaseEstimator, TransformerMixin):
# https://tfhub.dev/google/LaBSE/2

#class ELMo(BaseEstimator, TransformerMixin):
# https://github.com/HIT-SCIR/ELMoForManyLangs

class DocEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, method='word2vec',
                       aggregation='mean', 
                       embedding_path='./vectors/vec.bin',
                       embedding_object=None, 
                       tokenizer=None,
                       weights=None,
                       to_dataframe=False,
                       output_dimensions = None,
                       NSP=False,
                       dimension_reducer=None,
                       embedding_kwargs={},
                       missing_token_handling=True,
                       max_word_edist=2
                       ):
        '''
        DocEmbedder class is meant for embedding documents with word/tokenembedders 
        where a custom aggregation is required

            method: 'word2vec', 'fasttext'
            aggregation: 'mean', 'median', 'max', 'weighted_mean', 'hierarchical'
            embedding_path: path to embedding file, or name of 'transformer' model
            embedding_object: pre-loaded embedding object
            NSP: next-sentence-prediction as pre-training task for transformer
            tokenizer: function that takes in a string and returns a list of tokens
            weights: dictionary of weights for each token, 
                        OR a function that takes in a list of
                        tokens and returns a list of weights
            output_dimensions: 

        '''
        assert(method in ['word2vec', 'fasttext', 'transformer', 'sbert']), \
            'method must be one of: word2vec, fasttext, transformer, sbert'
        assert(aggregation in ['mean', 'median', 'max']), \
            'aggregation must be one of: mean, median, max'
        

        self.method = method
        self.aggregation = aggregation
        self.embedding_object = embedding_object
        self.tokenizer = tokenizer # by default this is overwritten if method is 'sbert' or 'transformer'
        self.embedding = self.load_embedding(embedding_path, **embedding_kwargs)
        self.weights = weights
        self.to_dataframe = to_dataframe
        self.output_dimensions = output_dimensions
        self.NSP = NSP
        self.missing_token_handling = missing_token_handling
        
        if (self.aggregation in ['max', 'median']) & (self.method=='sbert'):
            raise Warning('aggregation is not settable in SBERT models, it is integrated in the \
            pretrained model. Check your model for the aggregation')

        if self.tokenizer is None:
            self.get_tokenizer()

        self.data_index = None
        self.logger = logging.getLogger(__name__)
        self.logger.info('DocEmbedder initialized')

        if self.missing_token_handling:
            self.logger.info('Missing token handling is enabled')
            EmbeddingVocab = list(self.embedding.key_to_index.keys())
            self.vocab_db = fastss.FastSS(words=EmbeddingVocab, max_dist=max_word_edist)

    def get_tokenizer(self):
        '''
            tokenizer: function that takes in a string and returns a list of tokens
            if no tokenizer is provided, we need to return a
            function that takes in a string and returns a list of tokens
        '''        
        self.tokenizer = lambda x: x.split()
        return True

    def sent_embedder(self, sentence: str) -> np.ndarray:       
        tokens = self.tokenizer(sentence)
        embs = []
        self.no_replacements = []
        for _tok in tokens:
            try:
                embs.append(self.embedding[_tok])
            except:
                if self.missing_token_handling:
                    try:
                        replacements = self.vocab_db.query(_tok)
                        if sum([len(s) for s in replacements.values()])>0:
                            embs.append(self.embedding[replacements[1][0]])
                        else:
                            pass
                    except Exception as e:
                        self.no_replacements.append(_tok)
                        #self.logger.warning(f'Query for {_tok}: {self.vocab_db.query(_tok)}')
                        pass

                else:
                    pass
        return np.mean(embs, axis=0)

    def load_embedding(self, embedding_path, embedding_kwargs={}):
        if self.method in ['word2vec', 'fasttext']:            
            return KeyedVectors.load(embedding_path, **embedding_kwargs)
        else:
            raise ValueError('method must be one of: word2vec, fasttext')
    
    def _word2vec_transform(self, X):
        ''' Assumes that X is a list of lists with text

        '''
        result = []
        if self.aggregation in ['mean', 'max', 'median']:
            aggfun = eval('np.'+self.aggregation)
            for col in X:
                _col = [aggfun([self.embedding[t] 
                                        for t in self.tokenizer(doc).tokens],
                                                                                axis=0) 
                                        for doc in col]
                result.append(_col)
            return result
        else:
            raise ValueError('aggregation must be one of: mean, max, median')
    
    
    def reducer(self):
        '''
            Assume that X is a list of lists with text
        '''
        reducer = UMAP(n_components=self.output_dimensions)
        
        output = []
        for c in self._transformed:
            _X = np.arrray(c)
            __X = reducer.fit_transform(_X)
            output.append(__X)
        self._transformed = output
    
    def to_dataframe(self, X):
        '''
            Assume that X is a list of lists with text -> pd.DataFrame
        '''
        output = pd.DataFrame()
        for idx, c in enumerate(X):
            _X = np.array(c)
            temp = pd.DataFrame(data=_X,
                         columns=['col_'+str(idx)+'_dim_'+str(c) 
                                    for c in range(_X.shape[1])],
                         index=self.data_index)
            output = pd.concat([output, temp], axis=1)
        return output


    def _data_check(self, X):
        err_str = 'X should be a pandas.Series, a pandas.DataFrame, a np.array or a list of strings'
        if isinstance(X, list):
            # check if all items are lists, in which all items are text
            # of if all items is text
            
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
            return X.T.values.tolist()
        elif isinstance(X, pd.Series):
            assert X.dtype==object, 'X should be pd.Series with object dtype'
            return X.values.tolist()
        elif isinstance(X, np.ndarray):
            assert (X.dtype == np.dtype('object'))|(X.dtype==np.dtype('<U1')), \
                'X should be np.array of strings'
            return X.T.tolist()

    def fit(self, X, y=None):
        '''
            Check format of incoming data: turn into list of lists, 
            where each list within list represents a column 
        '''
        assert (isinstance(X, list) | 
                isinstance(X, pd.DataFrame) | 
                isinstance(X, np.ndarray)), "X should be lists of lists, \
                a list of text, pd.DataFrame or np.array"
        
        _X = self._data_check(X)

        if self.method in ['word2vec', 'fasttext']:
            self._transformed = self._word2vec_transform(_X)
        elif self.method == 'sbert':
            self._transformed = self._sbert_transform(_X)
        elif self.method == 'transformer':
            self._transformed == self._transformer_transform(_X)
            
        if isinstance(self.output_dimensions, int):
            self.reducer()
        return self
    
    def transform(self, X, y=None):
        '''
            X: list of strings
            y: list of labels
        '''
        if self.to_dataframe:
            return self.to_dataframe(self._transformed)
        else:
            return self._transformed

'''Create main method for testing purposes
'''
if __name__ == '__main__':
    '''
        loads in arguments through argparse
    '''
    parser = argparse.ArgumentParser(description='Embedding transformer')
    parser.add_argument('--method', type=str, default='word2vec')
    parser.add_argument('--model', type=str, default='word2vec')
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--output_dimensions', type=int, default=2)
    parser.add_argument('--to_dataframe', type=bool, default=False)