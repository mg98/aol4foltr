import re
import random
import numpy as np
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import hashlib
from ir_datasets.datasets.aol_ia import AolIaDoc

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

class TFIDF:
    def __init__(self, corpus: Dict[str, str]):
        """
        Initialize a TF-IDF model with a corpus of documents.
        The model computes TF-IDF vectors for all documents in the corpus and
        provides methods to calculate term weights and document similarity.
        
        Args:
            corpus: A dictionary mapping document IDs to document text.
                   Each document will be vectorized and indexed for retrieval.
        """
        self.corpus = {doc_id: text for doc_id, text in corpus.items()}
        
        if self.is_empty_corpus:
            self.tfidf_matrix = None
            self.feature_names = []
            self.feature_to_idx = {}
            self.doc_to_idx = {}
            self.term_counts = []
            self.total_terms = []
            self._vector_cache = {}
            self.vectorizer = None
            return
        
        self.vectorizer = TfidfVectorizer(
            use_idf=True, smooth_idf=True, norm=None,
            token_pattern=r'(?u)\b\w\w*\b|[0-9]+'
        )
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus.values())
        except ValueError as e:
            # Handle unexpected empty vocabulary error as fallback
            if "empty vocabulary" in str(e):
                print("Warning: Unexpected empty vocabulary error. Creating empty TF-IDF model.")
                print("This shouldn't happen if is_empty_corpus works correctly.")
                # Initialize empty attributes
                self.tfidf_matrix = None
                self.feature_names = []
                self.feature_to_idx = {}
                self.doc_to_idx = {}
                self.term_counts = []
                self.total_terms = []
                self._vector_cache = {}
                self.vectorizer = None
                return
            else:
                raise e
        except Exception as e:
            print("Corpus values causing the error:")
            print(list(self.corpus.values()))
            raise e
            
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Add these hash maps for O(1) lookups
        self.feature_to_idx = {term: idx for idx, term in enumerate(self.feature_names)}
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.corpus.keys())}
        # Precompute term counts for each document
        self.term_counts = [doc.split() for doc in self.corpus.values()]
        self.total_terms = [len(doc) for doc in self.term_counts]
        # Add vector cache
        self._vector_cache = {}

    @property
    def is_empty_corpus(self) -> bool:
        """Check if the corpus is empty, contains only whitespace, or only stop words."""
        if len(self.corpus) == 0:
            return True
        stop_words = set(ENGLISH_STOP_WORDS)
        for doc_text in self.corpus.values():
            if not doc_text.strip():
                continue
            tokens = tokenize(doc_text.lower())
            if any(token not in stop_words for token in tokens):
                return False
        return True

    def get_tf_idf(self, doc_id: str, term: str) -> dict[str, float]:
        if self.is_empty_corpus:
            return {"tf": 0, "tf_idf": 0, "idf": 0}
        
        try:
            word_idx = self.feature_to_idx[term]
        except KeyError:
            return {"tf": 0, "tf_idf": 0, "idf": 0}
        
        doc_idx = self.doc_to_idx[doc_id]
        tf_idf = self.tfidf_matrix[doc_idx, word_idx]
        idf = self.vectorizer.idf_[word_idx]
        tf = tf_idf / idf if idf != 0 else 0
        
        return {"tf": tf, "tf_idf": tf_idf, "idf": idf}
    
    def get_vector(self, query: str) -> np.ndarray:
        if self.is_empty_corpus:
            return np.array([])
        if query not in self._vector_cache:
            self._vector_cache[query] = self.vectorizer.transform([query]).toarray()[0]
        return self._vector_cache[query]

    def get_cos_sim(self, doc_id: str, query_terms: list[str]) -> float:
        """Compute cosine similarity between the query and a document."""
        if self.is_empty_corpus or doc_id not in self.doc_to_idx:
            return 0.0
        
        query = ' '.join(query_terms)
        query_vector = self.get_vector(query)
        doc_idx = self.doc_to_idx[doc_id]
        document_vector = self.tfidf_matrix[doc_idx]
        dot_product = document_vector.dot(query_vector)[0]
        query_magnitude = np.linalg.norm(query_vector)
        document_magnitude = np.linalg.norm(document_vector.toarray())
        
        if query_magnitude == 0 or document_magnitude == 0:
            return 0.0
        return dot_product / (query_magnitude * document_magnitude)
    
    def get_document_text(self, doc_id: str) -> str:
        """Get the original document text for a given document ID."""
        return self.corpus[doc_id]

class Corpus:
    def __init__(self, docs: Dict[str, str]):
        self.docs = docs
        self.tfidf = TFIDF(docs)
        self.bm25_doc_mapping = {}  # Maps original doc_id to BM25 index
        
        if not self.is_empty():
            doc_ids = list(docs.keys())
            tokenized_docs = [tokenize(docs[doc_id]) for doc_id in doc_ids]
            
            # Keep track of which documents have non-empty tokens
            non_empty_docs = []
            bm25_index = 0
            
            for i, tokens in enumerate(tokenized_docs):
                if tokens:  # Non-empty token list
                    non_empty_docs.append(tokens)
                    self.bm25_doc_mapping[doc_ids[i]] = bm25_index
                    bm25_index += 1
            
            if non_empty_docs:
                self.bm25 = BM25Okapi(non_empty_docs)
            else:
                self.bm25 = None

    def is_empty(self) -> bool:
        if len(self.docs) == 0 or all(not doc.strip() for doc in self.docs.values()):
            return True
        # Check if all documents result in empty token lists
        tokenized_docs = [tokenize(doc) for doc in self.docs.values()]
        return all(not tokens for tokens in tokenized_docs)


class Statistics:
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0
    mean: float = 0.0
    variance: float = 0.0

    @classmethod
    def make(cls, values: list[float] = []):
        v = cls()
        if len(values) == 0:
            return v
        v.min = min(values)
        v.max = max(values)
        v.sum = sum(values)
        v.mean = v.sum / len(values)
        v.variance = sum((x - v.mean) ** 2 for x in values) / len(values)
        return v
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        v = cls()
        v.min = float(arr[0])
        v.max = float(arr[1])
        v.sum = float(arr[2])
        v.mean = float(arr[3])
        v.variance = float(arr[4])
        return v
    
    @property
    def features(self) -> list[float]:
        return [
            self.min,
            self.max,
            self.sum,
            self.mean,
            self.variance
        ]

@dataclass
class TermBasedFeatures:
    bm25: float = 0.0

    tf: Statistics = Statistics()
    idf: Statistics = Statistics()
    tf_idf: Statistics = Statistics()
    stream_length: Statistics = Statistics()
    stream_length_normalized_tf: Statistics = Statistics()

    cos_sim: float = 0.0
    covered_query_term_number: int = 0
    covered_query_term_ratio: float = 0.0
    char_len: int = 0
    term_len: int = 0
    total_query_terms: int = 0
    exact_match: int = 0
    match_ratio: float = 0.0

    @classmethod
    def make(cls, corpus: Corpus, query: str, doc_id: str) -> 'TermBasedFeatures':
        v = cls()
        if corpus.is_empty():
            return v

        query_terms = tokenize(query)
        doc_text = corpus.docs[doc_id]
        doc_text_terms = tokenize(doc_text)

        if corpus.bm25 is not None and doc_id in corpus.bm25_doc_mapping:
            bm25_doc_index = corpus.bm25_doc_mapping[doc_id]
            v.bm25 = corpus.bm25.get_batch_scores(query_terms, [bm25_doc_index])[0]
        else:
            v.bm25 = 0.0

        tfidf_results = [corpus.tfidf.get_tf_idf(doc_id, term) for term in query_terms]

        v.tf = Statistics.make([r["tf"] for r in tfidf_results])
        v.idf = Statistics.make([r["idf"] for r in tfidf_results])
        v.tf_idf = Statistics.make([r["tf_idf"] for r in tfidf_results])
        
        v.stream_length = Statistics.make([len(term) for term in doc_text_terms])
        if len(doc_text_terms) > 0:
            v.stream_length_normalized_tf = Statistics.make([sum(r["tf"] for r in tfidf_results) / len(doc_text_terms)])
        else:
            v.stream_length_normalized_tf = Statistics()
        
        v.cos_sim = corpus.tfidf.get_cos_sim(doc_id, query_terms)

        v.covered_query_term_number = sum(1 for r in tfidf_results if r["tf"] > 0)
        v.covered_query_term_ratio = v.covered_query_term_number / len(query_terms)

        # Get document text from tfidf to calculate lengths
        v.char_len = len(doc_text)
        v.term_len = len(tokenize(doc_text))
        
        # Boolean features
        document_terms = tokenize(doc_text)
        matched_terms = set(query_terms) & set(document_terms)
        match_count = len(matched_terms)
        v.total_query_terms = len(query_terms)
        v.exact_match = 1 if match_count == v.total_query_terms else 0
        v.match_ratio = match_count / v.total_query_terms if v.total_query_terms > 0 else 0

        return v
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        v = cls()
        v.bm25 = float(arr[0])
        v.tf = Statistics.from_array(arr[1:6])
        v.idf = Statistics.from_array(arr[6:11])
        v.tf_idf = Statistics.from_array(arr[11:16])
        v.stream_length = Statistics.from_array(arr[16:21])
        v.stream_length_normalized_tf = Statistics.from_array(arr[21:26])
        v.cos_sim = float(arr[26])
        v.covered_query_term_number = int(arr[27])
        v.covered_query_term_ratio = float(arr[28])
        v.char_len = int(arr[29])
        v.term_len = int(arr[30])
        v.total_query_terms = int(arr[31])
        v.exact_match = int(arr[32])
        v.match_ratio = float(arr[33])
        return v
    
    @property
    def features(self) -> list[float]:
        return [
            self.bm25, # 1
            *self.tf.features, # 2-6
            *self.idf.features, # 7-11
            *self.tf_idf.features, # 12-16
            *self.stream_length.features, # 17-21
            *self.stream_length_normalized_tf.features, # 22-26
            self.cos_sim, # 27
            self.covered_query_term_number, # 28
            self.covered_query_term_ratio, # 29
            self.char_len, # 30
            self.term_len, # 31
            self.total_query_terms, # 32
            self.exact_match, # 33
            self.match_ratio # 34
        ]


class FeatureVector:
    title: TermBasedFeatures = TermBasedFeatures() # 0-33
    body: TermBasedFeatures = TermBasedFeatures() # 34-67
    url: TermBasedFeatures = TermBasedFeatures() # 68-101
    number_of_slash_in_url: int = 0 # 102
    
    @classmethod
    def make(cls, candidate_docs: list[AolIaDoc], doc: AolIaDoc, query: str) -> 'FeatureVector':
        v = cls()
        title_corpus = Corpus({ doc.doc_id: doc.title.lower().strip() for doc in candidate_docs })
        body_corpus = Corpus({ doc.doc_id: doc.text.lower().strip() for doc in candidate_docs })
        url_corpus = Corpus({ doc.doc_id: doc.url.lower().strip() for doc in candidate_docs })
        v.title = TermBasedFeatures.make(title_corpus, query, doc.doc_id)
        v.body = TermBasedFeatures.make(body_corpus, query, doc.doc_id)
        v.url = TermBasedFeatures.make(url_corpus, query, doc.doc_id)
        v.number_of_slash_in_url = url_corpus.docs[doc.doc_id].count('/')
        return v
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        v = cls()
        v.title = TermBasedFeatures.from_array(arr[0:102:3])
        v.body = TermBasedFeatures.from_array(arr[1:102:3])
        v.url = TermBasedFeatures.from_array(arr[2:102:3])
        v.number_of_slash_in_url = int(arr[102])
        return v
    
    @classmethod
    def n_features(cls) -> int:
        return len(cls().features)

    def __str__(self):
        return ' '.join(f'{i+1}:{val}' for i, val in enumerate(self.features))

    @property
    def features(self):
        return [
            # BM25 1-3
            self.title.bm25,
            self.body.bm25,
            self.url.bm25,

            # TF 4-18
            self.title.tf.min,
            self.body.tf.min,
            self.url.tf.min,

            self.title.tf.max,
            self.body.tf.max,
            self.url.tf.max,

            self.title.tf.sum,
            self.body.tf.sum,
            self.url.tf.sum,
            
            self.title.tf.mean,
            self.body.tf.mean,
            self.url.tf.mean,
            
            self.title.tf.variance,
            self.body.tf.variance,
            self.url.tf.variance,

            # IDF 19-33
            self.title.idf.min,
            self.body.idf.min,
            self.url.idf.min,

            self.title.idf.max,
            self.body.idf.max,
            self.url.idf.max,

            self.title.idf.sum,
            self.body.idf.sum,
            self.url.idf.sum,

            self.title.idf.mean,
            self.body.idf.mean,
            self.url.idf.mean,

            self.title.idf.variance,
            self.body.idf.variance,
            self.url.idf.variance,

            # TF-IDF 34-48
            self.title.tf_idf.min,
            self.body.tf_idf.min,
            self.url.tf_idf.min,

            self.title.tf_idf.max,
            self.body.tf_idf.max,
            self.url.tf_idf.max,

            self.title.tf_idf.sum,
            self.body.tf_idf.sum,
            self.url.tf_idf.sum,

            self.title.tf_idf.mean,
            self.body.tf_idf.mean,
            self.url.tf_idf.mean,

            self.title.tf_idf.variance,
            self.body.tf_idf.variance,
            self.url.tf_idf.variance,

            # Stream Length 49-63
            self.title.stream_length.min,
            self.body.stream_length.min,
            self.url.stream_length.min,

            self.title.stream_length.max,
            self.body.stream_length.max,
            self.url.stream_length.max,

            self.title.stream_length.sum,
            self.body.stream_length.sum,
            self.url.stream_length.sum,

            self.title.stream_length.mean,
            self.body.stream_length.mean,
            self.url.stream_length.mean,

            self.title.stream_length.variance,
            self.body.stream_length.variance,
            self.url.stream_length.variance,

            # Stream Length Normalized TF 64-78
            self.title.stream_length_normalized_tf.min,
            self.body.stream_length_normalized_tf.min,
            self.url.stream_length_normalized_tf.min,

            self.title.stream_length_normalized_tf.max,
            self.body.stream_length_normalized_tf.max,
            self.url.stream_length_normalized_tf.max,

            self.title.stream_length_normalized_tf.sum,
            self.body.stream_length_normalized_tf.sum,
            self.url.stream_length_normalized_tf.sum,

            self.title.stream_length_normalized_tf.mean,
            self.body.stream_length_normalized_tf.mean,
            self.url.stream_length_normalized_tf.mean,

            # Stream Length Normalized TF Variance 76-78
            self.title.stream_length_normalized_tf.variance,
            self.body.stream_length_normalized_tf.variance,
            self.url.stream_length_normalized_tf.variance,

            # Cosine Similarity 79-81
            self.title.cos_sim,
            self.body.cos_sim,
            self.url.cos_sim,

            # Covered Query Term Number 82-84
            self.title.covered_query_term_number,
            self.body.covered_query_term_number,
            self.url.covered_query_term_number,

            # Covered Query Term Ratio 85-87
            self.title.covered_query_term_ratio,
            self.body.covered_query_term_ratio,
            self.url.covered_query_term_ratio,

            # Char Length 88-90
            self.title.char_len,
            self.body.char_len,
            self.url.char_len,

            # Term Length 91-93
            self.title.term_len,
            self.body.term_len,
            self.url.term_len,

            # Total Query Terms 94-96
            self.title.total_query_terms,
            self.body.total_query_terms,
            self.url.total_query_terms,

            # Exact Match 97-99
            self.title.exact_match,
            self.body.exact_match,
            self.url.exact_match,

            # Match Ratio 100-102
            self.title.match_ratio,
            self.body.match_ratio,
            self.url.match_ratio,

            # Number of Slash in URL 103
            self.number_of_slash_in_url,
        ]

class ClickThroughRecord:
    rel: bool
    qid: int
    feat: FeatureVector

    def __init__(self, rel=False, qid=0, feat=None):
        self.rel = rel
        self.qid = qid
        self.feat = feat
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(
            rel=bool(arr[0]),
            qid=int(arr[1]),
            feat=FeatureVector.from_array(arr[2:])
        )

    def to_dict(self):
        return {
            'rel': self.rel,
            'qid': self.qid,
            'feat': self.feat
        }
    
    def to_array(self) -> np.ndarray:
        return np.array([self.rel, self.qid, *self.feat.features], dtype=np.float32)
    
    def __str__(self):
        return f'{int(self.rel)} qid:{self.qid} {self.feat}'
    
@dataclass
class SplitDataset:
    train: list[ClickThroughRecord]
    vali: list[ClickThroughRecord]
    test: list[ClickThroughRecord]

    def _shuffle_maintain_groups(self, records: list[ClickThroughRecord]) -> list[ClickThroughRecord]:
        """
        Helper method to shuffle records while maintaining query groups.
        
        Args:
            records: List of ClickThroughRecord objects
            
        Returns:
            Shuffled list with query groups preserved
        """
        # Group records by query ID
        query_groups = {}
        for record in records:
            if record.qid not in query_groups:
                query_groups[record.qid] = []
            query_groups[record.qid].append(record)
        
        # Get list of query IDs and shuffle their order
        query_ids = list(query_groups.keys())
        random.shuffle(query_ids)
        
        # Optionally shuffle records within each query group
        for qid in query_ids:
            random.shuffle(query_groups[qid])
        
        # Reconstruct the list with shuffled query order
        shuffled_records = []
        for qid in query_ids:
            shuffled_records.extend(query_groups[qid])
            
        return shuffled_records

    def shuffle(self):
        self.train = self._shuffle_maintain_groups(self.train)
        self.vali = self._shuffle_maintain_groups(self.vali)
        self.test = self._shuffle_maintain_groups(self.test)

    def to_dict(self):
        return {
            'train': self.train,
            'vali': self.vali,
            'test': self.test
        }

@dataclass
class Dataset:
    context: list[ClickThroughRecord]
    test: list[ClickThroughRecord]

    def split(self) -> SplitDataset:        
        context_copy = self.context.copy()
        random.shuffle(context_copy)
        total_context = len(context_copy)
        train_size = int((8/9) * total_context)
        train_records = context_copy[:train_size]
        vali_records = context_copy[train_size:]
        return SplitDataset(train_records, vali_records, self.test)