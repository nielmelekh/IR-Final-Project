#Backend.py
# =========================
# Imports
# =========================
from google.cloud import storage
import pickle
import sys
import math
import re
from heapq import heappush, heappop, heapify
from collections import Counter
from contextlib import closing

# make sure inverted_index_gcp is found
sys.path.append('/home/dataproc')
from inverted_index_gcp import *

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =========================
# Bucket & paths
# =========================
BUCKET_NAME = "ir_project_2025"

BODY_FOLDER = "body100"
BODY_INDEX_PATH = f"{BODY_FOLDER}/body100_index.pkl"
ID_TO_TITLE_PATH = "id_to_title_dict/id_to_title.pkl"

# =========================
# Load indexes from GCP
# =========================
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# load body index
index_body = pickle.loads(
    bucket.blob(BODY_INDEX_PATH).download_as_bytes()
)

# load id_to_title
id_to_title = pickle.loads(
    bucket.blob(ID_TO_TITLE_PATH).download_as_bytes()
)

# =========================
# Tokenizer setup
# =========================
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = {
    "category", "references", "also", "external", "links", "may",
    "first", "see", "history", "people", "one", "two", "part",
    "thumb", "including", "second", "following", "many", "however", "would"
}

all_stopwords = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# =========================
# Backend Search Class
# =========================
class Backend_Search:
    def __init__(self):
        pass

    def tokenize(self, text):
        stemmer = PorterStemmer()
        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
        tokens = [t for t in tokens if t not in all_stopwords]
        return [stemmer.stem(t) for t in tokens]

    def read_posting_list(self, inverted, w):
        with closing(MultiFileReader(BUCKET_NAME, BODY_FOLDER)) as reader:
            try:
                locs = inverted.posting_locs[w]
                b = reader.read(locs, inverted.df[w] * 6, BUCKET_NAME)
            except:
                return []

            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i*6:i*6+4], 'big')
                tf = int.from_bytes(b[i*6+4:(i+1)*6], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def get_candidate_docs(self, query, query_score):
        candidates = {}
        for term in set(query):
            try:
                posting_list = self.read_posting_list(index_body, term)
                for doc_id, tf in posting_list:
                    score = (
                        (tf / index_body.weights[doc_id][0]) *
                        math.log(index_body.N / index_body.df[term]) *
                        query_score[term]
                    )
                    candidates[doc_id] = candidates.get(doc_id, 0) + score
            except:
                continue
        return candidates

    def search_body(self, query):
        query_tokens = self.tokenize(query)
        query_score = calc_tf_idf_query(query_tokens, index_body)

        if not query_score:
            return []

        candidates = self.get_candidate_docs(query_tokens, query_score)
        scores = cos_sim(candidates, index_body, query_score)

        heap = []
        for doc_id, score in scores.items():
            heappush(heap, (-score, doc_id))

        results = []
        while heap and len(results) < 100:
            score, doc_id = heappop(heap)
            title = id_to_title.get(doc_id, "TITLE NOT FOUND")
            results.append((doc_id, title))

        return results

# =========================
# Helper functions
# =========================
def cos_sim(candidates, index, query_score):
    scores = {}
    query_norm = math.sqrt(sum(v*v for v in query_score.values()))
    for doc_id, numerator in candidates.items():
        scores[doc_id] = numerator / (
            math.sqrt(index.weights[doc_id][1]) * query_norm
        )
    return scores

def calc_tf_idf_query(query_tokens, index):
    tf = Counter(query_tokens)
    scores = {}
    for term in tf:
        try:
            scores[term] = (tf[term] / len(query_tokens)) * math.log(index.N / index.df[term])
        except:
            continue
    return scores
