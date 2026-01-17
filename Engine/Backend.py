from google.cloud import storage
import pickle
import sys
import math
import re
from heapq import heappush, heappop
from collections import Counter
from contextlib import closing
import nltk

# Ensure this file exists in the path, otherwise this import fails
from inverted_index_gcp import * 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Bucket and paths
BUCKET_NAME = "ir_project_2025"

BODY_FOLDER = "body100"
TITLE_FOLDER = "title1"
ANCHOR_FOLDER = "anchor4"
BODY_INDEX_PATH = f"{BODY_FOLDER}/body100_index.pkl"
TITLE_INDEX_PATH = f"{TITLE_FOLDER}/title1_index.pkl"
ANCHOR_INDEX_PATH = f"{ANCHOR_FOLDER}/anchor4_index.pkl"
ID_TO_TITLE_PATH = "id_to_title_dict/id_to_title.pkl"
PAGEVIEWS_PATH = "gs://ir_project_2025/pageviews_aug_2021"

# Load indexes from GCP
# This runs on import. Ensure credentials are set.
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

print("Loading Body Index...")
body_index = pickle.loads(
    bucket.blob(BODY_INDEX_PATH).download_as_bytes()
)
print("Loading Title Index...")
title_index = pickle.loads(
    bucket.blob(TITLE_INDEX_PATH).download_as_bytes()
)
print("Loading Anchor Index...")
anchor_index = pickle.loads(
    bucket.blob(ANCHOR_INDEX_PATH).download_as_bytes()
)
print("Loading ID map...")
id_to_title = pickle.loads(
    bucket.blob(ID_TO_TITLE_PATH).download_as_bytes()
)
print("Loading Pageviews...")
# Direct read into Pandas
# Requires pyarrow
parquet_pageviews = pd.read_parquet("gs://ir_project_2025/pageviews_aug_2021")
# Create the dictionary
pageviews = dict(zip(parquet_pageviews['wiki_id'], parquet_pageviews['pageviews']))
print("Indexes Loaded.")

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

    def read_posting_list(self, inverted, w, folder):
        with closing(MultiFileReader(BUCKET_NAME, folder)) as reader:
            try:
                # Check if term exists to avoid KeyError before calling reader
                if w not in inverted.posting_locs:
                    return []
                
                locs = inverted.posting_locs[w]
                # df * 6 because each posting is 6 bytes (4 for doc_id, 2 for tf)
                data = reader.read(locs, inverted.df[w] * 6, BUCKET_NAME)
            except Exception as e:
                print(f"An error occurred: {e}")
                return []

            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(data[i*6:i*6+4], 'big')
                tf = int.from_bytes(data[i*6+4:(i+1)*6], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    # =========================
    # BM25 Search (MAIN)
    # =========================
    def search_body_bm25(self, query, k1=1.5, b=0.75):
        body_index_local = body_index
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = {}
        N = body_index_local.N
        avgDL = sum(val[0] for val in body_index_local.weights.values()) / N

        for term in set(query_tokens):
            if term not in body_index_local.df:
                continue

            df = body_index_local.df[term]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            posting_list = self.read_posting_list(body_index_local, term, BODY_FOLDER)

            for doc_id, tf in posting_list:
                # Accessing weights[doc_id][0] -> Document Length
                doc_len = body_index_local.weights[doc_id][0]
                denom = tf + k1 * (1 - b + b * (doc_len / avgDL))
                score = idf * ((tf * (k1 + 1)) / denom)
                scores[doc_id] = scores.get(doc_id, 0) + score

        # Use the helper method to sort and format
        return self.get_top_n(scores, 100)

    # TF-IDF + Cosine Sim
    def search_body(self, query, k=100):
        """
        TF-IDF with cosine similarity over the body.
        Corrected to ensure Numerator (Dot Product) matches Denominator (Norm).
        """
        tokens = self.tokenize(query)
        if not tokens:
            return []
            
        tf_query = Counter(tokens)
        body_index_local = body_index

        # Calculate Query Weights (TF-IDF for query)
        query_vec = {}
        for t in tf_query:
            if t in body_index_local.df:
                idf = math.log(body_index_local.N / body_index_local.df[t])
                # Store tuple (weight, idf) or just recalculate IDF later.
                # Here we calculate w_q.
                weight = (tf_query[t] / len(tokens)) * idf
                query_vec[t] = weight

        # Dot Product
        candidates = {}
        for term, query_weight in query_vec.items():
            # Recalculate IDF for the document side
            idf = math.log(body_index_local.N / body_index_local.df[term])
            
            posting_list = self.read_posting_list(body_index_local, term, BODY_FOLDER)
            for doc_id, tf in posting_list:
                # Apply IDF to document TF to match the pre-calculated Norm
                doc_weight = tf * idf 
                
                candidates[doc_id] = candidates.get(doc_id, 0) + doc_weight * query_weight

        # Normalize (Cosine Similarity)
        query_norm = math.sqrt(sum(v**2 for v in query_vec.values()))
        
        final_scores = {}
        for doc_id, dot_product in candidates.items():
            # Check for safety to avoid DivisionByZero
            try:
                doc_norm = math.sqrt(body_index_local.weights[doc_id][1]) 
                if doc_norm > 0 and query_norm > 0:
                    final_scores[doc_id] = dot_product / (doc_norm * query_norm)
                else:
                    final_scores[doc_id] = 0
            except:
                final_scores[doc_id] = 0

        return self.get_top_n(final_scores, k)

    # Binary Title Ranking
    def search_title(self, query):
        """
        Ranks documents by the number of distinct query terms
        appearing in the title.
        """
        title_index_local = title_index
        tokens = set(self.tokenize(query))
        counts = Counter()

        for term in tokens:
            if term in title_index_local.df:
                for doc_id, _ in self.read_posting_list(title_index_local, term, TITLE_FOLDER):
                    counts[doc_id] += 1

        return self.sort_all(counts)

    # Binary Anchor Ranking
    def search_anchor(self, query):
        """
        Ranks documents by total frequency of query terms
        appearing in anchor text.
        """
        anchor_index_local = anchor_index
        tokens = self.tokenize(query)
        counts = Counter()

        for term in tokens:
            if term in anchor_index_local.df:
                for doc_id, tf in self.read_posting_list(anchor_index_local, term, ANCHOR_FOLDER):
                    counts[doc_id] += tf

        return self.sort_all(counts)

    # Lists the page views of the provided ids
    def get_pageviews(self, ids):
        results = []
        for id in ids:
            results.append(pageviews.get(id, 0))
        return results

    # Helpers
    def get_top_n(self, scores, k):
        """
        Returns top-k documents sorted by score.
        """
        # We use a Min-Heap to keep top K, or sort all if K is large
        # For small K (100) and large N, heaps are efficient.
        # But Python's heappop pops the smallest. 
        # So we push (-score, doc_id) and pop smallest to retrieve highest scores.
        
        heap = []
        for doc_id, score in scores.items():
            heappush(heap, (-score, doc_id))
        
        results = []
        while heap and len(results) < k:
            neg_score, doc_id = heappop(heap)
            # Use global id_to_title map
            title = id_to_title.get(doc_id, "TITLE NOT FOUND")
            results.append((str(doc_id), title))
            
        return results

    def sort_all(self, counts):
        """
        Returns all documents sorted by count (descending).
        """
        return [
            (str(doc_id), id_to_title.get(doc_id, "TITLE NOT FOUND"))
            for doc_id, _ in counts.most_common()
        ]