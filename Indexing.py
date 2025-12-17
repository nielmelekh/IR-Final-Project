import math
import numpy as np

def tf_idf(term_freq, doc_freq, total_docs):
    if term_freq == 0:
        return 0.0
    tf = 1 + math.log10(term_freq)
    idf = math.log10(total_docs / doc_freq) if doc_freq > 0 else 0.0
    return tf * idf

def tf_idf_inverted_index_row(t_pl_total, total_docs):
    tf_idf_vector = []
    doc_freq = len(t_pl_total[1])
    for posting in t_pl_total[1]:
        tf_idf_vector.append(posting[0], tf_idf(posting[1], doc_freq, total_docs))
    return tf_idf_vector

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)