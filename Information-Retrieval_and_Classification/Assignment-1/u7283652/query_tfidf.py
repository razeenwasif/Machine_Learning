import math
from collections import defaultdict

from query import (
    get_query_tokens,
    count_query_tokens,
    query_main,
)


def get_doc_to_norm(index, doc_freq, num_docs):
    """Pre-compute the norms for each document vector in the corpus using tfidf.

    Args:
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        num_docs (int): number of documents in the corpus

    Returns:
        dict(int: float): a dictionary mapping doc_ids to document norms
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the get_doc_to_norm function in query.py
    #       but should use tfidf instead of term frequency

    doc_to_sum_squares  = defaultdict(float)

    for term, postings in index.items():
        if term in doc_freq and doc_freq[term] > 0:
            idf = math.log10(num_docs / doc_freq[term])

            for docid, term_freq in postings:
                tf = term_freq
                tfidf_weight = tf * idf 
                doc_to_sum_squares[docid] += tfidf_weight ** 2
    doc_norm = {
        docid: math.sqrt(sum_sq)
        for docid, sum_sq in doc_to_sum_squares.items()
    }

    return doc_norm


def run_query(query_string, index, doc_freq, doc_norm, num_docs):
    """ Run a query on the index and return a sorted list of documents. 
    Sorted by most similar to least similar.
    Documents not returned in the sorted list are assumed to have 0 similarity.

    Args:
        query_string (str): the query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        doc_norm (dict(int : float)): a map from doc_ids to pre-computed document norms
        num_docs (int): number of documents in the corpus

    Returns:
        list(tuple(int, float)): a list of document ids and the similarity scores with the query
        sorted so that the most similar documents to the query are at the top.
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the run_query function in query.py
    #       but should use tfidf instead of term frequency

    # pre-process the query string
    qt = get_query_tokens(query_string)
    query_token_counts = count_query_tokens(qt)

    query_vector = {}
    query_norm = 0.0

    for term, tf in query_token_counts:
        if term in doc_freq:
            idf = math.log10(num_docs / doc_freq[term])
            weight = tf * idf

            query_vector[term] = weight

            query_norm += weight ** 2

    query_norm_final = math.sqrt(query_norm)

    if query_norm_final == 0:
        return []

    # calc dot products between query and documents
    doc_scores = defaultdict(float)

    for term, query_weight in query_vector.items():
        postings_list = index.get(term, [])

        idf = math.log10(num_docs / doc_freq[term])

        for docid, doctf in postings_list:
            doc_weight = doctf * idf
            doc_scores[docid] += query_weight * doc_weight

    final_scores = []
    for docid, dot_prod in doc_scores.items():
        doc_vector_norm = doc_norm.get(docid, 1.0)
        denominator = query_norm * doc_vector_norm
        if denominator > 0:
            similarity = dot_prod / denominator
            final_scores.append((docid, similarity))

    final_scores.sort(key=lambda x: x[1], reverse=True)

    return final_scores

if __name__ == '__main__':
    queries = [
        'Food Safety',
        'Ozone Layer',
    ]
    query_main(queries=queries, query_func=run_query, doc_norm_func=get_doc_to_norm)
    
    
