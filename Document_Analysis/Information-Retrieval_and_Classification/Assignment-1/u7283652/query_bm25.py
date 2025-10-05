
import os
import math
import pickle
from collections import defaultdict

from query import (
    get_query_tokens,
    count_query_tokens,
)


def get_doc_to_length(index, doc_freq, num_docs):
    """Pre-compute the length for each document in the corpus and the average document length.

    Args:
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        num_docs (int): number of documents in the corpus

    Returns:
        dict(int: float): a dictionary mapping doc_ids to document lengths
        float: the average document length
    """

    # TODO: Modify the code below to implement this function properly.

    doc_length = defaultdict(float)
    total_length = 0.0 

    for term in index:
        postings_list = index[term]
        for docid, term_freq in postings_list:
            # add freq of this term to doc's total len 
            doc_length[docid] += term_freq 

    total_length = sum(doc_length.values())

    avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0 

    return doc_length, avg_doc_length


def run_query(query_string, index, doc_freq, doc_length, avg_doc_length, num_docs):
    """ Run a query on the index and return a sorted list of documents. 
    Sorted by most similar to least similar.
    Documents not returned in the sorted list are assumed to have 0 similarity.

    Args:
        query_string (str): the query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        doc_length (dict(int : float)): a map from doc_ids to pre-computed document lengths
        avg_doc_length (float): the average document length
        num_docs (int): number of documents in the corpus

    Returns:
        list(tuple(int, float)): a list of document ids and the similarity scores with the query
        sorted so that the most similar documents to the query are at the top.
    """

    # TODO: Modify the code below to implement the Okapi BM25 retrieval function.

    # These are the default values of the parameters k1 and b. You do not need to change them.
    k1 = 1.5
    b = 0.75

    # get query term freq 
    query_tokens = get_query_tokens(query_string)
    query_tf = count_query_tokens(query_tokens)

    # init a dict to accumulate scores for e/a doc 
    doc_scores = defaultdict(float)

    # iter through e/a term in the query 
    for term, query_term_freq in query_tf:
        if term in doc_freq:
            idf = math.log(num_docs / doc_freq[term])
            postings_list = index.get(term, [])

            for docid, doc_term_freq in postings_list:
                numerator = doc_term_freq * (k1 + 1)
                doc_len = doc_length.get(docid, 0)
                denominator = doc_term_freq + k1 * (1 - b + b * (doc_len / avg_doc_length))
                term_score = idf * (numerator / denominator)

                doc_scores[docid] += term_score 

    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_docs

def query_main(queries=None):
    """Run all the queries in the evaluation dataset (and the specific queries if given)
    and store the result for evaluation.

    Args:
        queries (list(str)): a list of query strings (optional)
    """

    # load the index from disk
    (index, doc_freq, doc_ids, num_docs) = pickle.load(open("my_index.pkl", "rb"))

    # compute doc norms (in practice we would want to store this on disk, for
    # simplicity in this assignment it is computed here)
    (doc_lengths, avg_doc_length) = get_doc_to_length(index, doc_freq, num_docs) 

    # get a reverse mapping from doc_ids to document paths
    ids_to_doc = {docid: path for (path, docid) in doc_ids.items()}

    # if a list of query strings are specified, run the query and output the top ranked documents
    if queries is not None and len(queries) > 0:
        for query_string in queries:
            print(f'Query: {query_string}')
            res = run_query(query_string, index, doc_freq, doc_lengths, avg_doc_length, num_docs)
            print('Top-5 documents (similarity scores):')
            for (docid, sim) in res[:5]:
                print(f'{ids_to_doc[docid]} {sim:.4f}')

    # run all the queries in the evaluation dataset and store the result for evaluation
    result_strs = []
    with open(os.path.join('gov', 'topics', 'gov.topics'), 'r') as f:
        for line in f:
            # read the evaluation query
            terms = line.split()
            qid = terms[0]
            query_string = " ".join(terms[1:])

            # run the query
            res = run_query(query_string, index, doc_freq, doc_lengths, avg_doc_length, num_docs)

            # write the results in the correct trec_eval format
            # see https://trec.nist.gov/data/terabyte/04/04.guidelines.html
            for rank, (docid, sim) in enumerate(res):
                result_strs.append(f"{qid} Q0 {os.path.split(ids_to_doc[docid])[-1]} {rank+1} {sim} MY_IR_SYSTEM\n")

    with open('retrieved_documents.txt', 'w') as fout:
        for line in result_strs:
            fout.write(line)


if __name__ == '__main__':
    query_main()
    
