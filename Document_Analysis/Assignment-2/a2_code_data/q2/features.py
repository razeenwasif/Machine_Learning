import numpy as np
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

def get_document_vector(doc, word_vectors):
    """Takes a (tokenised) document and turns it into a vector by averaging
    its word vectors.

    Args:
        doc (list(list(str))): A document represented as list of
            sentences. Each sentence is a list of tokens.
        word_vectors (gensim.models.keyedvectors.KeyedVectors): A mapping 
            from words (string) to their embeddings (np.ndarray)

    Returns:
        np.array: The averaged word vector representing the input document.
    """
    # check the input
    assert isinstance(word_vectors, KeyedVectors)

    # TODO: convert each document into a vector using the average of
    #       the embedding vectors of the words in the document.



def get_features_w2v(Xt, word_vectors):
    """Given a dataset of (tokenised) documents (each represented as a list of
    tokenised sentences), return a (dense) matrix of aggregated word vector for
    each document in the dataset.

    Args:
        Xt (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        word_vectors (gensim.models.keyedvectors.KeyedVectors): A mapping 
            from words (string) to their embeddings (np.ndarray)

    Returns:
        np.ndarray: A matrix of features. The i-th row vector represents the i-th
            document in `Xr`.
    """
    print('Generating features (word2vec) ...')
    return np.vstack([get_document_vector(xt, word_vectors) for xt in tqdm(Xt)])

