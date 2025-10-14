import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer


tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def tokenise_text(str_):
    """Tokenize a string of text.

    Args:
        str_: The input string of text.

    Returns:
        list(str): A list of tokens.
    """
    # for simplicity, remove non-ASCII characters
    str_ = str_.encode(encoding='ascii', errors='ignore').decode()
    return [t for t in tokeniser.tokenize(str_.lower().translate(trans_table)) if t not in stopwords]


def get_features_tfidf(Xr_fit, Xr_pred=None):
    """Given the training documents, each represented as a string,
    return a sparse matrix of TF-IDF features.

    Args:
        Xr_fit (iterable(str)): The input documents, each represented
            as a string.
        Xr_pred (iterable(str)): Optional input documents, each 
            represented as a string. Documents in Xr_pred should NOT
            be used to compute the IDF (which should be computed using
            documents in Xr_fit).
    Returns:
        X_fit: A sparse matrix of TF-IDF features of documents in Xr_fit.
        X_pred: Optional. A sparse matrix of TF-IDF features of documents
            in Xr_pred if it is provided.
    """
    # TODO: compute the TF-IDF features of the input documents.
    #   You may want to use TfidfVectorizer in the scikit-learn package,
    #   see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    print('Generating features (TF-IDF) ...')

    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenise_text)

    X_fit = tfidf_vectorizer.fit_transform(Xr_fit)
    X_pred = None 

    if Xr_pred is not None:
        X_pred = tfidf_vectorizer.transform(Xr_pred)

    return X_fit if Xr_pred is None else (X_fit, X_pred)

