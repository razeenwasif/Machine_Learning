import nltk
from nltk.stem import PorterStemmer
import string

def process_tokens(toks):
    # TODO: 
    # Implement function process_tokens_new and uncomment it below
    # to test your new tokenisation method.
    # Make sure to rebuild the index.

    return process_tokens_new(toks)
    # return process_tokens_original(toks)


def process_tokens_original(toks):
    """ Perform processing on tokens. This function simply turns the
        input tokens into lowercase without any further processing.

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """

    new_toks = []
    for t in toks:
        t = t.lower()
        new_toks.append(t)
    return new_toks


# create a stemmer
stemmer = PorterStemmer()
# define a function to perform stemming using the Porter stemmer
from functools import lru_cache
@lru_cache(maxsize=None)
def stemword(word):
    """
    Perform stemming on the input word.

    Arg:
        word (str): an input word

    Returns:
        str: the stem of the input word obtained using the Porter Stemmer

    """
    return stemmer.stem(word)


# create a translation table to be used by str.translate() to remove 
# punctuation marks
trans_table = str.maketrans(dict.fromkeys(string.punctuation))

# get the nltk stopwords list
stopwords = set(nltk.corpus.stopwords.words("english"))

def process_tokens_new(toks):
    """ Perform processing on tokens.

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """

    #TODO: Modify the code below to include the following 
    # text pre-processing steps:
    # * stemming
    # * removal of punctuation marks
    # * removal of stop words
    # You should perform these steps in a reasonable order.

    return [stemword(t_lower) 
            for t in toks 
            if (t_lower := t.translate(trans_table).lower())]

#    return [stemword(tok_lower) 
#            for t in toks 
#            if (tok_lower := t.translate(trans_table).lower()) 
#            and tok_lower not in stopwords]


def tokenize_text(data):
    """Convert a document as a string into a document as a list of
    tokens. The tokens are strings.

    Args:
        data (str): The input document.

    Returns:
        list(str): The list of tokens converted from the input document.
    """
    # simply split the text by white space characters
    tokens = data.split()
    # further process the tokens
    tokens = process_tokens(tokens)
    return tokens
