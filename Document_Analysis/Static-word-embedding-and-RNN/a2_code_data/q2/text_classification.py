import os
import nltk
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader

from features import get_features_w2v
from classifier import (
    train_model,
    eval_model,
    search_C,
)

tokenizer = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def tokenise_document(text):
    """Tokenize a string representing a document.

    Args:
        text: The input string of the document to be tokenised.

    Returns:
        list(list(str)): A list of sentences, where each
            sentence is a list of strings (tokens).
    """
    sentences = nltk.sent_tokenize(text)
    return [tokenizer.tokenize(sent) for sent in sentences]


def prepare_dataset(filename):
    """Prepare the training/validation/test dataset.

    Args:
        filename (str): The name of file from which data will be loaded.

    Returns:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.
    """
    print('Preparing train/val/test dataset ...')
    # load raw data
    df = pd.read_csv(filename, delimiter="\t")
    # shuffle the rows
    df = df.sample(frac=1, random_state=250730).reset_index(drop=True)
    # get the train, val, test splits
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
    Xr = df["text"].tolist()
    train_end = int(train_frac*len(Xr))
    val_end = int((train_frac + val_frac)*len(Xr))
    Xr_train = Xr[0:train_end]
    Xr_val = Xr[train_end:val_end]
    Xr_test = Xr[val_end:]

    # encode class labels ('A' and 'B')
    yr = df["label"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(yr)
    y_train = np.array(y[0:train_end])
    y_val = np.array(y[train_end:val_end])
    y_test = np.array(y[val_end:])
    return Xr_train, y_train, Xr_val, y_val, Xr_test, y_test


def text_classification_w2v(word_vectors, Xr_train, y_train, Xr_val, y_val, Xr_test, y_test):
    """Text classification using word2vec word vectors.

    Args:
        word_vectors (KeyedVectors): some already trained word vectors 
            stored as KeyedVectors 
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.
        word2vec_model (Word2VecModel): A trained word2vec model.

    Returns:
        float: The accuracy of the text classifier on the test set.
    """

    # convert each document in the datasets into a list of sentences,
    # where each sentence is a list of tokens.
    Xt_train = [tokenise_document(xr) for xr in tqdm(Xr_train)]
    Xt_val = [tokenise_document(xr) for xr in tqdm(Xr_val)]
    Xt_test = [tokenise_document(xr) for xr in tqdm(Xr_test)]

    # generate word2vec features for texts in the training and validation sets 
    X_train = get_features_w2v(Xt_train, word_vectors)
    X_val = get_features_w2v(Xt_val, word_vectors)

    # search for the best C value
    print('Searching for the optimal hyperparameter using the validation data...')
    C = search_C(X_train, y_train, X_val, y_val)

    # re-train the classifier using the training set concatenated with the
    # validation set and the best C value
    print('Re-training the text classifier using the training and validation data...')
    X_train_val = get_features_w2v(Xt_train + Xt_val, word_vectors)
    y_train_val = np.concatenate([y_train, y_val], axis=-1)

    model = train_model(X_train_val, y_train_val, C)

    # evaluate performance on the test set
    print('Classifying the test data...')
    X_test = get_features_w2v(Xt_test, word_vectors)
    acc = eval_model(X_test, y_test, model)
    return acc


if __name__ == '__main__':
    # split dataset into training, validation and test sets
    data_file = os.path.join("data", "data_file.csv")

    model_files = ["m1.model", "m2.1.model", "m3.model", "m4.model"]

    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)

    results = {}

    for model in model_files:
        print(f"--- Evaluating model: {model} ---")
        file_path = os.path.join("../q1", model)
        # uncomment the line below to load the word2vec model file
        word_vectors = Word2Vec.load(file_path).wv
    
        # uncomment the code below to predict the class label on the test set using word2vec features 
        acc = text_classification_w2v(word_vectors, Xr_train, y_train, Xr_val, y_val, Xr_test, y_test)
        # print(f'Accuracy on test set (word2vec): {acc:.4f}')
        results[model] = acc
    
    # print table of results 
    print(f'{"Model":<20} | {"Accuracy"}')
    print("-" * 31) 

    for model, accuracy in results.items():
        print(f"{model:<20} | {accuracy:.4f}")


