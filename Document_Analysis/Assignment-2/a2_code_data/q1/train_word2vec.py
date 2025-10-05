from gensim.models import Word2Vec
import os
import nltk
import string
import numpy as np

tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))


def process_training_data(file_name, remove_stopwords=False):
    """
    Processes a text file that contains a collection of sentences.
    Performs tokenisation and optionally stop word removal.
    Returns a list of sentences, where each sentence is a list of tokens.

    Args:
        file_name (str): path to a text file, where each line of the file is a sentence
        remove_stopwords (bool): True if stop words are to be removed and False otherwise

    Returns:
        list(list(str)): list of sentences, where each sentence is a list of tokens
    """

    sentences = []

    # TODO: Process the input file to turn it into a list of sentences.
    # Hint: Use the tokeniser defined above or write your own tokeniser.




    return sentences



def train_model(sentences, window, seed):
    """
    Trains a word2vec model using the given sentences and the hyperparameters given.

    Args:
        sentences (list(list(str))): training sentences
        window: the size of the context windows used to train the model
        seed: seed of the random number generator used to initialise the word embeddings

    Returns:
        the trained word2vec model (which is a Word2Vec object)
    """

    # TODO: Use the Word2Vec class from gensim to train a word2vec model and return it.
    w2v_model = None




    return w2v_model




if __name__ == '__main__':

    file_name = os.path.join("data", "w2v_training_data.txt")
    
    # TODO: Select one of the two lines below 
    sentences = process_training_data(file_name)
    # sentences = process_training_data(file_name, True)

    print(f"Training word2vec using {len(sentences):,d} sentences ...")

    # TODO: Modify the hyperparameters below to compare models trained with different configurations.
    window = 2
    seed = 1

    w2v_model = train_model(sentences, window, seed)

    # Uncomment the line below to save the trained model.
    # w2v_model.save("word2vec.model")

    # TODO: Obtain the vectors for 'baseball', 'basketball', and 'computer' from the trained model. 
    #       Compute and display the cosine similarities between 'baseball' and 'basketball' and
    #       between 'baseball' and 'computer'.



    # Uncomment the lines below to display the top-20 words most similar to 'would' and 'greece'.
    # sims = w2v_model.wv.most_similar('would', topn=20)
    # print(sims)
    # sims = w2v_model.wv.most_similar('greece', topn=20)
    # print(sims)



