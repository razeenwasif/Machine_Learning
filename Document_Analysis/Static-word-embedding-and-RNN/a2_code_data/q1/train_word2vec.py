from gensim.models import Word2Vec
import os
import nltk
import string
import numpy as np
from scipy.spatial.distance import cosine
from itertools import product

tokenizer = nltk.tokenize.TreebankWordTokenizer()
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
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().translate(trans_table)  # remove punctuation
                if not line:
                    continue

                tokens = tokenizer.tokenize(line.lower())  # lowercase + tokenize
                
                if remove_stopwords:
                    tokens = [word for word in tokens if word not in stopwords]

                sentences.append(tokens)

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    
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

    w2v_model = Word2Vec(
        sentences=sentences, 
        vector_size=200, 
        window=window, 
        min_count=10, 
        negative=10, 
        seed=seed, 
        workers=1, 
        epochs=5
    )
    
    return w2v_model


if __name__ == '__main__':

    file_name = os.path.join("data", "w2v_training_data.txt")
   
    # define model configurations
    model_configs = {
        "m1": {"window":2, "remove_stopwords":False, "seed":1},
        "m2": {"window":2, "remove_stopwords":False, "seed":2},
        "m2.1": {"window":2, "remove_stopwords":True, "seed":1},
        "m3": {"window":5, "remove_stopwords":False, "seed":1},
        "m4": {"window":5, "remove_stopwords":True, "seed":1},      
    }

    for name, config in model_configs.items():
        print(f"\\n--- Training Model {name.upper()} ---")

        # Process the data
        sentences = process_training_data(
            file_name, remove_stopwords=config["remove_stopwords"]
        )
        print(f"Using {len(sentences):,d} sentences.(Stopwords removed: {config['remove_stopwords']})")

        # Train the model 
        print(f"Training with window={config['window']} and seed={config['seed']}...")
        w2v_model = train_model(
            sentences, window=config["window"], seed=config["seed"]
        )

        # save the model 
        save_path = f"{name}.model"
        w2v_model.save(save_path)
        print(f"Model saved as {save_path}")

    print("\\nAll required models have been trained.")

    ########################################################################

    print("\\n--- Starting Analysis for Q1 ---")

    # Obtain the vectors for 'baseball', 'basketball', and 'computer' from the trained model.
    # Compute and display the cosine similarities between 'baseball' and 'basketball' and between 'baseball' and 'computer'.
    try:
        print("Loading all models...")
        m1_model = Word2Vec.load("m1.model")
        m2_model = Word2Vec.load("m2.model")
        m3_model = Word2Vec.load("m3.model")
        m4_model = Word2Vec.load("m4.model")
        print("Models loaded successfully")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load models. {e}")
        exit()

    ############# --- Part (A) and (B) calculations --- ###################
    try:
        sim_bb_m1 = m1_model.wv.similarity("baseball", "basketball")
        sim_bc_m1 = m1_model.wv.similarity("baseball", "computer")
        print(f"\\n[Q1.A] M1 Similarity 'baseball' vs 'basketball': {sim_bb_m1:.4f}")
        print(f"[Q1.A] M1 Similarity 'baseball' vs 'computer': {sim_bc_m1:.4f}")
    except KeyError as e:
         print(f"A word was not in M1's vocabulary: {e}")

    try:
        sim_bb_m2 = m2_model.wv.similarity("baseball", "basketball")
        print(f"\\n[Q1.B] M2 Similarity 'baseball' vs 'basketball': {sim_bb_m2:.4f}")
    except KeyError as e:
        print(f"A word was not in M2's vocab: {e}")

    ################ --- Part (C) calculations --- #######################
    try:
        m1_baseball_vec = m1_model.wv["baseball"]
        m2_basketball_vec = m2_model.wv["basketball"]

        sim_m1b_m2b = 1 - cosine(m1_baseball_vec, m2_basketball_vec)
        print(f"\\n[Q1.C] Similarity between M1's 'baseball' and M2's 'basketball': {sim_m1b_m2b:.4f}")
    except KeyError as e:
        print(f"A word was not in the vocab: {e}")

    ############ --- Part (D), (E), (F) calculations --- #################
    print("\\n[Q1.D] Top 20 similar words from M1:")
    try:
        print("M1 similar to would:", 
              m1_model.wv.most_similar('would', topn=20))
        print("M1 similar to greece:", 
              m1_model.wv.most_similar('greece', topn=20))
    except KeyError as e:
        print(f"error: {e}")

    print("\\n[Q1.E] Top 20 similar words from M3:")
    try:
        print("M3 similar to would:", 
              m3_model.wv.most_similar('would', topn=20))
        print("M3 similar to greece:", 
              m3_model.wv.most_similar('greece', topn=20))
    except KeyError as e:
        print(f"error: {e}")
   
    print("\\n[Q1.F] Top 20 similar words from M4:")
    try:
        print("M4 similar to would:", 
              m4_model.wv.most_similar('would', topn=20))
        print("M4 similar to greece:", 
              m4_model.wv.most_similar('greece', topn=20))
    except KeyError as e:
        print(f"error: {e}")
