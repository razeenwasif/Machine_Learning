import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import string
import re
import csv

# punctuation marks
#trans_table = str.maketrans(dict.fromkeys(string.punctuation))

# get the nltk stopwords list
#stopwords = set(nltk.corpus.stopwords.words("english"))

def setup_nltk():
    """Downloads necessary NLTK models if not already present"""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt', quiet=True)
    try: 
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'stopwords' model...")
        nltk.download('stopwords', quiet=True)

def preprocess_tokens(tokens):
    """Applies a full pre-processing pipeline to a list of tokens"""
    # prepare tools
    stemmer = PorterStemmer()
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Create a translation table to remove punctuation
    trans_table = str.maketrans(dict.fromkeys(string.punctuation))

    processed_tokens = []
    for token in tokens:
        # remove punct 
        token_no_punct = token.translate(trans_table)
        # covnert to lowercase
        token_lower = token_no_punct.lower()

        # filter out stopwords and non-alphabetic tokens 
        if token_lower and token_lower not in stopwords and token_lower.isalpha():
            # stem the tok 
            stemmed_tok = stemmer.stem(token_lower)
            processed_tokens.append(stemmed_tok)

    return processed_tokens 

def analyze_document_classes(filepath='./data/data_file.csv'):
    """
    Reads a Tab-Separated (TSV) file, processes the text for two classes 
    (A and B), and prints the most common words for each class.
    """
    print(f"Parsing '{filepath}' as a Tab-Separated file (TSV)...")
    try:
        # The key fix is here: sep='\t' tells pandas to split columns on tabs.
        # We also use 'on_bad_lines' to handle potential errors in the file gracefully.
        # The 'engine='python'' parameter can help with complex parsing cases.
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='warn', engine='python')

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- From here on, the function should work perfectly ---

    print("Successfully loaded data.")
    print(f"Total documents parsed: {len(df)}")
    print("Column names found:", df.columns.tolist()) # Debug: show column names
    print(df['label'].value_counts())
    print("-" * 30)

    docs_A = df[df['label'] == 'A']['text']
    docs_B = df[df['label'] == 'B']['text']

    all_tokens_A = []
    all_tokens_B = []

    print("Processing documents for Class A...")
    for doc in docs_A:
        tokens = nltk.word_tokenize(doc)
        all_tokens_A.extend(preprocess_tokens(tokens))

    print("Processing documents for Class B...")
    for doc in docs_B:
        tokens = nltk.word_tokenize(doc)
        all_tokens_B.extend(preprocess_tokens(tokens))

    freq_A = Counter(all_tokens_A)
    freq_B = Counter(all_tokens_B)
    
    print("\n" + "="*40)
    print("      CORPUS STATISTICS SUMMARY")
    print("="*40)
    
    print("\n--- Top 20 Most Common Words in Class A ---")
    for word, count in freq_A.most_common(20):
        print(f"{word:<15} | Count: {count}")

    print("\n--- Top 20 Most Common Words in Class B ---")
    for word, count in freq_B.most_common(20):
        print(f"{word:<15} | Count: {count}")
        
    print("\n" + "="*40)
    print("Based on these word lists, you can infer the topics of each class.")
    

if __name__ == '__main__':
    # Ensure NLTK packages are available
    setup_nltk()
    # Run the analysis 
    analyze_document_classes()
