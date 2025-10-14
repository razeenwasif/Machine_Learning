# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inverted Index

# %% [markdown]
# This notebook demonstrates a simple indexer that constructs inverted index from raw text.

# %% [markdown]
# The Natural Language Toolkit (NLTK) will be used for tokenisation and text preprocessing. You may implement your own tokeniser and text preprocessing functions if you want.

# %%
import nltk

# %% [markdown]
# Download the *popular* subset of NLTK data for tokeniser etc.

# %%
nltk.download('popular')

# %% [markdown]
# A set of 3 documents about the Australian National University is provided for this exercise.

# %%
docs = [
    "The Australian National University (ANU) is a public research university and member of the Group of Eight, located in Canberra, the capital of Australia. Its main campus in Acton encompasses seven teaching and research colleges, in addition to several national academies and institutes.\nANU is ranked as the fourth university in Australia by the 2024 QS World University Rankings and second in Australia in the Times Higher Education rankings. Compared to other universities in the world, it is ranked 34th by the 2024 QS World University Rankings, and 62nd by the 2023 Times Higher Education.\nEstablished in 1946, ANU is the only university to have been created by the Parliament of Australia. It traces its origins to Canberra University College, which was established in 1929 and was integrated into ANU in 1960. ANU enrols 13,329 undergraduate and 11,021 postgraduate students and employs 4,517 staff. The university's endowment stood at A$1.8 billion as of 2018.\nANU counts six Nobel laureates and 49 Rhodes scholars among its faculty and alumni. The university has educated two prime ministers and more than a dozen current heads of government departments of Australia. The latest releases of ANU's scholarly publications are held through ANU Press online.\n",
    "Calls for the establishment of a national university in Australia began as early as 1900. After the location of the nation's capital, Canberra, was determined in 1908, land was set aside for the ANU at the foot of Black Mountain in the city designs by Walter Burley Griffin. Planning for the university was disrupted by World War II but resumed with the creation of the Department of Post-War Reconstruction in 1942, ultimately leading to the passage of the Australian National University Act 1946 by the Chifley government on 1 August 1946.\nA group of eminent Australian scholars returned from overseas to join the university, including Sir Howard Florey (co-developer of medicinal penicillin), Sir Mark Oliphant (a nuclear physicist who worked on the Manhattan Project), and Sir Keith Hancock (the Chichele Professor of Economic History at Oxford). The group also included a New Zealander, Sir Raymond Firth (a professor of anthropology at LSE), who had earlier worked in Australia for some years. Economist Sir Douglas Copland was appointed as ANU's first Vice-Chancellor and former Prime Minister Stanley Bruce served as the first Chancellor. ANU was originally organised into four centres—the Research Schools of Physical Sciences, Social Sciences and Pacific Studies and the John Curtin School of Medical Research.\nThe first residents' hall, University House, was opened in 1954 for faculty members and postgraduate students. Mount Stromlo Observatory, established by the federal government in 1924, became part of ANU in 1957. The first locations of the ANU Library, the Menzies and Chifley buildings, opened in 1963. The Australian Forestry School, located in Canberra since 1927, was amalgamated by ANU in 1965.\n",
    "Canberra University College (CUC) was the first institution of higher education in the national capital, having been established in 1929 and enrolling its first undergraduate pupils in 1930. Its founding was led by Sir Robert Garran, one of the drafters of the Australian Constitution and the first Solicitor-General of Australia. CUC was affiliated with the University of Melbourne and its degrees were granted by that university. Academic leaders at CUC included historian Manning Clark, political scientist Finlay Crisp, poet A. D. Hope and economist Heinz Arndt.\nIn 1960, CUC was integrated into ANU as the School of General Studies, initially with faculties in arts, economics, law and science. Faculties in Oriental studies and engineering were introduced later. Bruce Hall, the first residential college for undergraduates, opened in 1961.\n"
]

# %% [markdown]
# ## Indexer Step 1
# Scan the provided documents in `docs` for indexable terms and produce a list of `(token, docID)` tuples.
#
# You may use a tokeniser provided by NLTK or implement your own tokeniser.
# You also need to assign a unique `docID` for each document in `docs`.

# %%
from nltk.tokenize import word_tokenize
#
# TODO: 
# produce a list of (token, docID) tuples

token_tuples = []

# %%
# Solution:
for (docid, doc) in enumerate(docs):
    token_tuples.extend([(token, docid) for token in word_tokenize(doc)])

# %% [markdown]
# Print the total number of `(token, docID)` tuples

# %%
print(f'Number of (token, docID) tuples: {len(token_tuples)}')

# %%
# uncomment the code below for sanity check
# token_tuples[:20]

# %% [markdown]
# ### Question
#
# Are all the tokens/terms in `token_tuples` useful for keyword search?
#
# What are the potential effect of additional pre-processing (e.g., removing stopwords and punctuation marks, stemming, lemmatisation, etc.)?

# %% [markdown]
# ## Indexer Step 2

# %% [markdown]
# Sort token tuples `(token, docID)` (first by `token` then by `docID`)

# %%
# TODO:
# sort token tuples

sorted_token_tuples = []

# %%
# Solution:
sorted_token_tuples = sorted(token_tuples)

# %%
# uncomment the code below for sanity check
sorted_token_tuples[:20]

# %% [markdown]
# ## Indexer Step 3

# %% [markdown]
# Construct inverted index - a Python dictionary where
# - the key is a unique token/term
# - the value is a list of `(docID, term_freq)` tuples for the token/term, here `term_freq` is the term frequency of the token/term in a document (i.e., the number of duplicated `(token, docID)` tuples)
#
# NOTE: An efficient implementation should scan each `(token, docID)` tuple in `sorted_token_tuples` only once!

# %%
# TODO:
# construct inverted index using the sorted list of (token, docID) tuples

index = dict()

# %%
# Solution:
doc_freq = dict()
for (token, docid) in sorted_token_tuples:
    if token not in index:
        index[token] = [(docid, 1)]
        doc_freq[token] = 1
    else:
        docid_, tf = index[token][-1]
        if docid_ == docid:
            index[token][-1] = (docid, tf+1)
        else:
            index[token].append((docid, 1))
            doc_freq[token] += 1

# %% [markdown]
# Print the total number of unique tokens indexed by your system.

# %%
print(f'Number of indexed tokens/terms: {len(index)}')

# %% [markdown]
# ### Question
#
# How do you efficiently compute the document frequency of a token/term using the constructed inverted index?
