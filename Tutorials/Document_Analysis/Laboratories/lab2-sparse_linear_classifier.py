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
# # Movie Review Sentiment Classifier

# %% [markdown]
# In this notebook, you will implement a simple linear classifier to infer the sentiment of a movie review from its text. 
#
# You will also implement a hyper-parameter tuning method presented in the lectures to find a good value for the regularisation parameter of your logistic regression classifier. 
#
# The [scikit-learn](https://scikit-learn.org/stable/index.html) machine learning package will be used throughout this notebook.

# %%
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %% [markdown]
# Load the movie review data

# %%
df = pd.read_csv(os.path.join("data", "movie_reviews_labelled.csv"))

# %% [markdown]
# Shuffle the rows and sample a fraction of the dataset for this lab so that you don't have to wait so long for the model to train

# %%
df = df.sample(frac=0.3, random_state=1).reset_index(drop=True)

# %% [markdown]
# Split the data into training, validation and test sets.

# %%
# convert pandas series to lists
Xr = df["text"].tolist()
Yr = df["label"].tolist()

# compute the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))

# store the train val test splits
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]


# %% [markdown]
# Fit a linear classification model

# %%
def fit_model(Xtr, Ytr, C):
    """Tokenizes the sentences, calculates TF vectors, and trains a logistic regression model.
    
    Args:
    - Xtr: A list of training documents provided as text
    - Ytr: A list of training class labels
    - C: The regularization parameter
    """

    # TODO: write model fitting code using CountVectorizer and LogisticRegression
    #       CountVectorizer is used to convert the text into sparse TF vectors
    #       See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    #       LogisticRegression will train the classifier using these vectors
    #       See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    # Solution:
    count_vectoriser = CountVectorizer()
    Xv = count_vectoriser.fit_transform(Xtr)
    model = LogisticRegression(C = C, max_iter=1000)
    model.fit(Xv, Ytr)
    
    # return the model and CountVectorizer
    # Note: we need to return the CountVectorizer because 
    # it stores a mapping from words -> ids which we need for testing
    return model, count_vectoriser


# %% [markdown]
# Test a fitted linear classifier

# %%
def test_model(Xtst, Ytst, model, count_vectoriser):
    """Evaluate a trained classifier on the test set.
    
    Args:
    - Xtst: A list of test or validation documents
    - Ytst: A list of test or validation class labels
    - count_vectoriser: A fitted CountVectorizer
    """
    
    # TODO: write code to test a fitted linear model and return accuracy
    #       you will need to use count_vec to convert the text into TF vectors
    # Hint: the function accuracy_score from sklearn may be helpful
    #       See ttps://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html 

    Xv = count_vectoriser.transform(Xtst)
    ypred = model.predict(Xv)
    score = accuracy_score(Ytst, ypred)
    
    return score


# %% [markdown]
# Hyper-parameter tuning: search a good value for the hyper-parameter `C`

# %%
# TODO: search for the best C parameter by 
#       training on the training set and testing on the validation set
#       you should use fit_model and test_model

# Solution:
best_C = 0
best_score = -1
print(f'{"C":>10} {"Accuracy":>10} {"Best C":>10} {"Best Accuracy":>20}')
for k in range(-5, 5):
    C = 3**k
    model, cvec = fit_model(X_train, Y_train, C)
    score = test_model(X_val, Y_val, model, cvec)
    if score > best_score:
        best_score = score
        best_C = C
    print(f'{round(C, 4):>10} {round(score, 4):>10} {round(best_C, 4):>10} {round(best_score, 4):>20}')

# %% [markdown]
# Train your classifier using both the training and validation data, and the best value of `C`

# %%
# TODO: fit the model to the concatenated training and validation set
#       test on the test set and print the result

# Solution:
model, cvec = fit_model(Xr[:val_end], Yr[:val_end], best_C)
score = test_model(X_test, Y_test, model, cvec)
print(score)

# %% [markdown]
# Inspect the co-efficients of your logistic regression classifier

# %%
# TODO: find the words corresponding to the 5 largest (most positive) and 
#       5 smallest (most negative) co-efficients of the linear model
# Hint: a fitted LogisticRegression model in sklearn has a coef_ attribute which stores the co-efficients
#       CountVectorizer has a vocabulary_ attribute that stores a mapping of terms to feature indices

# Solution:
order = np.argsort(model.coef_)[0]
id_to_w = {v:k for k,v in cvec.vocabulary_.items()}
print("Most negative:")
for v in order[:5]:
    print(id_to_w[v])
print("\nMost positive:")
for v in order[-5:]:
    print(id_to_w[v])

# %%
