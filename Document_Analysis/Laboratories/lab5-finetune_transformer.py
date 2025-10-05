# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fine-tuning a Pre-trained Transformer for Document Classification

# %% [markdown]
# In this lab you will fine-tune a pre-trained transformer model using the huggingface `transformers` library. This library provides a number of transformer models such as BERT, XLNet, and GPT, that can be used with PyTorch or Tensorflow. The tokenisers for these models are also included, which makes using transformers with this library much easier than developing them from scratch.

# %% [markdown]
# The [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) which is a smaller version of BERT will be used in this lab. It is much faster to fine-tune, but will give slightly worse performance than the original BERT model.

# %% [markdown]
# For the dataset we will use the [IMDB movie review data](https://huggingface.co/datasets/imdb) where the task is to classify a review as either positive if the reviewer liked the movie or negative if the reviewer did not like the movie. The input is the text of the review and the output is a binary label either 0 (negative) or 1 (positive).
# In previous labs and the assignments, you have explored this dataset multiple times but with a differnt train/validation/test splits. 

# %% [markdown]
# We will use both `pytorch` and the `transformers` library, as well as a few other useful libraries such as `tqdm` to make progress bars, and `sklearn` for its evaluation metric.
#
# We will also use the `datasets` library which is the huggingface datasets library and can be installed using `pip`.

# %%
# Uncomment the code below to install the packages needed.

# # %pip install transformers datasets ipywidgets

# %%
import torch
from torch.utils.data import DataLoader, dataloader
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import numpy as np
from pprint import pprint

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# ## Dataset
#
# First, load the `imdb` data using the huggingface `datasets` library, which will automatically download the data if it is not already cached on your system.

# %%
data = load_dataset("imdb")

# %% [markdown]
# If you print the `data` variable you will see that you have a dictionary like object with keys `test`, `train`, `unsupervised`, and values of `TorchIterableDataset`. We will only use the `test` and `train` data for this lab. You can inspect the first element of the data as follows:

# %%
pprint(data['train'][0])

# %% [markdown]
# If you do this you will get a dictionary with both the `label` and the `text` of the first element. The label indicates if this text is a positive or negative review.

# %% [markdown]
# ### Tokeniser
#
# Since the `text` is currently a string of characters we need to:
# 1. Split the sequence into tokens (i.e. word-pieces for BERT models but will be different depending on the pre-trained model)
# 2. Represent the sequence as token/word-piece ids
# 3. Add a `[CLS]` token to the start of the sequence, and a `[SEP]` token to the end
# 4. Pad the sequences with `0`'s so they are all of the same length
# 5. Construct an attention mask for the input (to identify which parts of the input are padding and so should be ignored)
#
# Fortunately, the `transformers` library provides an easy way to do all of this.

# %% [markdown]
# First, get the tokeniser specific to DistilBert ([documentation here](https://huggingface.co/docs/transformers/model_doc/distilbert)). \
# This will download the tokeniser for the lowercase only version of DistilBert. The download is only a few kilobytes. 

# %%
model_name = "distilbert-base-uncased"
tok = DistilBertTokenizerFast.from_pretrained(model_name)


# %% [markdown]
# Next, write a function to apply the tokeniser to the `text` field in the dataset.
#
# This will do all the steps 1-5 listed above and return the attention mask and the sequence of ids. The arguments indicate we are truncating sequences longer than the maximum length, and padding all sequences that are less than the maximum length so that they are exactly the maximum length. The maximum length for the DistilBert model is 512 word-pieces.

# %%
def tokenize_fn(X):
    return tok(X["text"], truncation = True, padding="max_length")


# %% [markdown]
# ### Training data

# %% [markdown]
# Now we can apply the tokeniser to the text as follows:

# %%
shuffle_train_data = data['train'].shuffle(seed=42)
tokenized_train_data = shuffle_train_data.map(tokenize_fn, batched=True)
small_train_data = tokenized_train_data.select(range(1000))
small_train_data = small_train_data.with_format("torch", columns=["label", "input_ids", "attention_mask"])

# %% [markdown]
# The part which actually applies the tokeniser is the `map` call. This will call the `tokenize_fn` function on each of the examples in `data['train']`.
#
# The `shuffle` ensures the data is in a random order (if you don't do this you will run into problems because the dataset has all the 0 classes first followed by all the 1 classes).
#
# The `select` statement just extracts the first 1000 examples (of the shuffled list), if you were training this model to get the best performance you would use all the examples but for this lab we will use only 1000 to avoid waiting around for the model to train.
#
# The final line which calls `with_format` is responsible for converting the data columns `label`, `input_ids`, and `attention_mask` into PyTorch tensors.

# %% [markdown]
# It is worthwhile to inspect `small_train_data` at this point. To do this use:

# %%
pprint(small_train_data[0])

# %% [markdown]
# You will see that the datapoint has an `attention_mask`, `input_ids`, and `label`. The `attention_mask` and `input_ids` are new and were added by `tokenize_fn`.
#
# The `input_ids` are the padded sequences of word-piece ids, while the `attention_mask` identifies which parts of the `input_ids` are padding and so should be ignored by the attention layer. 
#
# Note that the first id in the tensor of `input_ids` is `101`, which represents the `[CLS]` token, while the last non-zero id is `102` which is the `[SEP]` token. If you want to explore a bit further the mapping from tokens to ids can be accessed through the dictionary `tok.vocab`.

# %% [markdown]
# ### Validation data

# %% [markdown]
# We can now create a validation dataset from the `test` data in the same way:

# %%
shuffle_val_data = data['test'].shuffle(seed=42)
tokenized_val_data = shuffle_val_data.map(tokenize_fn, batched=True)
small_val_data = tokenized_val_data.select(range(1000))
small_val_data = small_val_data.with_format("torch", columns=["label", "input_ids", "attention_mask"])

# %% [markdown]
# ## Model
# Defining and downloading the pre-trained transformer is straightforward but be aware that the download is ~270M:

# %%
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# %% [markdown]
# The above specifies downloads and sets-up a pre-trained `DistilBert` model and configures it for the sequence classification task. 
#
# The download is specifically the [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) pre-trained model, which only considers lowercase characters (there is a cased version available also `distilbert-base-cased`). 
#
# Under the hood, the huggingface sequence classifier model will feed the output of the transformer (the output value at the position of the `[CLS]` token) to a new Linear layer to get logits (un-normalised scores for each of the classes). We need to provide the number of classes so the linear layers weight matrix can be properly specified. The way to do this is with the `num_labels` argument. As there are only two classes in our task (positive and negative) we specify `2` as the number of labels.

# %% [markdown]
# The loaded `DistilBert` model is in the variable `model` which is also a PyTorch module with the normal set of PyTorch methods, such as `forward`, `parameters`, and `to`.
#
# We first move the model to the desired device such as the GPU (if available) using `model.to(device)`.

# %%
model.to(device)


# %% [markdown]
# ### Fine-tuning

# %% [markdown]
# We initialize an `AdamW` optimiser, and then start the main training loop. Which consists of a model forward pass, followed by a backward pass and optimiser step. 
#
# Note that the model takes a number of arguments:
# - `labels` (optional) which are the ground truth labels (for calculating the loss function when training),
# - `input_ids` which are the padded sequences of word-pieces, and
# - `attention_mask` which is an binary mask indicating which parts of the inputs are actual tokens and which are padding tokens (we do not want the transformer to pay attention to any padding tokens);
#
# and returns a [SequenceClassifierOutput](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) object.

# %%
def finetune_model(model, dataset_train, dataset_val=None, eval_fn=None, batch_size=8, n_epochs=2, learning_rate=1e-5):
    model.train(True)
    
    # create a pytorch data loaders to make it easy to iterate over batches of training data
    # see https://pytorch.org/docs/stable/data.html
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    n_batches = (len(dataset_train) - 1) // batch_size + 1
    
    # get the AdamW optimizer
    optimiser = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    # run the training loop
    print(f'{"Epoch":>10} {"Batch":>20} {"Loss":>10}')
    for epoch in range(n_epochs):
        for (b, batch) in enumerate(tqdm(dataloader)):
    
            # run the transformer forwards
            outputs = model(
                labels = batch["label"].to(device),
                input_ids = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
    
            # get the classification loss
            loss = outputs.loss
            
            # backpropagate then apply the optimiser
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    
            # print the loss
            if (b+1) % 5 == 0:
                print(f'{epoch+1:>10} {f"{b+1} / {n_batches}":>20} {format(loss.cpu().item(), ".3f"):>10}')

        # evaluate the model on validation data
        if (dataset_val is not None) and (eval_fn is not None):
            print('Evaluating ...')
            f1 = eval_fn(model, dataset_val)
            print(f'Epoch {epoch+1}: F1 score (validation) = {format(f1, ".3f")}') 

    return model


# %% [markdown]
# ### Exercise
#
# Implement a `evaluate_model` function which returns the F1 score of the model on the validation data. This is going to be similar to the evaluation functions you have seen in previous labs/assignments and also reasonably similar to the training loop provided above.

# %%
@torch.no_grad()
def evaluate_model(model, dataset_val, batch_size=8):
    # TODO: Implement this function which returns the F1 score of the model on the validation data.
    model.eval() # model.train(False)
    dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False) 
    predictions = [] 
    true_labels = []
    # iterate through the dataloader in batches 
    for (b, batch) in enumerate(tqdm(dataloader)):
        # get the data point within the batch 
        outputs = model(
            input_ids = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
        )
 
        logits = outputs.logits 

        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch["label"].cpu().numpy())
    
    model.train(True)
    return f1_score(true_labels, predictions, average="binary")

# %% [markdown]
# Fine tune the model and evaluate it on the validation set after each training epoch (this may take ~30 minutes if using CPU).

# %%
model = finetune_model(model, small_train_data, small_val_data, eval_fn=evaluate_model, n_epochs=2)

# %%
