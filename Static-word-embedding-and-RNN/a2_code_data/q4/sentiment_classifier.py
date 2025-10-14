import os
import nltk
import string
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from bisect import bisect_right
from collections import Counter
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import math


# DEVICE = torch.device('cpu')  # change this if you want to use GPU

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use Apple GPU
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Fallback to CPU
print("Using device:", DEVICE)


PAD_IDX = 0
CLS_TOKEN = '[CLS]'  # the special [CLS] token to be prepended to each sequence
SEED = 101

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
    df = pd.read_csv(filename)
    # shuffle the rows
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # get the train, val, test splits
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
    Xr = df["text"].tolist()
    train_end = int(train_frac*len(Xr))
    val_end = int((train_frac + val_frac)*len(Xr))
    Xr_train = Xr[0:train_end]
    Xr_val = Xr[train_end:val_end]
    Xr_test = Xr[val_end:]

    # encode sentiment labels ('pos' and 'neg')
    yr = df["label"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(yr)
    y_train = np.array(y[0:train_end])
    y_val = np.array(y[train_end:val_end])
    y_test = np.array(y[val_end:])
    return Xr_train, y_train, Xr_val, y_val, Xr_test, y_test


def build_vocab(Xt, min_freq=1):
    """Create a list of sentences, build the vocabulary and compute word frequencies from the given text data.

    Args:
        Xr (iterable(str)): A list of strings each representing a document.
        min_freq: The minimum frequency of a token that will be kept in the vocabulary.

    Returns:
        vocab (dict(str : int)): A dictionary mapping a word/token to its index.
    """
    print('Building vocabulary ...')
    counter = Counter()
    for xt in Xt:
        counter.update(xt)
    sorted_token_freq_pairs = counter.most_common()

    # find the first index where freq=min_freq-1 in sorted_token_freq_pairs using binary search/bisection
    end = bisect_right(sorted_token_freq_pairs, -min_freq, key=lambda x: -x[1])
    vocab = {token: idx+PAD_IDX+1 for (idx, (token, freq)) in enumerate(sorted_token_freq_pairs[:end])}  # PAD_IDX is reserved for padding
    vocab[CLS_TOKEN] = len(vocab) + PAD_IDX

    print(f'Vocabulary size: {len(vocab)}')
    return vocab


class MovieReviewDataset(Dataset):
    """A Dataset to be used by a data loader.
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    def __init__(self, X_all, y_all, cls_idx, max_seq_len):
        # X_all, y_all are the labelled examples
        # cls_idx is the index of token '[CLS]' in the vocabulary
        # max_seq_len is the maximum length of a sequence allowed
        self.X_all = X_all
        self.y_all = y_all
        self.cls_idx = cls_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        # prepend the index of the special token '[CLS]' to each sequence
        x = [self.cls_idx] + self.X_all[idx]
        # truncate a sequence if it is longer than the maximum length allowed
        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]
        return x, self.y_all[idx]


def collate_fn(batch):
    """Merges a list of samples to form a mini-batch for model training/evaluation.
    To be used by a data loader. See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    Xb = pad_sequence([torch.tensor(x, dtype=torch.long) for (x, _) in batch], padding_value=PAD_IDX)
    yb = torch.tensor([y for (_, y) in batch], dtype=torch.float32)
    return Xb.to(DEVICE), yb.to(DEVICE)


def get_positional_encoding(emb_size, max_seq_len):
    """Compute the positional encoding.

    Args:
        emb_size (int): the dimension of positional encoding
        max_seq_len (int): the maximum allowed length of a sequence

    Returns:
        torch.tensor: positional encoding, size=(max_seq_len, emb_size)
    """
    PE = torch.zeros(max_seq_len, emb_size)
    # Create the position tensor t 
    t = torch.arange(0, max_seq_len).unsqueeze(1)
    # create the division term 
    two_i = torch.arange(0, emb_size, 2) 
    div_term = torch.exp(two_i * (-math.log(10000.0) / emb_size))
    PE[:, 0::2] = torch.sin(t * div_term)
    PE[:, 1::2] = torch.cos(t * div_term)
    return PE


class SentimentClassifier(nn.Module):
    """A movie review sentiment classifier using transformers."""
    def __init__(self, vocab_size, emb_size=128, ffn_size=128, num_tfm_layer=2, num_head=2, p_dropout=0.2, max_seq_len=300):
        """SentimentClassifier initialiser.
        Args:
            vocab_size (int): the size of vocabulary
            emb_size (int): the dimension of token embedding (and position encoding)
            ffn_size (int): the dimension of the feedforward network model in a transformer encoder layer
            num_tfm_layer (int): the number of transformer encoder layers
            p_dropout (float): the dropout probability (to be used in a transformer encoder layer as well as the dropout
                layer of this class.
            max_seq_len (int): the maximum allowed length of a sequence
        """
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        
        # registers the positional encoding so that it is saved with the model
        # see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer(
            "positional_encoding", get_positional_encoding(emb_size, max_seq_len), persistent=False
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(emb_size, 1)  # for binary classification

        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=num_head, 
            dim_feedforward=ffn_size, dropout=p_dropout
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_tfm_layer)

    def forward(self, x):
        """The forward function of SentimentClassifier.
        x: a (mini-batch) of samples, size=(SEQUENCE_LENGTH, BATCH_SIZE)
        """
        # get token embeddings 
        embeddings = self.token_embeddings(x) 

        # add positional encodings 
        # get sequence length l 
        S = x.shape[0]
        embeddings = embeddings + self.positional_encoding[:S].unsqueeze(1)

        # Apply dropout 
        embeddings = self.dropout(embeddings)

        # pass through the transformer 
        transformer_output = self.encoder(embeddings)
        
        # Extract the [CLS] Token's Output
        cls_output = transformer_output[0]

        # Final linear layer 
        logits = self.linear(cls_output)
        logits = logits.squeeze(1)

        return logits

def eval_model(model, dataset, batch_size=64):
    """Evaluate a trained SentimentClassifier.

    Args:
        model (SentimentClassifier): a trained model
        dataset (MovieReviewDataset): a dataset of samples
        batch_size (int): the batch_size

    Returns:
        float: The accuracy of the model on the provided dataset
    """
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        preds = []
        targets = []
        for (Xb, yb) in tqdm(dataloader):
            out = model(Xb)
            preds.append(out.cpu().numpy() > 0)
            targets.append(yb.cpu().numpy())

        score = accuracy_score(np.concatenate(targets), np.concatenate(preds).astype(np.int32))
        
    model.train()
    return score


def train_model(model, dataset_train, dataset_val, batch_size=64, num_epoch=1, learning_rate=0.001, fmodel='best_model.pth'):
    """Train a SentimentClassifier.

    Args:
        model (SentimentClassifier): a model to be trained
        dataset_train (MovieReviewDataset): a dataset of samples (training set)
        dataset_val (MovieReviewDataset): a dataset of samples (validation set)
        batch_size (int): the batch_size
        num_epoch (int): the number of training epochs
        learning_rate (float): the learning rate
        fmodel (str): name of file to save the model that achieves the best accuracy on the validation set

    Returns:
        SentimentClassifier: the trained model
    """
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()  # the binary cross entropy loss using logits
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_batch = (len(X_train) - 1) // batch_size + 1
    print(f'{"Epoch":>10} {"Batch":>10} {"Train loss (running avg.)":>20}')
    
    # see https://pytorch.org/tutorials/beginner/saving_loading_models.html
    
    best_val_acc = -1.0 

    dataloader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    for epoch in range(num_epoch):
        for (Xb, yb) in tqdm(dataloader):
            optimiser.zero_grad()
            outputs = model(Xb) # forward pass 
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimiser.step()
        acc = eval_model(model, dataset_val)
        if acc > best_val_acc: 
            best_val_acc = acc 
            torch.save(model.state_dict(), fmodel)
    
    model.load_state_dict(torch.load(fmodel))
    return model


if __name__ == '__main__':
    torch.manual_seed(SEED)
    data_file = os.path.join("data", "movie_reviews_labelled.csv")
    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)

    get_tokenised_docs = lambda Xr: [tokenise_text(xr) for xr in tqdm(Xr)]
    get_token_indices = lambda Xt, vocab: [[vocab[token] for token in xt if token in vocab] for xt in Xt]

    Xt_train, Xt_val, Xt_test = [get_tokenised_docs(Xr) for Xr in [Xr_train, Xr_val, Xr_test]]
    vocab = build_vocab(Xt_train + Xt_val, min_freq=5)
    X_train, X_val, X_test = [get_token_indices(Xt, vocab) for Xt in [Xt_train, Xt_val, Xt_test]]
    
    max_seq_len = 500
    cls_idx = vocab[CLS_TOKEN]
    dataset_train = MovieReviewDataset(X_train, y_train, cls_idx, max_seq_len)
    dataset_val = MovieReviewDataset(X_val, y_val, cls_idx, max_seq_len)
    dataset_test = MovieReviewDataset(X_test, y_test, cls_idx, max_seq_len)

    # Note that we do not directly use the combined training set and validation set to re-train the model
    # we use this strategy for simplicity, 
    # see Section 7.8 in the deep learning textbook for other possible options
    # https://www.deeplearningbook.org/contents/regularization.html

    clf = SentimentClassifier(
        len(vocab),
        emb_size = 300,
        ffn_size = 512,
        num_tfm_layer = 2,
        num_head = 4,
        p_dropout = 0.5,
        max_seq_len = max_seq_len,
    ).to(DEVICE)

    fmodel = 'best_model.pth'
    clf = train_model(clf, dataset_train, dataset_val, batch_size=32, num_epoch=20, learning_rate=3e-4, fmodel=fmodel)
    print(clf)

    print(f'Loading model from {fmodel} ...')
    clf.load_state_dict(torch.load(fmodel, map_location=torch.device(DEVICE)))
    clf = clf.to(DEVICE)

    acc_test = eval_model(clf, dataset_test, batch_size=256)
    print(f'Accuracy (test): {acc_test:.4f}')

