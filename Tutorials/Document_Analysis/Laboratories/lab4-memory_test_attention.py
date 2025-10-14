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

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import matplotlib.pyplot as plt


# %%
class RNN(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.rnn = nn.RNN(dim_input, dim_hidden, 1, nonlinearity='relu')
        self.W = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h_all, h_final = self.rnn(x)
        return self.W(h_final.squeeze(0))


class GRU(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.gru = nn.GRU(dim_input, dim_hidden, 1)
        self.W = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h_all, h_final = self.gru(x)
        return self.W(h_final.squeeze(0))


class LSTM(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.lstm = nn.LSTM(dim_input, dim_hidden, 1)
        self.W = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        h_all, (h_final, c_final) = self.lstm(x)
        return self.W(h_final.squeeze(0))


# %%
class RandomSequenceDataset():
    def __init__(self, dimension, sequence_length, batch_size, variance=1.0):
        self.dim = dimension
        self.sl = sequence_length
        self.bs = batch_size
        self.var = variance

    def get_batch(self):
        return torch.randn((self.sl, self.bs, self.dim)) * self.var


# %% [markdown]
# ### TODO: Modify **AttentivePooling** and **AttentionGRU**
# 1) Implement different types of attention score functions.
# 2) Implement the ReLU normalisation.

# %%
class AttentivePooling(nn.Module):
    # Computes attention with the last hidden state as the key
    # Note: will calculate attention for all elements in the batch in parallel.

    def __init__(self, dim_in):
        super().__init__()
        self.W = nn.Linear(dim_in, dim_in)

    def forward(self, h_all, xin):
        # NOTE: h_all and xin both have the shape: [sequence length, batch size, hidden size]        

        # get the last hidden state and compute a key from it 
        h_last = h_all[-1] # h_last is of shape [batch_size, hidden size]
        # key_a = self.W(h_last) # key_a is of shape [batch_size, hidden size]

        # calculate the attention vector:
        # permute dimensions in h_all to order: [batch, seq, hidden]
        h_all = torch.permute(h_all, (1, 0, 2)) 

        # batch matrix multiplication of [batch, seq, hidden] x [batch, hidden, 1] = [batch, seq, 1]
        # a = torch.matmul(h_all, key_a.unsqueeze(2)) # x^TWy
        a = torch.matmul(h_all, h_last.unsqueeze(2)) / math.sqrt(h_all.shape[2])

        # remove the trailing dimension of a and then compute the softmax over the sequence dimension
        relu_scores = nn.functional.relu(a.squeeze(2))
        relu_scores_norm = relu_scores / (torch.sum(relu_scores, dim=1, keepdim=True) + 1e-9)
        # small epsilon added to prevent division by zero

        # calculate the context vector using the attention and hidden states
        # [batch, 1, seq] x [batch, seq, hidden] = [batch, 1, hidden]
        output = torch.bmm(relu_scores_norm.unsqueeze(1), h_all)

        return output.squeeze(1)


# %%
class AttentionGRU(nn.Module):
    # A GRU with attention before a linear classification layer
    # this is an attention GRU for classification, attention is only calculated
    # at the end of the sequence rather than every step which is done in 
    # self-attention or attention auto-regressive models.
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.gru = nn.GRU(dim_input, dim_hidden, 1)
        self.W = nn.Linear(dim_hidden, dim_output)
        self.attention = AttentivePooling(dim_hidden)

    def forward(self, x):
        h_all, h_final = self.gru(x)
        attn_output = self.attention(h_all, x)
        res = self.W(attn_output)
        return res


# %%
input_dimension = 4
hidden_dimension = 4
sequence_lengths = [3, 5, 10, 20, 40, 80, 100]

# you can try this but the cpu is recommended for this task
# it is often faster when the models are small like this
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
def train_model(model, optimizer, sequence_length, dimension=input_dimension, batch_size=512):
    # random dataset generator
    dataset = RandomSequenceDataset(dimension, sequence_length, batch_size)

    num_epochs = 100
    num_batches = 5  # batches per epoch
    best_loss = 1e10 # stores the best average training loss in any epoch

    for epoch in range(num_epochs):
        average_loss = 0
        for i in range(num_batches):

            # get the training batch and move it to the gpu if one is found
            # shape: [sequence_length x batch_size x dimension]
            x = dataset.get_batch().to(device)

            # the target is the first element in the sequence, shape [batch_size x dimension]
            y = x[0]

            # run the specified type of RNN model and get the output, shape [batch_size x dimension]
            y_h = model(x)

            # calculate loss and update network
            loss = F.mse_loss(y_h, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute average loss for this epoch
            average_loss += loss.item() / num_batches
        if average_loss < best_loss:
            best_loss = average_loss
    return best_loss


# %%
def plot_losses(losses, sequence_lengths):
    plt.plot(sequence_lengths, losses, marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('MSE')
    plt.ylim([-0.05, None])
    plt.show()


# %%
def memory_test(model_func, sequence_lengths):
    # train the model on different length sequences 
    # and record the average training loss
    losses = []
    for sequence_length in sequence_lengths:
        # create a new copy of the model
        model = model_func().to(device)

        # create an optimizer for the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # train the model and store the best average loss
        loss = train_model(model, optimizer, sequence_length=sequence_length)
        print(f'Sequence length: {sequence_length}, \t running average loss: {loss}')
        losses.append(loss)
    
    plot_losses(losses, sequence_lengths)


# %%
memory_test(lambda: RNN(input_dimension, hidden_dimension, input_dimension), sequence_lengths)

# %%
memory_test(lambda: GRU(input_dimension, hidden_dimension, input_dimension), sequence_lengths)

# %%
memory_test(lambda: LSTM(input_dimension, hidden_dimension, input_dimension), sequence_lengths)

# %%
memory_test(lambda: AttentionGRU(input_dimension, hidden_dimension, input_dimension), sequence_lengths)

# %%
