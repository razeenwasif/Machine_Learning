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
# # PyTorch Basics
#
# [PyTorch](https://pytorch.org/) is a library for creating and training neural-network-based models. It provides functions for performing various matrix and vector operations, computing losses and metrics, and optimization. Crucially, it also implements auto-differentiation. This means that whenever you use one of the built-in PyTorch operations to perform a computation, the result of the computation will be stored in memory. These stored computations can then be used to compute gradients for each of the intermediate computation steps using back-propagation. This allows for arbitrary differentiable functions to be optimized.

# %% [markdown]
# ## Tensors
#
# ![tensors.png](attachment:tensors.png)

# %% [markdown]
# In PyTorch (and other popular neural network libraries), all of the operations act on <i>tensors</i> of data. 
#
# These tensors are really just multi-dimensional arrays of numbers. A 1-dimensional tensor is an array of numbers indexed by one number (that is, a vector), and a 2-dimensional tensor is an array of numbers indexed by two numbers (that is, a matrix), and so on. 
#
# In PyTorch, tensors are 0-indexed.

# %%
import torch
import torch.nn.functional as F
import numpy as np
import os

# %% [markdown]
# ## Practice with tensor operations

# %% [markdown]
# ### Indexing tensors
#
# Tensors can be indexed just like arrays in Python.

# %%
a = torch.FloatTensor([[1, 2], [3, 4]]) # Create a 2x2 matrix.
print(f'a: {a}')
print(f'first row: {a[0]}')
print(f'first column: {a[:, 0]}')

# %% [markdown]
# ### Computing with tensors

# %%
a = torch.FloatTensor([-1, 1]).view(2, 1) # Create a 2x1 matrix.
b = torch.FloatTensor([0, 1]).view(1, 2) # Create a 1x2 matrix.
c = a @ b # Perform matrix multiplication on a and b. The result of an AxB matrix multiplied with a BxC matrix is a AxC matrix.
print(f'c has a shape of {c.shape}, this means that it has {c.shape[0]} rows and {c.shape[1]} columns.')
print(f'c: {c}')

# %%
a = torch.arange(24).view(4, 2, 3) # Create a tensor of shape 4x2x3.
# torch.arange creates a 1d tensor of integers ranging from 0 up to the specified number.
print(f'a: {a}')

a_sum = a.sum(dim=1) # The sum function adds up all the values along the specified dimension.
                     # In this case we sum over the second dimension.
print(f'a_sum has a shape of {a_sum.shape}')
print(f'a_sum: {a_sum}')

# %% [markdown]
# Try changing the dimension in a.sum() above to see how it affects the result. What happens if you don't pass in any dimension at all?

# %%
a = torch.FloatTensor([2, 4]).view(2, 1) # Create a 2x1 matrix.
print(f'a: {a}')
b = torch.FloatTensor([1,-1]).view(2, 1) # Create another 2x1 matrix.

ab = a * b # This symbol * means element-wise multiplication (not to be confused with matrix multiplication!).
print(f'ab: {ab}')

# %% [markdown]
# ### Exercise
# Copy the cell above and then try changing the shape of $a$ to (1, 2) and then re-running it. What do you expect the result to be?

# %%
# TODO: change the shape of a to (1, 2) and run a * b
a = torch.FloatTensor([2, 4]).view(1, 2)
ab = a * b 
print(f'ab: {ab}')


# %% [markdown]
# ## Broadcasting
#
# What you have just discovered is known as broadcasting. When you try to perform an element-wise operation on two tensors that do not have the same shape, PyTorch will attempt to broadcast them to the same shape. Whenever one tensor's dimension has a size of 1, that tensor will be repeatedly stacked onto itself until it has the same shape as the other one. This process begins at the last dimension then proceeds towards the first dimension.
#
# In the example above, since $b$ has a shape of (2,1), it is first repeated twice in the second dimension, creating the matrix
# $
# \begin{bmatrix}
# 1 & 1 \\
# -1 & -1
# \end{bmatrix}
# $.
# Then, since $a$ has a shape of (1, 2), it is repeated twice in the first dimension creating the matrix 
# $
# \begin{bmatrix}
# 2 & 4 \\
# 2 & 4
# \end{bmatrix}
# $.
# Finally these 2 matrices have the same shape and so the element-wise operation can proceed.
#
# Broadcasting can come in handy in certain situations, making otherwise complicated operations very simple to code. But it can also cause bugs and unexpected results if you aren't aware of it!
#

# %% [markdown]
# ### Exercise
#
# Compute the matrix product $a$ @ $b$ using only broadcasted element-wise multiplication and sums. You will need to reshape $a$ to the correct shape for it to work.

# %%
a = torch.FloatTensor([1, 3])
b = torch.FloatTensor([[2, 1], [-1, 4]])

# TODO: Compute your ab here. It should give the same result as a@b
# ab = 

print(ab)
print(a@b)

# %% [markdown]
# ## Concatenation
#
# One very common tensor operation is concatenation. Concatenation means to join multiple tensors together end-to-end to create one tensor.

# %%
a = torch.FloatTensor([-1, 2]).view(2, 1)
b = torch.FloatTensor([2, 3]).view(2, 1)
c = torch.FloatTensor([4, 1]).view(2, 1)

concatenation0 = torch.cat([a, b, c], dim=0)
print(f'concatenate dimension 0: {concatenation0}\n')

concatenation1 = torch.cat([a, b, c], dim=1)
print(f'concatenate dimension 1: {concatenation1}')

# %% [markdown]
# Concatenation is often used to combine tensors together, so that computations can be performed on 1 large tensor instead of multiple smaller ones, since this is usually faster.

# %% [markdown]
# # Building a simple model
# We're now going to build a simple linear regression model. 
#
# To begin with, we will define some random $x$ and $y$ variables to make up our training dataset.

# %%
n_data = 1000
dim = 3

train_X = torch.randn(n_data, dim) * 2
eps = torch.randn(n_data, 1) * 0.1

# Randomly generate points around the line f(x, y, z) = 2x - y + 4z + 0
train_y = (train_X @ torch.FloatTensor([2, -1, 4]).view(dim, 1)) + eps

# %%
print(f'x: {train_X[:10]}')
print(f'y: {train_y[:10]}')


# %% [markdown]
# Next we will define our regression model.

# %%
class LinearRegressor(torch.nn.Module): 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out) 
        # This creates a "linear" module. The linear module consists of a weight matrix [d_in, d_out] and a bias vector [d_out].
        # When a new instance of this class is created this module will be initialized with random values.
        # Since this class subclasses torch.nn.Module, all of the parameters in the linear module will be added
        # to this LinearRegressor's parameters.
    
    def forward(self, x):
        # The forward function is applied whenever this model is called on some input data.
        # The forward function specifies how the model computes its output.
        y_h = self.linear1(x) # Apply our linear operation to the input.
        return y_h

model = LinearRegressor(dim, 1)

# %% [markdown]
# Next we need an optimiser. 
#
# The optimiser will change the values of all of the tensors given to it. Each time the optimisation step is run, the variables are changed by a small amount (how much they change each step is controlled by the learning rate).

# %%
# Create a stochastic gradient descent optimizer, and give it all our model's parameters as tensors to be optimised
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)


# %% [markdown]
# And finally, we can train our model. 
#
# When training neural networks, it is common to run optimisation steps on every input in the dataset multiple times. Each pass through the entire dataset is called one *epoch* of training. Usually models are trained for multiple epochs.

# %%
def train_model(model):
    n_epochs = 5
    
    # Train our model for `n_epochs` epochs
    for i in range(n_epochs):
        yh = model(train_X) # Apply model to inputs.

        loss = F.mse_loss(yh, train_y) # Compute mean squared error between our model output and the correct labels.

        optimiser.zero_grad() # Set all gradients to 0.
        loss.backward() # Calculate gradient of loss w.r.t all tensors used to compute loss.
        optimiser.step() # Update all model parameters according to calculated gradients.

        print(f'epoch {i}, parameter values:')
        print(list(model.parameters()))
        print('\n')

# Train the regressor
train_model(model)    

# Our model can now be used to predict y values for new X as follows.
new_X = torch.arange(dim).float().view(1, dim)
new_y_h = model(new_X)
print(f'The model predicts {new_y_h}')

# %% [markdown]
# ## Inspecting gradients
#
# Although optimisation is handled under-the-hood by PyTorch, there are some situations where it is helpful to access gradients directly. Gradients are stored in each variables' `.grad` attribute.

# %%
w = torch.FloatTensor([1, 2])
w.requires_grad = True # If this flag is set then this variable will store its gradient during backward().
x = torch.FloatTensor([3, 4])
x.requires_grad = True

z = (w*w*x).sum()
z.backward()

print(w.grad)
print(x.grad)


# %% [markdown]
# ## Neural networks
#
# ### Exericse
# Complete the implementation below of a 1 hidden layer neural network. You will need to define 2 linear functions, one for the hidden layer and one for the output layer. In the forward function you can use torch.relu() as the activation function.

# %%
class NeuralNetwork(torch.nn.Module): 
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        # TODO: Define layers here
        
    
    def forward(self, x):
        # TODO: Compute output here
        
        return y_h

model = NeuralNetwork(dim, 128, 1)
train_model(model)

# %% [markdown]
# # Embedding words into vector space
#
# Neural networks can only be applied to numeric valued inputs, since they rely on multiplication operations. This means if we want to apply a neural network to non-numeric data, such as text, it must first be converted into a numeric value. One very common way of doing this is to use one-hot encoding (also known as one-of-k encoding). In the one-hot encoding scheme, each input value is represented as a vector which has 0 in every component except for one component corresponding to this input's unique index.

# %% [markdown]
# For example, to handle text input, every word in our language would be assigned a vector like this:
#
# $an = [1, 0, 0, \dots, 0, 0]$
#
# $all = [0, 1, 0, \dots, 0, 0]$
#
# $be = [0, 0, 1, \dots, 0, 0]$
#
# $\dots$
#
# $zebra = [0, 0, 0, \dots, 1, 0]$
#
# $zoom = [0, 0, 0, \dots, 0, 1]$
#
# Note that these vectors have a component for every word in our language, we will call the number of words in our language $|V|$, so these vectors have a shape of $(1, |V|)$. And note that $|V|$ could be a number in the tens of thousands.

# %% [markdown]
# Once we have converted each word into a vector, we can then use them as input to our neural network. The first hidden layer would apply a weight matrix of shape $(|V|, h_1)$, where $h_1$ is the number of units in the first hidden layer, which results in an output of shape $(1, h_1)$.
#
# PyTorch's <a href=https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>Embedding</a> module does both of these operations (one-hot encoding then matrix multiplication) in a computationally efficient way. Note that because all of our input vectors are of the special one-hot form, the result of the matrix multiplication is precisely equivalent to the $i$'th row of the matrix, where $i$ is the ID of the word. So all the embedding module does is lookup this row.

# %%
num_words = 5 # This is |V|.
embedding_dim = 20 # This is h1.

embedder = torch.nn.Embedding(num_words, embedding_dim)

input_words = torch.LongTensor([0, 3, 2]) # This is a batch of word IDs.

out = embedder(input_words) # Apply the embedder to our input, the result is a batch of 3 vectors of size 20.
out # These are known as the 'word embeddings'.

# %%
