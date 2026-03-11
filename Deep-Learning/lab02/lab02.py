# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <center>
# <h1>COMP3242/6242: Deep Learning</h1>
# <h2>Lab #2: Linear Classifiers and Multilayer Perceptrons</h2>
# Semester 1, 2026<br>
# </center>
#
#
# **Due**: 11:59pm on Sunday 8 March, 2026.<br>
#
# Submit solutions as a single Jupyter Notebook via Canvas. Make sure that your name and student ID appears in the section below. You may not work with any other person in completing this laboratory. You must acknowledge any non-course texts or online material used. See the course policy on the use of generative AI tools such as ChatGPT and Claude.
#
# This assignment will be **automatically graded**. Read and follow instructions carefully!
#
# Test code is provided for you to check your work as you progress through the assignment. Feel free to add further tests and output useful for your own debugging. Note that this code will not be run when we automatically grade your submission. We will exercise your code beyond what is provided here. Do not add any Jupyter notebook magic commands (i.e., those starting with `%` or `%%`). These may cause the autograding script to fail.
#
# Complete all **TODOs** and delete any placeholder (`pass` and `...`).
#
# **Run all code blocks from start to end (`Restart & Run All`) and then save your Jupyter Notebook before submitting your assignment to ensure everything works as expected.**

# %%
# TODO: Replace with your name and university ID\
student_name = "Razeen Wasif"
student_id = "u7283652"

# %% [markdown]
# ## Disclosure
#
# TODO: doucment any use of online resources (other than course material) and/or generative AI tools.

# %%
# The ANU lab machines will have python and the necessary packages already installed. If you're working
# on your own machine you will need to install python with Jupyter notebook, OpenCV and PyTorch:
#  - Documentation for the OpenCV computer vision library is here: https://opencv.org/. You should be able
#    to install the library on your computer using `pip install opencv-python` or if you have conda
#    `conda install conda-forge::opencv`.
#  - Documentation for the PyTorch deep learning library is here: https://pytorch.org/. Follow the
#    installation instructions (for the stable release, v2.9.1 at time of writing), being sure to install
#    both `pytorch` and `torchvision`. You will not need access to a GPU for this course but having one
#    may increase computation speed.
#  Browse through the user documentation and tutorials for these libraries.

import sys
import getpass

def is_notebook():
    return 'ipykernel' in sys.modules

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
if is_notebook():
    print("User: {} ({}, {})".format(getpass.getuser(), student_name, student_id))
    print("Python Version: {}".format(sys.version))
    print("PyTorch Version: {}".format(torch.__version__))

# %%
device = torch.device("cpu")
random_seed = 32426242
torch.manual_seed(random_seed)


# %%
# Preliminaries -- Generating an synthetic binary classification dataset that follows the XOR function 
# Conceptually, the 2D feature vectors can be seen as randomly generated points on a 2d plane
# and the corresponding labels denotes where the coordinates of the points have the same sign. 
def gen_xor_data(
    num_samples: int, train_ratio: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate XOR data.

    Args:
        num_samples (int): number of samples to generate.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: tuple of (x_train, y_train, x_test, y_test) where x is the input data and y is the label.
    """
    x = torch.rand(num_samples, 2)
    y = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        if x[i, 0] > 0.5:
            if x[i, 1] > 0.5:
                y[i] = 0
            else:
                y[i] = 1
        else:
            if x[i, 1] > 0.5:
                y[i] = 1
            else:
                y[i] = 0
    x -= 0.5
    num_train_data = int(len(x) * train_ratio)
    return x[:num_train_data], y[:num_train_data], x[num_train_data:], y[num_train_data:]


x_train, y_train, x_test, y_test = gen_xor_data(1000)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
if is_notebook():# Visualise XOR data
    def visualize_xor_data(x: torch.Tensor, y: torch.Tensor, title: str = None):
        plt.scatter(x[y == 0, 0], x[y == 0, 1], color="red", label="0")
        plt.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", label="1")
        plt.legend()
        if title:
            plt.title(title)
        plt.show()


    visualize_xor_data(x_train, y_train, "Train Data")
    visualize_xor_data(x_test, y_test, "Test Data")


# %% [markdown]
# ## Task 1 -- Linear Forward 

# %%
# --- TASK 1 -----
# Implement an linear forward function 
# Given input x, weight matrix A and bias vector b, return the output y
def linear_forward(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One forward pass of a single-layer perceptron (one linear layer).

    Args:
        x (torch.Tensor): The input to the linear layer, of shape (B, D).
        A (torch.Tensor): The weight matrix of the linear layer, of shape (H, D).
        b (torch.Tensor): The bias vector of the linear layer, of shape (H,).

    Returns:
        torch.Tensor: The output of the linear layer, needs to be in the shape of (B, H).

    Notes:
        Checkout PyTorch's implementation of `torch.nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html.
    """
    # TODO: implement the forward pass of a single-layer perceptron.
    # y = xA^T + b
    return x @ A.T + b 


# %%
# Tests for task 1
if is_notebook():
    # creating a linear layer with randomly initialised weight and bias
    in_features = 2
    out_features = 2
    linear_model = nn.Linear(in_features, out_features)
    # print out the weight and bias
    weight = linear_model.weight
    bias = linear_model.bias
    print(f'weight: {weight}')
    print(f'bias: {bias}')
    # verify that the output of the function has the correct shape
    output = linear_forward(x_test, weight, bias)
    print(f'shape of output: {output.shape}')
    # verify that linear_forward is functionally correct
    assert torch.allclose(linear_model(x_test), output), 'Your linear forward function does not match PyTorch\'s implementation\n'


# %% [markdown]
# ## Task 2 -- Calculate Accuracy 
#

# %%
# --- TASK 2 --------------------------------------------------------------------------------------
# Implement a basic classification metric -- accuracy   
def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates the accuracy given `output` and `target`.

    The accuracy is the number of correct predictions divided by the total number of predictions.

    Args:
        output (torch.Tensor): The unnormalized output from the model, of shape (B, C).
        target (torch.Tensor): The ground truth labels, of shape (B,).

    Returns:
        float: The accuracy.

    Notes:
        The result should be converted to a `float`.
    """
    # TODO: implement the accuracy calculation.
    predictions = torch.argmax(output, dim=1) 
    correct = (predictions == target).float()
    accuracy = correct.mean().item()
    return accuracy 


# %%
# Tests for Task 2.

if is_notebook():
    from lab02_utils import  train_model
    # Recall the linear model created for testing task 1     
    in_features = 2
    out_features = 2
    linear_model = nn.Linear(in_features, out_features)
    weight = linear_model.weight
    bias = linear_model.bias
    linear_model.forward = lambda x: linear_forward(x, weight, bias)    
    # TODO: Feel free to change the hyperparameters
    train_model(
        model=linear_model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        loss_fn=nn.CrossEntropyLoss(),
        accuracy_fn=calculate_accuracy,
        num_epochs=100,
        print_every_epochs=10,
        evaluate_every_epochs=100,
        lr=0.1,
        temperature=None,
    )

    

# %% [markdown]
# ## Task 3 -- 2-layer MLP

# %%
# --- TASK 3 --------------------------------------------------------------------------------------
# Build a basic neural network 

def build_two_layer_mlp(in_features: int = 2, out_features: int = 2, hidden_features: int = 4) -> nn.Module:
    """Builds a two-layer multi-layer perceptron (MLP) with ReLU activation functions in between.

    Args:
        in_features (int, optional): Number of input features. Defaults to 2.
        out_features (int, optional): Number of output features. Defaults to 2.
        hidden_features (int, optional): Number of hidden features. Defaults to 4.

    Returns:
        nn.Module: A PyTorch module representing the MLP.

    Notes:
        Use the `torch.nn` module to create `Linear` (with bias) and `ReLU` layers. We then wrap all layers using `Sequential`.
        Documentation for `Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html.
        Documentation for `ReLU`: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html.
        Documentation for `Sequential`: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html.
    """
    # TODO: implement the two-layer MLP by creating two linear layers of appropriate size and a non-linear activation layer,
    # replacing `None` with the relevant PyTorch functions. See comments above for details.
    linearLayer1 = nn.Linear(in_features, hidden_features)
    activationLayer = nn.ReLU()
    linearLayer2 = nn.Linear(hidden_features, out_features)
    
    return nn.Sequential(linearLayer1, activationLayer, linearLayer2)



# %%
# Tests for Task 3.

if is_notebook():
    # testing out the 2-layer mlp
    # TODO: Feel free to change the hyperparameters
    train_model(
        model=build_two_layer_mlp(2, 2, 4),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        loss_fn=nn.CrossEntropyLoss(),
        accuracy_fn=calculate_accuracy,
        num_epochs=100,
        print_every_epochs=10,
        evaluate_every_epochs=100,
        lr=0.1,
        temperature=None,
    )


# %% [markdown]
# ## Task 4 -- Temperature-scaled Cross-entropy Loss

# %%
# --- TASK 4 --------------------------------------------------------------------------------------
# Define the loss function for the classification task 

def cross_entropy_loss(output: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    """Calculates the temperature scaled cross-entropy loss given `output` and `target`.

    Args:
        output (torch.Tensor): The unnormalized output from the model, of shape (B, C).
        target (torch.Tensor): The ground truth labels, of shape (B,).
        temperature (float): The temperature scaling for the softmax function.

    Returns:
        torch.Tensor: The loss value.

    Notes:
        See the lecture slides for the temperature scaling formula.
        The loss should be a single scalar tensor, averaged over the batch dimension.
        It is allowed to use `CrossEntropyLoss`, but we encourage you to challenge yourself by implementing it manually.
        Documentation for `CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html.
    """
    # TODO: implement the cross-entropy loss calculation.
    return nn.functional.cross_entropy((output/temperature), target)


# %%
# Some tests for Task 4.

if is_notebook():
    # TODO: Feel free to change the hyperparameters
    train_model(
        model=build_two_layer_mlp(2, 2, 4),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        loss_fn=cross_entropy_loss,
        accuracy_fn=calculate_accuracy,
        num_epochs=100,
        print_every_epochs=10,
        evaluate_every_epochs=100,
        lr=0.1,
        temperature=0.1,
    )

# %% [markdown]
# ## Task 5 -- Minimum Layers for XOR 

# %%
# --- TASK 5 -----------------------------------------------------------------------------------------------------------
# TODO: Set this constant to the minimum number of layers required to solve the XOR problem using an MLP theoretically.
# You are welcome to borrow code from previous tasks to verify your answer.
# Be sure not to directly modify or accidentally overwrite Task 1-4 upon submission. 
# Note that we will not run your code. We'll only check the value of this constant.
MINIMUM_NUMBER_OF_LAYERS_TO_SOLVE_XOR_PROBLEM: int = 2 


