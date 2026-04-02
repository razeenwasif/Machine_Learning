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
# <h2>Lab #6: RNN, LSTM, and Backprop Through Time</h2>
# Semester 1, 2026<br>
# </center>
#
# **Due**: 11:59pm on Sunday 5 Apr, 2026.<br>
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
# TODO: Replace with your name and university ID
student_name = "Razeen Wasif"
student_id = "u7283652"

# %%
import sys
import getpass

from io import open


def is_notebook():
    return 'ipykernel' in sys.modules


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("User: {} ({}, {})".format(getpass.getuser(), student_name, student_id))
print("Python Version: {}".format(sys.version))
print("PyTorch Version: {}".format(torch.__version__))


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 3242
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


# %% [markdown]
# ## Loading and Exploring the Data
#
# In this lab we are going to build character-level language models — models that learn to predict the next character in a sequence of text. Every unique character (letter, space, punctuation) is a token in our vocabulary. By the end, your models will generate new text that mimics the style of the training corpus.

# %%
if is_notebook():
    # Load text data
    data = open('data/textData.txt', 'r').read()
    chars = sorted(list(set(data)))
    V = len(chars)  # vocabulary size
    data_size = len(data)

    print(f"Data has {data_size} characters, {V} unique.")
    print(f"First 200 characters:\n{data[:200]}")
    print(f"\nVocabulary: {''.join(chars)}")

    # Create mappings between characters and indices
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}


# %% [markdown]
# ---
# ## Task 1: Implement an RNN from Scratch
#
# ### The RNN Equations
#
# At each timestep $t$, the Elman RNN computes the latent state $h_t$ and output $y_t$ as:
#
# $$h_t = \textbf{tanh}(W_h h_{t-1} + U_h x_{t} + b_h)$$
# $$y_t = \textbf{sigmoid}(W_{y} h_t + b_y)$$
#
# Where:
# - $x_t \in \mathbb{R}^{V}$ is the one-hot encoded input character at time $t$ (the t-th character in text)
# - $h_{t-1} \in \mathbb{R}^{H}$ is the hidden state from the previous time step
# - $y_t \in \mathbb{R}^{V}$ is the output logits
# - $W_{h}, U_{h}, W_{y}, b_h, b_y$ are learnable parameters.

# %% [markdown]
# ### 1.1 Encoding Characters as Vectors
#
# Convert character indices into **one-hot column vectors** (all `0`'s except a single `1` at the character’s vocabulary index). In code we store a length-$T$ sequence as a matrix of shape **`(V, T)`**.
#
# For example: we read in 'deep learning', `'d'` and `'e'` will be used as a training pair, which will be encoded into vectors separately. Assuming that we have only these 9 distict characters in our vocabulary: `{'d','e','p',' ','l','a','r','n','i','g'}`, then 
# - 'd' will be encoded as $[1,0,0,0,0,0,0,0,0]^T$, and 
# - 'e' will be encoded as $[0,1,0,0,0,0,0,0,0]^T$.

# %%
def one_hot_encode(indices, V):
    """Convert a sequence of character indices to a tensor of one-hot column vectors.

    Args:
        indices: list/tuple of ints, or any object convertible to a 1D long tensor
        V: vocabulary size

    Returns:
        Float tensor of shape (V, T), column `t` is the one-hot for `indices[t]`.
    """
    idx = torch.as_tensor(indices, dtype=torch.long).view(-1)
    T = idx.numel()
    encoding = torch.zeros(V, T, dtype=torch.float32)
    encoding[idx, torch.arange(T)] = 1.0

    return encoding



# %%
if is_notebook():
    test_words = ["hello", "deep", "learning"]
    for word in test_words:
        indices = [char_to_ix[ch] for ch in word]
        print("Example: ", word)
        encoded = one_hot_encode(indices, V)
        print(f"Encoded shape: {tuple(encoded.shape)}")
        assert torch.allclose(encoded.sum(dim=0), torch.ones(len(word))), \
            "Each column should sum to 1, got " + str(encoded.sum(dim=0))
        print(f"Each column of encoded matrix sums to 1")
        for t, ix in enumerate(indices):
            assert encoded[ix, t].item() == 1.0, \
                f"Position [{ix}, {t}] should be 1.0 for char '{word[t]}'"
        print()

# %% [markdown]
# ### 2.2 Defining the RNN Forward Pass

# %%
if is_notebook():
    H = 100   # hidden size

    # Define model parameters
    # nn.Parameter(tensor) so PyTorch tracks gradients.
    Uh = nn.Parameter(torch.randn(H, V)) * 0.01
    Wh = nn.Parameter(torch.randn(H, H)) * 0.01
    Wy = nn.Parameter(torch.randn(V, H)) * 0.01
    bh = nn.Parameter(torch.zeros(H, 1))
    by = nn.Parameter(torch.zeros(V, 1))

    rnn_params = [Uh, Wh, Wy, bh, by]
    print(f"Total parameters: {sum(p.numel() for p in rnn_params)}")


# %%
## -------------- Task 1A: Defining RNN Update Function ------------------------------
def rnn_update(xt, hprev, Uh, Wh, Wy, bh, by):
    """

    Args:
        xt (torch.Tensor): one-hot encoded input vector
        hprev (torch.Tensor): previous hidden state
        Uh (torch.nn.Parameter): input-to-hidden weight matrix
        Wh (torch.nn.Parameter): hidden-to-hidden weight matrix
        Wy (torch.nn.Parameter): hidden-to-output weight matrix
        bh (torch.nn.Parameter): hidden bias
        by (torch.nn.Parameter): output bias

    Returns:
        h (torch.Tensor): new hidden state of shape (H, 1)
        y (torch.Tensor): output logits of shape (V, 1)
        p (torch.Tensor): softmax probabilities of shape (V, 1)
    """
    
    # TODO: Implement the RNN update function
    
    # h_t = tanh(Wh h_{t-1} + Uh x_t + bh)
    h = torch.tanh(Wh @ hprev + Uh @ xt + bh)
    logits = Wy @ h + by
    # y_t = sigmoid(Wy h_t + by)
    y = torch.sigmoid(logits)
    
    # softmax probabilities
    p = torch.exp(logits) / torch.sum(torch.exp(logits))
    return h, y, p


# %%
if is_notebook():
    test_xt = torch.zeros(V, 1)
    test_xt[0] = 1.0
    test_hp = torch.zeros(H, 1)
    test_h, test_y, test_p = rnn_update(test_xt, test_hp, Uh, Wh, Wy, bh, by)
    assert test_h.shape == (H, 1), \
        f"Hidden state shape: expected ({H}, 1), got {tuple(test_h.shape)}"
    assert test_y.shape == (V, 1), \
        f"Logits shape: expected ({V}, 1), got {tuple(test_y.shape)}"
    assert test_p.shape == (V, 1), \
        f"Probability shape: expected ({V}, 1), got {tuple(test_p.shape)}"
    p_sum = test_p.sum().item()
    assert abs(p_sum - 1.0) < 1e-5, f"Probabilities should sum to 1, got {p_sum}"
    print(f"rnn_update h shape: {tuple(test_h.shape)} ")
    print(f"rnn_update y shape: {tuple(test_y.shape)} ")
    print(f"rnn_update p shape: {tuple(test_p.shape)}, sum={p_sum:.4f} ")
    print(f"Task 1A passed!")


# %%
## -------------- Task 1B: Forward pass to encode text file ------------------------------
def rnn_forward(inputs, targets, hprev, Uh, Wh, Wy, bh, by):
    """Forward pass of the RNN.

    Args:
        inputs: list of integer character indices (length T)
        targets: list of integer character indices (length T), shifted by 1 from inputs
        hprev: initial hidden state, shape (H, 1)

    Returns:
        loss: scalar cross-entropy loss over the sequence
        xs: dict of one-hot input vectors at each timestep (timestep int as key)
        hs: dict of hidden states at each timestep (hs[-1] is the initial state)
        ps: dict of probability distributions at each timestep
    """
    V = Uh.shape[1]
    xs, hs, ps = {}, {}, {}
    hs[-1] = hprev.clone()
    loss = torch.zeros((), dtype=hprev.dtype)
    all_xs = one_hot_encode(inputs, V)
    for t in range(len(inputs)):

        # Step 1: one-hot input of shape (V, 1)
        xs[t] = all_xs[:, t:t+1]

        # TODO: Step 2: hidden state and probabilities via rnn_update()
        hs[t], _, ps[t] = rnn_update(xs[t], hs[t-1], Uh, Wh, Wy, bh, by)

        # TODO: Step 3: Cross-entropy loss for this timestep: -log p[target]
        loss = loss + -torch.log(ps[t][targets[t], 0])
 
    return loss, xs, hs, ps



# %%
if is_notebook():
    test_inputs  = [char_to_ix[ch] for ch in data[:5]]
    test_targets = [char_to_ix[ch] for ch in data[1:6]]
    test_h0 = torch.zeros(H, 1)
    test_loss, test_xs, test_hs, test_ps = rnn_forward(test_inputs, test_targets, test_h0, Uh, Wh, Wy, bh, by)
    assert isinstance(test_loss, torch.Tensor) and test_loss.dim() == 0, \
        f"Loss should be a scalar tensor, got shape {test_loss.shape}"
    assert test_hs[0].shape == (H, 1), \
        f"Hidden state shape: expected ({H}, 1), got {tuple(test_hs[0].shape)}"
    assert test_ps[0].shape == (V, 1), \
        f"Probability shape: expected ({V}, 1), got {tuple(test_ps[0].shape)}"
    assert test_xs[0].shape == (V, 1), \
        f"Input one-hot shape: expected ({V}, 1), got {tuple(test_xs[0].shape)}"
    p_sum = test_ps[0].sum().item()
    assert abs(p_sum - 1.0) < 1e-5, f"Probabilities should sum to 1, got {p_sum}"
    print(f"rnn_forward loss:  {test_loss.item():.4f} (scalar)")
    print(f"Hidden state shape: {tuple(test_hs[0].shape)}")
    print(f"Probability shape:  {tuple(test_ps[0].shape)}, sum={p_sum:.4f}")


# %% [markdown]
# We can generate text by sampling one character at a time and feeding the output back as input for the next step: we sample from $p(w_t | h_t, w_{t-1})$ and feed $w_t$ back as the next input.

# %%
## -------------- Task 1C: Sampling — Auto-regressive Generation --------------------------
def sample(h, seed_ix, n, Uh, Wh, Wy, bh, by):
    """Sample a sequence of n characters from the model.

    Args:
        h: current hidden state, shape (H, 1)
        seed_ix: index of the seed character to start generation
        n: number of characters to generate
        Uh, Wh, Wy, bh, by: RNN parameters

    Returns:
        Long tensor of shape (n,) with sampled character indices.
    """
    V = Uh.shape[1]
    x = torch.zeros(V, 1, dtype=torch.float32)
    x[seed_ix, 0] = 1.0
    generated = []

    with torch.no_grad():
        for _ in range(n):
            # TODO: run rnn_update() to obtain the probability distribution p of the next character
            h, _, p = rnn_update(x, h, Uh, Wh, Wy, bh, by)

            probs = p.view(-1).clamp(min=1e-12)
            probs /= probs.sum()
            next_ix = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_ix)

            # TODO: one-hot encode the next character and update x (shape (V, 1))
            x = torch.zeros(V, 1, dtype=Uh.dtype, device=Uh.device)
            x[next_ix, 0] = 1.0

    return torch.tensor(generated, dtype=torch.long)


# %%
if is_notebook():
    # Test sampling before training (should be random gibberish)
    h0 = torch.zeros(H, 1)
    sampled_length = 20
    sampled_ix = sample(h0, char_to_ix['A'], sampled_length, Uh, Wh, Wy, bh, by)
    sampled_text = ''.join(ix_to_char[int(ix)] for ix in sampled_ix)
    print("Sample before training:")
    print(sampled_text)
    assert len(sampled_ix) == sampled_length, \
        f"Sampled indices: expected length of {sampled_length}, got {len(sampled_ix)}"
    print(f"Task 1C successfully sampled {sampled_length} characters!")


# %% [markdown]
# ---
# ## Task 2: Manual Back-propagation Through Time (BPTT)
#
# Implement the backward pass manually for **cross-entropy** loss over a character sequence.
#
# **Forward pass** — at each timestep $t = 1, \dots, T$:
#
# | Step | Equation | Shape |
# |:-----|:---------|:------|
# | Hidden state | $h_t = \tanh(z_t) = \tanh(W_h h_{t-1} + U_h x_t + b_h)$ | $(H, 1)$ |
# | Logits | $y_t = W_y h_t + b_y$ | $(V, 1)$ |
# | Probabilities | $p_t = \text{softmax}(y_t)$ | $(V, 1)$ |
# | Loss | $\mathcal{L}_t = -\log p_t[\text{target}_t]$ | scalar |
#
# Total loss: $\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$.
#
# **Goal:** Manually compute the partial derivatives of the total loss $\mathcal{L}$ with respect to each parameter, then check them against **`loss.backward()`**.  
#
# | Parameter | Quantity | Shape |
# |:----------|:---------|:------|
# | $W_y$ | $\displaystyle\frac{\partial \mathcal{L}}{\partial W_y}$ | $\mathbb{R}^{V \times H}$ |
# | $b_y$ | $\displaystyle\frac{\partial \mathcal{L}}{\partial b_y}$ | $\mathbb{R}^{V \times 1}$ |
# | $W_h$ | $\displaystyle\frac{\partial \mathcal{L}}{\partial W_h}$ | $\mathbb{R}^{H \times H}$ |
# | $U_h$ | $\displaystyle\frac{\partial \mathcal{L}}{\partial U_h}$ | $\mathbb{R}^{H \times V}$ |
# | $b_h$ | $\displaystyle\frac{\partial \mathcal{L}}{\partial b_h}$ | $\mathbb{R}^{H \times 1}$ |
#
# We have provided you with the mathematical derivations. You need to implement these in the code block below.
#

# %% [markdown]
# ### Step 1: Softmax + Cross-Entropy Gradient
#
# With $p_t = \text{softmax}(y_t)$ and $\mathcal{L}_t = -\log p_t[\text{target}_t]$, the element-wise derivative is:
#
# $$\frac{\partial \mathcal{L}_t}{\partial y_t[j]} = p_t[j] - \mathbb{1}[j = \text{target}_t] = \begin{cases} p_t[j] - 1 & \text{correct class} \\ p_t[j] & \text{wrong class} \end{cases}$$
#
# In vector form:
#
# $$\boxed{\frac{\partial \mathcal{L}_t}{\partial y_t} = p_t - e_t \quad \in \mathbb{R}^{V \times 1}} \tag{2.1}$$
#
# where $e_t \in \mathbb{R}^{V}$ denotes the one-hot target vector at time $t$.

# %% [markdown]
# ### Step 2: Output Layer Gradients
#
#
# From $y_t = W_y h_t + b_y$:
#
# $$\boxed{\frac{\partial \mathcal{L}_t}{\partial W_y} = \frac{\partial \mathcal{L}_t}{\partial y_t} \frac{\partial \mathcal{y}_t}{\partial W_y} = (p_t - e_t)\, h_t^\top \quad \in \mathbb{R}^{V \times H}} \tag{2.2}$$
#
# $$\boxed{\frac{\partial \mathcal{L}_t}{\partial b_y} = \frac{\partial \mathcal{L}_t}{\partial y_t} \frac{\partial \mathcal{y}_t}{\partial b_y} = p_t - e_t \quad \in \mathbb{R}^{V \times 1}} \tag{2.3}$$
#
# Gradient flowing back into the hidden state (direct contribution from $\mathcal{L}_t$):
#
# $$\frac{\partial \mathcal{L}_t}{\partial h_t} = \frac{\partial \mathcal{L}_t}{\partial y_t} \frac{\partial \mathcal{y}_t}{\partial h_t} = W_y^\top\,(p_t - e_t) \quad \in \mathbb{R}^{H \times 1} \tag{2.4}$$
#

# %% [markdown]
# ### Step 3: Recurrence (BPTT)
#
# Since $h_t$ contributes to all future losses $\mathcal{L}_{t+1}, \dots, \mathcal{L}_T$ through the recurrence, the **total** gradient on $h_t$ has a direct term plus a contribution propagated back from $t+1$:
#
# $$\frac{\partial \mathcal{L}}{\partial h_t}
# = \underbrace{\frac{\partial \mathcal{L}_t}{\partial h_t}}_{\text{direct path}}
# \;+\; \underbrace{\left(\frac{\partial h_{t+1}}{\partial h_t}\right)^\top \frac{\partial \mathcal{L}}{\partial h_{t+1}}}_{\text{indirect path through } h_{t+1}}$$
#
# Given $h_{t+1} = \tanh(W_h h_t + U_h x_{t+1} + b_h)$ and $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$, the indirect path:
#
# $$
# \begin{aligned}
# \left(\frac{\partial h_{t+1}}{\partial h_t}\right)^\top\! \frac{\partial \mathcal{L}}{\partial h_{t+1}}
# &= \left[\text{diag}(1 - h_{t+1}^2) \cdot W_h\right]^\top \frac{\partial \mathcal{L}}{\partial h_{t+1}} \\[6pt]
# &= W_h^\top \cdot \text{diag}(1 - h_{t+1}^2)^\top \cdot \frac{\partial \mathcal{L}}{\partial h_{t+1}} \\[6pt]
# &= W_h^\top \cdot \text{diag}(1 - h_{t+1}^2) \cdot \frac{\partial \mathcal{L}}{\partial h_{t+1}} \\[6pt]
# &= W_h^\top \!\left[(1 - h_{t+1}^2) \odot \frac{\partial \mathcal{L}}{\partial h_{t+1}}\right]
# \end{aligned}
# $$
#
# $$\boxed{\frac{\partial \mathcal{L}}{\partial h_t} = W_y^\top(p_t - e_t) \;+\; W_h^\top\!\left[(1 - h_{t+1}^2) \odot \frac{\partial \mathcal{L}}{\partial h_{t+1}}\right] \quad \in \mathbb{R}^{H \times 1}} \tag{2.5}$$
#
# At the final timestep $T$, only the direct term remains: $\;\frac{\partial \mathcal{L}}{\partial h_T} = W_y^\top(p_T - e_T)$.

# %% [markdown]
# ### Step 4: Parameter Gradients — Summed Over Time
#
# Let $z_t = W_h h_{t-1} + U_h x_t + b_h$ denote the pre-activation ($h_t = \tanh(z_t)$):
#
# $$\boxed{\frac{\partial \mathcal{L}}{\partial z_t} = \frac{\partial \mathcal{L}}{\partial h_t} \odot \frac{\partial h_t}{\partial z_t} = (1 - h_t^2) \odot \frac{\partial \mathcal{L}}{\partial h_t} \quad \in \mathbb{R}^{H \times 1}} \tag{2.6}$$
#
# Each weight matrix appears at **every** timestep, so its total gradient is the sum of per-timestep contributions. Applying the chain rule to each parameter:
#
# At each timestep $t$,
#
# $$
# \frac{\partial \mathcal{L}}{\partial U_h}\bigg|_t = \frac{\partial \mathcal{L}}{\partial z_t} \cdot \frac{\partial z_t}{\partial U_h} = \frac{\partial \mathcal{L}}{\partial z_t} x_t^\top
# $$
#
# Summing over all timesteps:
#
# $$
# \boxed{\frac{\partial \mathcal{L}}{\partial U_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial z_t}\, x_t^\top \quad \in \mathbb{R}^{H \times V}} \tag{2.7}
# $$
#
# $$\boxed{\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial z_t}\, h_{t-1}^\top \quad \in \mathbb{R}^{H \times H}} \tag{2.8}$$
#
# $$\boxed{\frac{\partial \mathcal{L}}{\partial b_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial z_t} \quad \in \mathbb{R}^{H \times 1}} \tag{2.9}$$
#
# Output weights $W_y$ and bias are not part of the recurrence. From $y_t = W_y h_t + b_y$, Eq 2.2 and 2.3:
#
# $$\boxed{\frac{\partial \mathcal{L}}{\partial W_y} = \sum_{t=1}^{T} (p_t - e_t)\, h_t^\top \quad \in \mathbb{R}^{V \times H}} \tag{2.10}$$
#
# $$\boxed{\frac{\partial \mathcal{L}}{\partial b_y} = \sum_{t=1}^{T} (p_t - e_t) \quad \in \mathbb{R}^{V \times 1}} \tag{2.11}$$

# %%
def rnn_forward_and_backward(inputs: list[int], 
                             targets: list[int], 
                             hprev: torch.Tensor, 
                             Uh: torch.Tensor, Wh: torch.Tensor, Wy: torch.Tensor, bh: torch.Tensor, by: torch.Tensor) -> tuple[float, dict[str, torch.Tensor], torch.Tensor]:
    """
    Manual forward and backward pass for the vanilla RNN
    
    Args:
        inputs (list[int]): list of character indices
        targets (list[int]): list of character indices
        hprev (torch.Tensor): initial hidden state
        Uh (torch.Tensor): input-to-hidden weight matrix
        Wh (torch.Tensor): hidden-to-hidden weight matrix
        Wy (torch.Tensor): hidden-to-output weight matrix
        bh (torch.Tensor): hidden bias
        by (torch.Tensor): output bias
        
    Returns:
        loss (float): loss value
        grads (dict[str, torch.Tensor]): dictionary of gradients
        h_final (torch.Tensor): final hidden state
    """
    H, V = Uh.shape
    T = len(inputs)
    
    # forward pass
    loss, xs, hs, ps = rnn_forward(inputs, targets, hprev, Uh, Wh, Wy, bh, by)

    # Initialise gradient accumulators
    dUh = torch.zeros_like(Uh)
    dWh = torch.zeros_like(Wh)
    dWy = torch.zeros_like(Wy)
    dbh = torch.zeros_like(bh)
    dby = torch.zeros_like(by)
    
    # No future gradient at the end of the sequence
    dhnext = torch.zeros(H, 1)
    
    for t in reversed(range(T)):
        # ------------- Task 2A: Implement each step of the BPTT algorithm --------------
        
        dy = ps[t].clone()
        dy[targets[t]] -= 1 # sum(p_t - e_t) from t=1 to T 
        # TODO: Output layer gradients (Eq 2.2, 2.3)
        dWy += dy @ hs[t].T 
        dby += dy 
        
        dh = Wy.T @ dy + dhnext 
        # TODO: Backprop through tanh, let z = W_h h_{t-1} + U_h x_t + b_h (Eq 2.6)
        dz = (1 - hs[t]**2) * dh 
        
        # TODO: Hidden layer gradients (summed over time)
        dUh += dz @ xs[t].T  # Eq 2.7
        dWh += dz @ hs[t-1].T  # Eq 2.8
        dbh += dz  # Eq 2.9
        
        # Pass gradient to previous timestep (for the next iteration of this loop)
        dhnext = Wh.T @ dz
    
    grads = {'dUh': dUh, 'dWh': dWh, 'dWy': dWy, 'dbh': dbh, 'dby': dby}
    h_final = hs[T-1]
    return loss.item(), grads, h_final


# %%
if is_notebook():
    # Verification: compare manual BPTT gradients against autograd
    _Uh = torch.randn(H, V) * 0.01
    _Wh = torch.randn(H, H) * 0.01
    _Wy = torch.randn(V, H) * 0.01
    _bh = torch.zeros(H, 1)
    _by = torch.zeros(V, 1)

    test_len     = 200
    test_inputs  = [char_to_ix[ch] for ch in data[:test_len]]
    test_targets = [char_to_ix[ch] for ch in data[1:test_len+1]]
    h0 = torch.zeros(H, 1)

    manual_loss, manual_grads, _ = rnn_forward_and_backward(
        test_inputs, test_targets, h0, _Uh, _Wh, _Wy, _bh, _by
    )

    Uh_ag = nn.Parameter(_Uh.clone())
    Wh_ag = nn.Parameter(_Wh.clone())
    Wy_ag = nn.Parameter(_Wy.clone())
    bh_ag = nn.Parameter(_bh.clone())
    by_ag = nn.Parameter(_by.clone())

    ag_loss, _, _, _ = rnn_forward(
        test_inputs, test_targets, h0, Uh_ag, Wh_ag, Wy_ag, bh_ag, by_ag
    )
    
    ag_loss.backward()

    tol = 1e-5
    for name, param_ag, gkey in [
        ("dUh", Uh_ag, "dUh"), ("dWh", Wh_ag, "dWh"),
        ("dWy", Wy_ag, "dWy"), ("dbh", bh_ag, "dbh"), ("dby", by_ag, "dby"),
    ]:
        mg = manual_grads[gkey]
        ag = param_ag.grad
        match = torch.allclose(mg, ag, atol=tol)
        print(f"  {name:4s} {str(list(mg.shape)):12s}  Match={match},  max|diff|={(mg-ag).abs().max():.2e}")


# %% [markdown]
# ---
# ## Task 3: RNN using `nn.RNN`
#
# Rebuild the same model using PyTorch's built-in `nn.RNN` module.
#
# > Complete the `CharRNN` class below. You need to:
# 1. Define a `nn.RNN` module in `__init__`  with `input_size=V`, `hidden_size=H`, `batch_first=True`, which is equivalent to a single layer Elman RNN
# 2. Define an `nn.Linear` layer that maps from H to V (the output layer)
# 3. Implement `forward()`
#
# Refer to the PyTorch documentation [`nn.RNN`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html). The module handles the loop over timesteps internally — what goes in is the full input sequence, and what comes out is all hidden states and the final hidden state.

# %%
## ----------------- Task 3A: Define CharRNN ------------------------------------
class CharRNN(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.H = H
        
        # TODO: Define layers using `nn.RNN` and `nn.Linear`
        self.rnn = nn.RNN(input_size=V, hidden_size=H, batch_first=True)
        self.fc = nn.Linear(H, V)
    
    def forward(self, x, h):
        """
        Args:
            x: input tensor of shape (batch, seq_len, V) — one-hot encoded. Note the last dimension is the vocabulary size
            h: hidden state of shape (batch, 1, H)
        
        Returns:
            output: logits of shape (batch, seq_len, V)
            h_new: final hidden state of shape (batch, 1, H)
        """
        
        # TODO: Pass x and h through self.rnn to get rnn_output and new hidden state
        rnn_output, h_new = self.rnn(x, h)
        
        # TODO: Pass rnn_output through self.fc to get logits
        output = self.fc(rnn_output)
        
        return output, h_new
    
    def init_hidden(self, batch_size=1):
        """Create an initial hidden state with all zeros.
        
        Args:
            batch_size (int): batch size
        
        Returns:
            initial_hidden: initial hidden state of shape (batch_size, 1, H)
        """
        
        # TODO: Return a tensor of shape (batch_size, 1, H)
        initial_hidden = torch.zeros(1, batch_size, self.H)
        
        return initial_hidden



# %%
if is_notebook():
    test_word = "hello"
    test_model = CharRNN(V, H)
    test_x = one_hot_encode([char_to_ix[ch] for ch in test_word], V).T.unsqueeze(0)
    test_h = test_model.init_hidden()
    test_out, test_h_new = test_model(test_x, test_h)
    assert test_out.shape == (1, len(test_word), V), \
        f"Output shape: expected (1, {len(test_word)}, {V}), got {tuple(test_out.shape)}"
    assert test_h_new.shape == (1, 1, H), \
        f"Hidden shape: expected (1, 1, {H}), got {tuple(test_h_new.shape)}"
    assert test_h.shape == (1, 1, H), \
        f"init_hidden shape: expected (1, 1, {H}), got {tuple(test_h.shape)}"
    print(f"CharRNN output shape:  {tuple(test_out.shape)}")
    print(f"CharRNN hidden shape:  {tuple(test_h_new.shape)}")
    print("Task 3A passed!")


# %% [markdown]
# ### 3.1 Training with `CharRNN`

# %%
## ----------------- Task 3B: Train CharRNN and sample from a random timestep ------------------------------------
def sample_from_model(model: nn.Module, seed_ix: int, n: int) -> torch.Tensor:
    """Auto-regressive generation from an nn.Module character model.

    Args:
        model (nn.Module): character model (RNN or LSTM)
        seed_ix (int): index of the seed character to start generation
        n (int): number of characters to generate

    Returns:
        Long tensor of shape (n,) on CPU with character indices.
    """
    model.eval()
    V = model.fc.out_features
    h = model.init_hidden()
    x = torch.zeros(1, 1, V)
    x[0, 0, seed_ix] = 1.0
    generated = []

    with torch.no_grad():
        for _ in range(n):

            # TODO: model forward → logits; softmax to probs
            output, h = model(x, h)
            probs = torch.softmax(output, dim=-1).view(-1)

            probs = probs.clamp(min=1e-12)
            probs = probs / probs.sum()
            ix = torch.multinomial(probs, num_samples=1).item()
            generated.append(ix)

            x = torch.zeros(1, 1, V)
            x[0, 0, ix] = 1.0

    model.train()
    return torch.tensor(generated, dtype=torch.long)



# %%
def train_model(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                data: str, char_to_ix: dict, ix_to_char: dict,
                seq_length: int = 50, n_iters: int = 20000,
                clip_value: float = 5.0
) -> list[float]:
    """Train a character-level model.
    
    Args:
        model (nn.Module): character model (CharRNN or CharLSTM)
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.Module): loss function
        data (str): training data
        char_to_ix (dict): character to index mapping
        ix_to_char (dict): index to character mapping
        seq_length (int): characters per training chunk — BPTT unroll length 
                          (longer = more memory/compute per step and usually more stable gradients; 
                          shorter = more frequent updates).
        n_iters (int): number of iterations
        clip_value (float): clipping value for parameters

        
    Returns:
        losses (list[float]): per-iteration training loss
    """
    V = model.fc.out_features
    data_pointer = 0
    h = model.init_hidden()
    losses = []

    for n_iter in range(n_iters):
        if data_pointer + seq_length + 1 >= len(data):
            h = model.init_hidden()
            data_pointer = 0

        inputs  = [char_to_ix[ch] for ch in data[data_pointer:data_pointer+seq_length]]
        targets = [char_to_ix[ch] for ch in data[data_pointer+1:data_pointer+seq_length+1]]
        x = one_hot_encode(inputs, V).transpose(0, 1).unsqueeze(0)
        y = torch.tensor(targets, dtype=torch.long)

        if (n_iter + 1) % 2000 == 0 or n_iter == 0:
            sampled_ix = sample_from_model(model, inputs[0], 200)
            txt = ''.join(ix_to_char[int(i)] for i in sampled_ix)
            print(f"\n--- Iter {n_iter} ---\n{txt}\n---")

        output, h = model(x, h)
        loss = criterion(output.squeeze(0), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        h = tuple(t.detach() for t in h) if isinstance(h, tuple) else h.detach()

        train_loss = loss.item()
        losses.append(train_loss)
        if n_iter % 2000 == 0:
            print(f"Training loss: {train_loss:.4f}")
        data_pointer += seq_length

    print("Training complete.")
    return losses


# %%
if is_notebook():
    seq_length  = 50

    model_rnn = CharRNN(V, H)
    learning_rate = 1e-1
    optimizer = torch.optim.Adagrad(model_rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses_rnn = train_model(model_rnn, optimizer, criterion, data, char_to_ix, ix_to_char, seq_length)

# %% [markdown]
# **Q3.2:** Look at the samples generated at different timesteps during training. How does the quality of generated text change? What does the model learn first — individual characters, common short sequences, or word-like structures?

# %%
## Your answer:
# Q3.2: The quality of the generated text improves significantly as training progresses. 
# Initially, the model learns the basic character distribution (picking common letters 
# like 'e' and 't' more often than 'z'). Very early on, it also learns the concept of 
# word spacing and character grouping. Next, it picks up common short sequences (bigrams 
# and trigrams) like "th" or "ing". Finally, the model begins to form recognizable, 
# word-like structures and short phrases (e.g., "the", "and"), showing that it is 
# capturing local linguistic patterns from the training corpus.


# %% [markdown]
# ---
# ## Task 4: Long-short Term Memory (LSTM) Character Model
#
# Now replace the RNN with an LSTM. The LSTM addresses the vanishing gradient problem by introducing:
#
# - **Cell state** $c_t$: a "highway" for information that can flow through time with minimal interference
#
# - **Three gates**:
#   - **Forget gate** $f_t$: what to discard from the cell state
#   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) $$
#
#   - **Input gate** $i_t$: what new information to store
#   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
#
#   - **Output gate** $o_t$: what to output from the cell state
#   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) $$
#
#
# The update equations for $c_t$ and output $h_t$ are:
#
# $$c_t = f_t \circ c_{t-1} + i_t \circ \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
# $$h_t = o_t \circ \tanh(c_t)$$
#
# **Key difference from vanilla RNN:** The path from $c_{t-1}$ to $c_t$ only involves an elementwise multiply — no matrix multiplication, no `tanh` saturation — making it far easier for gradients to flow through many timesteps.
#
# > Build the same character-level pipeline as Task 3, but use PyTorch’s [`nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) for the recurrent core (it implements the gating equations above internally).
#

# %%
## ----------------- Task 4A: Define CharLSTM ------------------------------------
class CharLSTM(nn.Module):
    """Character LM with `nn.LSTM` + linear readout (same as `CharRNN`)."""

    def __init__(self, V, H):
        super().__init__()
        self.H = H
        
        # TODO: one LSTM layer (input_size=V, hidden_size=H, batch_first=True)
        self.lstm = nn.LSTM(V, H, batch_first=True)
        
        # TODO: linear layer that maps from the LSTM hidden state to the vocabulary size
        self.fc = nn.Linear(H, V)

    def forward(self, x, hidden):
        """
        Args:
            x: input tensor of shape (batch, seq_len, V). Note the last dimension is the vocabulary size
            hidden: tuple of (h, c), each of shape (1, batch, H)

        Returns:
            output: logits of shape (batch, seq_len, V)
            hidden: tuple of (h_new, c_new)
        """
        
        # TODO: run self.lstm, then self.fc on the sequence of hidden states
        out, (h_new, c_new) = self.lstm(x, hidden)
        logits = self.fc(out)
        
        return logits, (h_new, c_new)

    def init_hidden(self, batch_size=1):
        """Initial (h, c), each (batch_size, 1, H) for nn.LSTM."""
        
        # TODO: return a tuple of two all-zero tensors, each of shape (1, batch_size, H)
        h0 = torch.zeros(1, batch_size, self.H)
        c0 = torch.zeros(1, batch_size, self.H)
        
        return (h0, c0)



# %%
if is_notebook():
    _V, _H = V, H
    _model = CharLSTM(_V, _H)
    assert hasattr(_model, "lstm") and isinstance(_model.lstm, nn.LSTM)

    _seq = one_hot_encode([char_to_ix[ch] for ch in data[:seq_length]], _V).T.unsqueeze(0)
    _h0 = _model.init_hidden()
    _out, (_hm, _cm) = _model(_seq, _h0)

    assert _out.shape == (1, seq_length, _V), \
        f"Output shape: expected {(1, seq_length, _V)}, got {tuple(_out.shape)}"
    assert _hm.shape == (1, 1, _H), f"h shape: expected (1,1,{_H}), got {tuple(_hm.shape)}"
    assert _cm.shape == (1, 1, _H), f"c shape: expected (1,1,{_H}), got {tuple(_cm.shape)}"

    print(f"CharLSTM output shape: {tuple(_out.shape)}")
    print(f"Hidden (h) shape:      {tuple(_hm.shape)}")
    print(f"Cell   (c) shape:      {tuple(_cm.shape)}")
    print("Task 4A passed!")

# %% [markdown]
# ### 4.1 Training with `CharLSTM`
#
# The training loop is nearly identical to Section 3.1. The main differences:
# - `model.init_hidden()` now returns a tuple `(h, c)` 
# - We detach **both** `h` and `c` after each chunk

# %%
if is_notebook():
    model_lstm = CharLSTM(V, H)
    learning_rate_lstm = 1e-1
    optimizer_lstm = torch.optim.Adagrad(model_lstm.parameters(), lr=learning_rate_lstm)
    criterion_lstm = nn.CrossEntropyLoss()

    losses_lstm = train_model(
        model_lstm, optimizer_lstm, criterion_lstm, data, char_to_ix, ix_to_char, seq_length
    )

# %% [markdown]
# ### 4.2 Compare RNN vs LSTM

# %%
if is_notebook():    
    # Plot per-iteration training curves side by side
    plt.figure(figsize=(12, 5))
    min_len = min(len(losses_rnn), len(losses_lstm))
    plt.plot(losses_rnn[:min_len], label=f"RNN (final loss: {losses_rnn[-1]:.4f})", alpha=0.8)
    plt.plot(losses_lstm[:min_len], label=f"LSTM (final loss: {losses_lstm[-1]:.4f})", alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("RNN vs LSTM Training Loss")
    plt.legend(fontsize='medium')
    plt.grid(True, alpha=0.5)
    plt.show()

# %%
if is_notebook():
    # Generate final samples from both models
    print("=" * 60)
    print("FINAL RNN SAMPLE:")
    print("=" * 60)
    sample_ix = sample_from_model(model_rnn, char_to_ix['T'], 500)
    print(''.join(ix_to_char[int(ix)] for ix in sample_ix))

    print("\n" + "=" * 60)
    print("FINAL LSTM SAMPLE:")
    print("=" * 60)
    sample_ix = sample_from_model(model_lstm, char_to_ix['T'], 500)
    print(''.join(ix_to_char[int(ix)] for ix in sample_ix))

# %% [markdown]
# **Q4.3:** Does the LSTM converge faster or achieve a lower loss than the vanilla RNN? Why?
#
# **Q4.4:** These models use character-level tokenisation. What advantages and disadvantages does this have compared to the word-level encoding introduced in Section 6.5.1 of the lecture notes?
#
# **Q4.5:** What's the implication of initialising $(h_0, c_0)$ to 0's? You can play with different initialisations, such as carrying the hidden state forward (while detaching it from the gradient graph).

# %%
## Your answer:
# Q4.3: The LSTM typically converges faster and achieves a lower final loss compared to the vanilla RNN. 
# This is because the LSTM's architecture—specifically the cell state (c_t) and the gating mechanism—mitigates 
# the vanishing gradient problem. The "constant error carousel" of the cell state allows gradients to 
# propagate through many timesteps without being exponentially decayed by repeated matrix multiplications 
# and tanh activations found in the vanilla RNN.

# Q4.4: 
# Advantages: Character-level models have a much smaller vocabulary (limited to the character set), 
# which reduces memory requirements for the embedding/output layers. They are also immune to 
# "Out-of-Vocabulary" (OOV) tokens and can learn to generalize to novel words or handle spelling errors.
# Disadvantages: They require much longer sequence lengths to capture the same semantic context as 
# a word-level model. This puts more strain on the model's long-term memory and makes training 
# computationally more expensive due to the increased number of timesteps.

# Q4.5: Initializing (h0, c0) to zeros is a standard, neutral starting point. However, 
# carrying the hidden state forward across training chunks (stateful training) allows the model 
# to maintain context over a much longer "effective" sequence than the BPTT unroll length (seq_length). 
# This helps the model learn dependencies that span multiple chunks, provided the state is 
# detached from the gradient graph to keep memory usage stable.


# %% [markdown]
# ---
# ## Task 5 (Not Marked, Just For Fun): Investigating Vanishing & Exploding Gradients
#
# During BPTT, the gradient contribution from timestep $s < t - 1$ involves a product of Jacobians:
#
# $$
# \frac{\partial L}{\partial y_t}\;
# \frac{\partial y_t}{\partial h_{t-1}}\;
# \left(\prod_{\tau=s}^{t-1}
# \frac{\partial h_{\tau+1}}{\partial h_\tau}\right)
# \frac{\partial h_s}{\partial \theta}
# $$
#
# And
#
# $$
# \frac{\partial h_{\tau+1}}{\partial h_\tau}
# = \textbf{diag}(1 - h_{\tau+1}^2)\; W_h
# $$
#
# The combined effect of element-wise tanh derivative and the weight matrix $W_h$ determines whether gradients vanish or explode. Theoretically,
#
# - Any $h_{\tau+1}$ close to $\pm 1$ **or** $\sigma_{\max}(W_h) < 1$ $\quad \to \quad$ Gradient **vanishes** 
# - All $h_{\tau+1}$ close to $0$ **and** $\sigma_{\max}(W_h) > 1$ $\quad \to \quad$ Gradient **explodes** 

# %% [markdown]
# ### 5.1 Experiment: How $\sigma_{\max}(W_h)$ Controls Gradient Flow
#
# Reuse the `CharRNN` from Task 3 and **rescale** its hidden-to-hidden weight $W_h$ with singular value decomposition (SVD) so that $\sigma_{\max}(W_h)$ equals each value in **`SIGMAS`**.
#
# We feed a real text sequence, use a scalar loss built from the **final** hidden state, and plot $\|\partial\mathcal{L}/\partial h_t\|$ at every time step.
#
# **What to expect:** $\sigma_{\max} < 1$ → norms **shrink** backward in time (vanishing); $\sigma_{\max} > 1$ → norms **grow** (exploding); $\sigma_{\max} \approx 1$ sits near the **boundary** between the two behaviours.

# %%
import copy

def make_rnn_with_sigma(sigma_max, V, H, model=None, seed=0):
    """Create/deep-copy a CharRNN and rescale Wh.
    """
    if model is None:
        m = CharRNN(V, H)
    else:
        m = copy.deepcopy(model)
    with torch.no_grad():
        W = m.rnn.weight_hh_l0
        U, S, Vh = torch.linalg.svd(W)
        m.rnn.weight_hh_l0.copy_(U @ torch.diag(S / S.max() * sigma_max) @ Vh)
    return m

def gradient_norms(model, xs):
    """
    Returns ||dL/dh_t|| per timestep.
    xs: Tensor of shape (1, seq_len, V)
    """
    is_lstm = hasattr(model, "lstm")
    T = xs.shape[1]

    if is_lstm:
        h, c = model.init_hidden()
    else:
        h = model.init_hidden()

    hs = []
    for t in range(T):
        x_t = xs[:, t : t + 1, :]  # (1, 1, V)
        if is_lstm:
            _, (h, c) = model(x_t, (h, c))
        else:
            _, h = model(x_t, h)
        h.retain_grad()
        hs.append(h)

    hs[-1].mean().backward()
    return [ht.grad.norm().item() for ht in hs]


# %%
if is_notebook():
    seq_length = 30
    xs_exp = one_hot_encode(
        [char_to_ix[ch] for ch in data[:seq_length]], V
    ).T.unsqueeze(0) 
    SIGMAS = [0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 3.0]

    cmap = plt.cm.RdYlGn_r
    colors = [cmap(i / (len(SIGMAS) - 1)) for i in range(len(SIGMAS))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for sigma, col in zip(SIGMAS, colors):
        norms = gradient_norms(make_rnn_with_sigma(sigma, V, H), xs_exp)
        ts = list(range(len(norms)))
        norms_safe = [max(v, 1e-20) for v in norms]
        label = rf'$\sigma_{{\max}}={sigma}$'
        axes[0].plot(ts, norms, color=col, lw=2, label=label)
        axes[1].semilogy(ts, norms_safe, color=col, lw=2, label=label)

    for ax in axes:
        ax.set_xlabel('Time step $t$')
        ax.legend(fontsize=12)
        ax.axvline(seq_length - 1, color='gray', lw=1, ls='--')

    axes[0].set_ylabel(r'$\|\partial\mathcal{L}/\partial h_t\|$')
    axes[0].set_title('Linear scale')
    axes[0].set_ylim(0, 0.5)
    axes[1].set_ylabel(r'$\|\partial\mathcal{L}/\partial h_t\|$ (log-scaled)')
    axes[1].set_title('Log scale')
    plt.suptitle(r'Exp 5.1: Gradient norm for different $\sigma_{\max}(W_h)$',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **What to observe:**
#
# - $\sigma_{\max} < 1$: gradient norms usually **decay** backward in time — on the log plot this often looks like a **downward** trend toward smaller $t$.
# - $\sigma_{\max} \approx 1$: often near the **transition** between decay and growth.
# - **Large** $\sigma_{\max} \gg 1$ (e.g. $3$): norms often **grow** toward earlier time steps — **exploding** behaviour on the log plot (upward trend to the left).

# %% [markdown]
# ### Practical Effect: Generated Text After Perturbation

# %%
if is_notebook():
    sampled_length = 200
    print(f"{'σ_max':>8s}   Sample (first {sampled_length} chars)")
    print("=" * 130)
    for i, sigma in enumerate(SIGMAS):
        m = make_rnn_with_sigma(sigma, V, H, model=model_rnn)
        sample_ixs = sample_from_model(m, char_to_ix["\n"], sampled_length)
        text = "".join(ix_to_char[int(ix)] for ix in sample_ixs).replace("\n", "↵")
        print(f"{sigma:8.2f}  {text[:sampled_length]}")

# %% [markdown]
# **Q5.2:** In practice, does $\sigma_{\max} > 1$ *always* cause exploding gradients? What role does
# $\tanh$ saturation ($h_{\tau+1} \approx \pm 1$, so $(1 - h_{\tau+1}^2) \approx 0$) play?
#
# **Q5.3:** Can gradient clipping *solve* the vanishing gradient problem, or does it only address exploding? Play around with the *clip_value* parameter in `train_model()`.
#
# **Q5.4:** Why is the LSTM robust to changes in $\sigma_{\max}(W_h)$ that destabilise the RNN?
# Relate your answer to the cell-state gradient $\partial c_t / \partial c_{t-1} = \text{diag}(f_t)$.

# %%
## Your answer:
# Q5.2: No, sigma_max > 1 does not always lead to exploding gradients. 
# The tanh saturation plays a crucial role: when h_t approaches ±1, the derivative 
# (1 - h_t^2) goes to 0. This effectively "shuts down" the gradient flow through 
# that specific hidden neuron, preventing the product of matrices from growing 
# uncontrollably even when the weight matrix itself is large.

# Q5.3: Gradient clipping only addresses the exploding gradient problem. 
# It caps the norm of the gradient at a specific value to prevent huge updates 
# from destabilizing the model. It does not solve the vanishing gradient problem; 
# it cannot amplify a gradient that is already near zero or help it pass 
# through many layers/timesteps more effectively.

# Q5.4: The LSTM is robust because its cell state gradient is not governed 
# by a weight matrix (W_h) directly, but by the forget gate (f_t). 
# Since the derivative of the cell state w.r.t. the previous cell state 
# is simply f_t, the gradient can be passed through many steps linearly 
# without being scaled by the singular values of a weight matrix at every step. 
# This "additive" update path is inherently more stable than the 
# "multiplicative" path in vanilla RNNs.

# %%
