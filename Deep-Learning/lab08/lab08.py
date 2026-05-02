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
# <h2>Lab #8: Attentions and Transformer</h2>
# Semester 1, 2026<br>
# </center>
#
# **Due**: 11:59pm on Sunday 3 May, 2026.<br>
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
student_name = "Gemini CLI"
student_id = "u7654321"

# %%
import math
import os
import sys
import getpass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def is_notebook():
    return 'ipykernel' in sys.modules

print("User: {} ({}, {})".format(getpass.getuser(), student_name, student_id))
print("Python Version: {}".format(sys.version))
print("PyTorch Version: {}".format(torch.__version__))


# %% [markdown]
# ## Overview
# In this lab, we will build our own **BABY GPT** that can be trained on your laptop. We will try to solve the same next token prediction task from Lab 6 but with our decoder-only transformer architecture. Our baby GPT will have multiple submodules, including tokenisations, learnable word embeddings, masked multi-head self atttention etc. We will code each of these independently and assemble them in the end. The entire architecture is illustrated below,
# <center>
# <img src="architecture.png" width="700" height="500">
# </center>
#
# While self-attention mechanism feels magical, at the end of this lab, we will explore some important pathological cases of the attention mechanism.
#
# Make sure you understand the architecture above, as you will code each of the modules next. 
#
# Let's get started !!

# %% [markdown]
# ## TASK 1: Character Tokeniser
#
# Similar to Lab 6, the first thing we need to do is to tokenise our text data. In particular, we want to convert each character in the data to a single index instead of one-hot encoding. For example, with a vocabulary [a,b,c,d], 'a' should be mapped to 0 instead of [1,0,0,0]. The reason for this is that we will later on use learnable word embeddings instead of sinusoidal word embeddings taught in the lecture.

# %%
class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        # TODO Task 1a: given a text (str), return a list of encoded 
        # integer indices correspondings to each character in the string
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        # TODO Task 1b: given a list of encoded indices, return the 
        # corresponding string
        return "".join([self.itos[i] for i in ids])


# %%
if is_notebook():
    # Test the CharTokenizer
    text = "hello world"
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print("Original text:", text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    assert text == decoded, "Decoded text does not match original"
    print("Task 1 tests passed!")


# %% [markdown]
# ## Task 2: Text Dataset Loader
# In Lab6, we load characters one by one, however, in this lab we will look at a small context window instead. In such a contextual window, the attention mechanism can infer which parts of the character, (sub)-words are important for generating the next token.

# %%
class TextDataset(Dataset):
    """
    Splits a token sequence into (input, target) windows of length context_len.
    target[t] = input[t+1]  (next-token prediction).
    """

    def __init__(self, tokens: torch.Tensor, context_len: int):
        self.tokens      = tokens
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx: int):
        # TODO Task2a: given a specific index query, 
        # return two items.
        # - x: a short context window of tokens starting
        # at the queried index. The window size is self.context_len
        # - y: a target, which is the next token window defined as y[t] = x[t+1]
        #
        # No need to worry about going past the end of the tokens
        # idx will always be within the range len(TextDataset).
        x = self.tokens[idx : idx + self.context_len]
        y = self.tokens[idx + 1 : idx + self.context_len + 1]
        return x, y


# %%
if is_notebook():
    # Test the txt dataset
    toks = torch.randint(0, 10, (20,))
    dataset = TextDataset(toks, context_len=5)
    x, y = dataset[0]
    assert len(x) == 5 and len(y) == 5, "Context and target should both have length 5"
    assert torch.all(x[1:] == y[:-1]), "Target should be the next token of the input"
    print("Task 2 tests passed")


# %% [markdown]
# ## Task 3: Learnable Word Embedding Module
# With our dataset helper functions implemented, we now shift our attention to architecture design of our baby GPT. Similar to GPT-2, we next implement a learnable word embedding module that takes in tokens and their positional information and embed them to latent features for the next stage. Specifically, given a token index $x\in \{0, \dots, |\mathcal{V}|\}$ (where $|\mathcal{V}|$ is the vocabulary size) and its positional embedding $p\in \{0, \dots, T\}$ (where $T$ is the contexual window size), the embedding module is defined as,
# $$
# z = W_x x + W_p p
# $$
# where $W_x$ and $W_p$ are the learnable parameters. Essentially our work embedding module is just a special linear feedforward layer! However, a catch is that to save memory, we choose to use character index token instead of one-hot encoding, therefore instead of `nn.linear`, we will neeed to use a special module PyTorch provides, `nn.Embedding`. Read the [embedding documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and implement the module below.

# %%
class WordEmbedding(nn.Module):
    """
    Combines a trainable token embedding with a trainable positional embedding.

    Parameters
    ----------
    vocab_size : int
    embed_dim  : int
    context_len: int   – maximum sequence length
    """

    def __init__(self, vocab_size: int, 
                 embed_dim: int,
                 context_len: int):
        super().__init__()
        # TODO Task 3a: define the learnable parameters for our embedding module
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(context_len, embed_dim)
        
        self.context_len = context_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : LongTensor [B, T], B is batch size, T is contextual window size

        Returns
        -------
        Tensor [B, T, embed_dim]
        """
        
        _, T = idx.shape
        assert T <= self.context_len, "Sequence longer than context_len"
        # TODO Task 3b: implement the forward pass of our embedding module
        tok_emb = self.token_emb(idx) # [B, T, embed_dim]
        pos_idx = torch.arange(T, device=idx.device) # [T]
        pos_emb = self.pos_emb(pos_idx) # [T, embed_dim]
        return tok_emb + pos_emb


# %%
if is_notebook():
    # Test the WordEmbedding module
    word_emb = WordEmbedding(vocab_size=20, embed_dim=4, context_len=3)
    assert isinstance(word_emb.token_emb, nn.Embedding), "token_emb should be an instance of nn.Embedding"
    assert isinstance(word_emb.pos_emb, nn.Embedding), "pos_emb should be an instance of nn.Embedding"
    idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = word_emb(idx)
    assert out.shape == (2, 3, 4), "Output shape should be [B, T, embed_dim]"
    print("Task 3 tests passed")


# %% [markdown]
# ## Task 4: Masked Multi-Head Self Attention Module
# As the name suggests, a Masked Multi-Head Self Attention module builds on top of the scaled dot product attention by 
# 1. enforcing the Q, K, V matrices are generated linearly from a same input (hence the word "self"),
# 2. generating multiple attention outputs $Z_i = \text{attention}(QW^Q_i, KW^K_i, VW^V_i)$ and concatenate them together $y = [Z_1,\dots,Z_h]$,
# 3. using casual masks to ensure tokens can only attend to other tokens in the past.
# 4. using a final linear layer to process the concatenated multi-head outputs.
#
# See lecture slide 7-11 for more details. Here we going to implement the module from scratch. 
#
# **Warning** You are not allowed to use `nn.attention` or `nn.MultiheadAttention` in this lab. No marks will be given if you do.
#
# Here are some hints,
# 1. Given a batched inputs with dimension `[B, T, embed_dim]` where B is the batch size, T is the contextual window length, we want to further split the embedded tokens into mini batches for the multi-head attention mechanism. Specifically, for each embedded vector, we will split it into $n$ parts, where $n$ is number of heads. What is the dimension of the input vector for each head?
# 2. Multi-head attention process each mini batches independently, but should attend to all the elements in the window within each mini batch, hence the shape of Q, K, V needs to be `[B, num_heads, T, head_dim]`. Why?
# 3. Causal mask $M$ is a **fixed** (non-learnable) lower triangular matrix (what is its dimensions?) Specifically, $$y = \text{softmax}(Q^TK/\sqrt{d} \odot M ) V.$$ However, for zeroed out scores, $exp(0) = 1$, which is wrong, instead we should set the upper triangular part of the mask to be -inf so that after the exponential function in softmax, we get the correct output because $exp(-\inf) = 0$. Check out the magic function `tensor.masked_fill`.
#

# %%
class MaskedMultiHeadSelfAttn(nn.Module):
    """
    Causal (auto-regressive) multi-head self-attention.

    A lower-triangular mask prevents each position from attending to future
    tokens.

    Parameters
    ----------
    embed_dim : int, attention embedding dimensions, must be divisible by num_heads
    num_heads : int, number of attention heads
    context_len: int, maximum token input sequence length.
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 context_len: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim   = embed_dim
        self.num_heads   = num_heads

        # TODO Task 4a: define the trainable parameters
        # and the causal mask (use self.register_buffer)
        # to push non-trainable torch tensors into GPU
        # if you use a GPU for training.
        
        self.head_dim    = embed_dim // num_heads

        # Fused Q, K, V projection
        self.q_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim) # final linear layer to process all concatenated attentions

        # Create a 2D Causal mask matrix (it should be fixed, not a learnble tensor.)
        mask = torch.tril(torch.ones(context_len, context_len))

        # Here we register the fixed parameters into the GPU memeory.
        # Do not delete this line even if you use CPU.
        self.register_buffer("mask", mask.view(1, 1, context_len, context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, T, embed_dim]

        Returns
        -------
        Tensor [B, T, embed_dim]
        """

        # TODO Task 4b: implement the forward pass
        # for our module.
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T, hs]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T, hs]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T, hs]

        # scale dot product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) # [B, nh, T, T]
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        return self.out_proj(out)


# %%
if is_notebook():
    # Test the masked multi-head self-attention module
    attn = MaskedMultiHeadSelfAttn(embed_dim=8, num_heads=2, context_len=4)
    assert attn.head_dim == 4, "Head dimension should be 4"
    idx = torch.randn(2, 4, 8)  # [B, T, embed_dim]
    out = attn(idx)
    assert out.shape == (2, 4, 8), "Output shape should be [B, T, embed_dim]"
    print("Task 4 tests passed.")


# %% [markdown]
# ## Task 5: Decoder Attention Block
# Since our transformer is tasked to solve word generation problems, we drop the encoder attention blocks proposed by Vaswani et.al. (NeurIPs, 2017), but only implementing the decoder blocks. 
#
# The decoding attention block wraps the attention mechanism we implemented earlier with extra learnable layers. In particular given an input $x_0$, our attention block should apply layer norms, then compute the attention results. A skip connection is applied between the initial input and the attention output. In short, $x_1 = x_0 + \text{attention}(LN_1(x_0))$. We will then use a feedforward layer (MLP with 1 hidden layer and ReLU activation) to process layer normalised $x_1$ with another round of skip connection for this layer, i.e. $x_2 = x_1 + FF(LN_2(x_1))$. Don't put an activation layer at the output of the feedforward layer, only use ReLU for the hidden layer.
#
# Refer to the decoder block in the architecture illustration found at the top of this notebook for a visualisation of the information flow. Note we should use seperate layer normalisation modules, as they are actually learnable. Check out the documentation for `nn.LayerNorm` before using it!

# %%
class AttentionBlock(nn.Module):
    """Single decoder block: LayerNorm → MHA → residual → LayerNorm → FFN → residual.
    FFN is a simple 2-layer (1 hidden layer) MLP with ReLU nonlinearity in between, no ReLU at its output.
    Parameters
    ----------
    embed_dim : int, attention embedding dimensions, must be divisible by num_heads
    num_heads : int, number of attention heads
    ff_dim    : int, hidden dimension of the feed-forward network, it can be different from embed_dim
    context_len : int, maximum token input sequence length
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, context_len: int):
        super().__init__()
        #TODO Task 5a: define all the components needed 
        # for our decoding attention block. Note you should
        # use seperate layer normalisation for each stage.
        # use nn.Sequential() for self.ff to put the final feedforward layers (activation + linear + activation) together.
        
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = MaskedMultiHeadSelfAttn(embed_dim, num_heads, context_len)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# %%
if is_notebook():
    # Test the decoding attention block
    block = AttentionBlock(embed_dim=8, num_heads=2, ff_dim=16, context_len=4)
    idx = torch.randn(2, 4, 8)
    out = block(idx)
    assert out.shape == (2, 4, 8), "Output shape should be (2, 4, 8)"
    print("Task 6 tests passed.")

# %% [markdown]
# ## Task 6: Transformer
# We have implemented everything we need to assemble our baby GPT's transformer backbone. Now let's put everything together to define the entire architecture (again see the illustrations at the top for the overall architecture design). 
#
# ### Forward pass
#
# The input tokens firstly pass through our learnable word embedding module. Then the embedded tokens pass through multiple decoding attention blocks. The final output is passed through a final layer normalisation followed by a linear layer to get the predicted logits.
#
# We have already initialised all the modules for you. Your task is to implement the forward pass of the entire architecture and how we can sample from the transformer.
#
# Note we can compute losses on the fly during the forward pass by feeding the target tokens to our transformer. Hence, in the forward method you need to compute the loss for each prediction (what was the loss you used in lab6)?
#
# ### Token generation
#
# We use beaming to generate future tokens. Specifically, given some starting tokens (e.g. "I study"), we proceed to query vocabulary logits from our model. Imporatantly, after each query, we keep the top K highest logits and set the rest of logits to `-inf`. This is to ensure that we always focus on the most likely sentences without becoming too random. These logits should be converted to a multinomial distribution to generate an actual sample. 
#
# Ensure the contextual window for generating the next token does not exceed the maximum window length our transformer can handle. The contextual window should always be at the end of our predicted tokens, e.g. pred[:-3] for a window size 3!

# %%
CONTEXT_LEN  = 16      # sequence length (tokens)
EMBED_DIM    = 128      # model width
NUM_HEADS    = 2        # attention heads  (must divide EMBED_DIM)
NUM_LAYERS   = 2        # transformer blocks
FF_DIM       = 256      # feed-forward hidden size

BATCH_SIZE   = 16
EPOCHS       = 10
LR           = 3e-4
EVAL_SPLIT   = 0.1      # fraction of tokens held out for validation
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH    = "data/textData.txt"

class BabyGPT(nn.Module):
    """
    GPT style decoder-only transformer.

    Architecture
    ------------
    WordEmbedding
    N × AttentionBlock  (MaskedMultiHeadSelfAttn + FFN)
    LayerNorm
    Linear head  →  vocab logits

    Parameters
    ----------
    vocab_size  : int
    embed_dim   : int
    num_heads   : int
    num_layers  : int (number of attention blocks)
    ff_dim      : int
    context_len : int
    """

    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
                 ff_dim: int = FF_DIM, context_len: int = CONTEXT_LEN):
        super().__init__()
        self.embedding = WordEmbedding(vocab_size, embed_dim, context_len)
        self.blocks    = nn.Sequential(*[
            AttentionBlock(embed_dim, num_heads, ff_dim, context_len)
            for _ in range(num_layers)
        ])
        self.ln_f      = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share token embedding weights with output projection
        self.head.weight = self.embedding.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        """
        Parameters
        ----------
        idx     : LongTensor [B, T]
        targets : LongTensor [B, T]  (optional; used during training)

        Returns
        -------
        logits : Tensor [B, T, vocab_size]
        loss   : scalar Tensor or None, combined losses of all predicted tokens
        """
        # TODO Task 6a: implement the forward pass of the 
        # transformer and the self.head to get the final logits
        x = self.embedding(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # TODO Task 6b: implement the loss for each predicted
        # logits. Recall target[t] = x[t+1].
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Auto-regressively sample `max_new_tokens` tokens.

        Parameters
        ----------
        idx           : LongTensor [1, T]  – prompt token ids
        max_new_tokens: int
        top_k         : int or None        – restrict sampling to top-k logits

        Returns
        -------
        LongTensor [1, T + max_new_tokens]
        """
        # TODO Task 6c: implement next token generation
        ctx = self.embedding.context_len
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # [1, V]
            if top_k is not None:
                # use torch.topk to pull out the top k logits
                v, _ = torch.topk(logits, top_k)
                # then set the rest of the logits to -inf
                logits[logits < v[:, [-1]]] = float("-inf")
            # compute the probabilities of the logits using softmax
            probs = F.softmax(logits, dim=-1)

            # sample from multinomial distribution
            next_t = torch.multinomial(probs, num_samples=1)

            # append the new token into the idx list
            idx = torch.cat((idx, next_t), dim=1)
        return idx


# %%
if is_notebook():
    # Test the transformer architecture
    B, T, V = 2, 4, 20          # batch size, context length, vocab size
    baby_gpt = BabyGPT(
        vocab_size=V,
        embed_dim=8, 
        num_heads=2, 
        num_layers=2, 
        ff_dim=16, 
        context_len=T)

    # Need T+1 tokens so we can form T input-target pairs via a 1-step shift:
    #   in_idx[t]  = idx[t]        (token at position t)
    #   targets[t] = idx[t+1]      (the next token = ground truth)
    idx = torch.randint(0, V, (B, T + 1))   # [B, context_len + 1]
    in_idx  = idx[:, :-1]                   # [B, T]  - model input
    targets = idx[:, 1:]                    # [B, T]  - next-token targets

    print(f"Input  shape : {in_idx.shape}")   # expected (2, 4)
    print(f"Target shape : {targets.shape}")  # expected (2, 4)

    assert in_idx.shape  == (B, T),    f"Input shape should be [B, T] = {(B, T)}"
    assert targets.shape == (B, T),    f"Targets shape should be [B, T] = {(B, T)}"

    logits, loss = baby_gpt(in_idx, targets)
    assert logits.shape == (B, T, V),  f"Logits shape should be [B, T, vocab_size] = {(B, T, V)}"
    assert loss is not None,           "Loss should not be None when targets are provided"
    assert loss.item() > 0,            "Loss should be a positive scalar"

    # Generate test: start from a prompt of length T_prompt, produce T_new new tokens
    T_prompt, T_new = 2, 5
    prompt = torch.randint(0, V, (1, T_prompt))   # [B=1, T_prompt]
    out = baby_gpt.generate(prompt, max_new_tokens=T_new, temperature=1.0, top_k=3)
    assert out.shape == (1, T_prompt + T_new), \
        f"Generated output shape should be [1, T_prompt + T_new] = {(1, T_prompt + T_new)}"
    print("Task 6 tests passed.")


# %% [markdown]
# ## Baby GPT in Action !
# Finally we are ready to train and test our baby GPT! And we have implemented the entire training pipeline for you. Make sure familiar yourself with the code, they are quite important especially for your group project. Once you are happy, just hit the button, grab a coffee (training time ~8 mins on a Mac Air, could take up to 20 - 30 mins on some other laptops) and wait for the magic to happen, you should see epoch stats printed out gradually.

# %%
def train(model: BabyGPT, loader: DataLoader,
          optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model: BabyGPT, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / len(loader)

if is_notebook():
    def main():
        # ── Load & tokenise ───────────────────────────────────────────────────────
        with open(DATA_PATH, encoding="utf-8") as f:
            text = f.read()
    
        tokenizer = CharTokenizer(text)
        print(f"Vocabulary size : {tokenizer.vocab_size}")
        print(f"Total characters: {len(text):,}")
    
        all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
        # ── Train / val split ─────────────────────────────────────────────────────
        split      = int(len(all_tokens) * (1 - EVAL_SPLIT))
        train_tok  = all_tokens[:split]
        val_tok    = all_tokens[split:]
    
        train_set  = TextDataset(train_tok, CONTEXT_LEN)
        val_set    = TextDataset(val_tok,   CONTEXT_LEN)
    
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, drop_last=False)
    
        print(f"Train samples: {len(train_set):,}  |  Val samples: {len(val_set):,}")
    
        # ── Build model ───────────────────────────────────────────────────────────
        model = BabyGPT(
            vocab_size  = tokenizer.vocab_size,
            embed_dim   = EMBED_DIM,
            num_heads   = NUM_HEADS,
            num_layers  = NUM_LAYERS,
            ff_dim      = FF_DIM,
            context_len = CONTEXT_LEN,
        ).to(DEVICE)
    
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters      : {n_params:,}\n")
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR / 10
        )
    
        # ── Train / eval loop ─────────────────────────────────────────────────────
        best_val = float("inf")
        for epoch in range(1, EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer)
            val_loss   = evaluate(model, val_loader)
            scheduler.step()
    
            print(f"Epoch {epoch:>3}/{EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"ppl={math.exp(val_loss):.1f}")
    
        # ── Generate a sample ─────────────────────────────────────────────────────
        print("\n── Sample generation (temperature=0.8, top_k=40) ──\n")
        model.eval()
    
        seed_text = "I study"
        seed_ids  = torch.tensor(tokenizer.encode(seed_text),
                                 dtype=torch.long, device=DEVICE).unsqueeze(0)
        out_ids   = model.generate(seed_ids, max_new_tokens=300,
                                   temperature=0.8, top_k=40)
        print(tokenizer.decode(out_ids[0].tolist()))

    main()


# %% [markdown]
# ## Remarks
#
# With such a small dataset and model, our baby GPT performed well. The generated samples read more like a real sentence compared to RNNs, though still it doesn't generate full meaningful sentences. You can tune the context size, number of heads, layers to create bigger models that can handle longer contextual information and hopefully better sentences can be generated, though be aware this may consume a lot of computing resources.
#
# As you probably have noticed, overfitting has ocurred during training (the training error decreased but the evaluation error. Some techniques to improve generalisation is to use random droppouts at the output of some modules, see `nn.Dropout`. 
#
# Empirically transformers have been observed to require large amount of data to train. The lite version of GPT-2 has more than 124 million parameters, and trained on 40 GB internet scale text data for days on multiple GPUs. Our baby GPT here is just a taste of what it feels like. 
#
# **Warning**
# If you changed the default parameters and other settings, make sure change them back for submitting and marking.
#

# %% [markdown]
# ## Task 7: Is Attention All You Need?
#
# Attentions and transformers have became the central backbone of today's AI. However, they might fail in different situations. Let's explore some of these pathological cases for the most basic self-attention mechanism.

# %%
def self_attention(Q: torch.tensor, K: torch.tensor, V:torch.tensor):
    '''
    given a two dimensional matrices (no batches), return
    the simple dot product self-attention.
    '''
    # TODO Task 7a: implement the basic self-attention 
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, V)


# %% [markdown]
# ### Task 7b: Attention Sink
# The first case we explore allocate all attentions to one single token and may erase the rest of the information, which means later layers will never process other parts of the input. 

# %%
def attention_sink():
    Q = torch.tensor([[0, 0, 1,],
                      [0, 0, 1,],
                      [0, 0, 1.]])
    V = torch.eye(3)
    
    # TODO Task 7a: design the 3x3 matrix
    # K such that the final attention
    # output allocate the same attention
    # to the first two tokens, but allocate
    # at least twice higher attention to the 
    # final token. i.e.
    # att[:, 0] == att[:, 1]
    # att[:, 0] * 2 < att[:, 2]
    
    K = torch.tensor([[0, 0, 0.0],
                      [0, 0, 0.0],
                      [0, 0, math.log(5.0)]])
    
    att = self_attention(Q, K, V)
    return att

if is_notebook():
    print(attention_sink())


# %% [markdown]
# ### Task 7c: Uniform Attention
#
# Now let's look at the opposite case, uniform attention refers to treating all tokens as the same. When will this happen? When the attention outputs are all the same, all information is treated equally, making later layers hard to distinguish which parts are relevent for the output. 

# %%
def uniform_attention():
    # TODO Task 7b:
    # design the 3x3 Q, K matrices 
    # such that the final attention
    # is uniform 
    # i.e. att[i, j] = att[x, y]
    # for all i, j, x, y in {0, 1, 2}
    Q = torch.zeros(3, 3)
    K = torch.zeros(3, 3)
    
    V = torch.eye(3)
    
    att = self_attention(Q, K, V)
    return att

if is_notebook():    
    print(uniform_attention())

# %% [markdown]
# ### Final Remark
#
# In fact there are many other pathological cases. For instance, in multi-head attentions, some heads may output similar results (redundancy) or output diminishing results (dead heads). Without positional encoding, latter tokens will dominate, causing greedy attention. And in fact, the FF layer, residual connections in side the attention block you implemented are all important too. For instance, [Dong et.al., (ICML, 2021)](https://proceedings.mlr.press/v139/dong21a.html) shows that removing the skip connections and MLPs, the output of attention blocks converge quickly to a degenerated rank-1 matrix. Therefore the key success of transformers is not just attention, but the entire information flow within the transformer architecture!

# %%
