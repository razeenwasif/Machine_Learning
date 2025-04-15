#################################### Chat bot ####################################

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import os
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Config ---
DEFAULT_BLOCK_SIZE = 128
MAX_NEW_TOKENS = 5000 

# --- File paths ---
VOCAB_FILE = os.path.join(".", "dataset_code_contests", "vocab.txt")
FINAL_MODEL_STATE_DICT_FILE = "final_model_state.pth"
CONFIG_FILE = "final_model_config.pkl"


############################## Data Setup ########################################
chars = ""
DEFAULT_VOCAB_SIZE = None
try:
    with open(VOCAB_FILE, "r", encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
    DEFAULT_VOCAB_SIZE = len(chars)

except FileNotFoundError:
    print(f"Error: Vocab file not found at {VOCAB_FILE}")
    exit()

except Exception as e:
    print(f"Error reading vocab file: {e}")
    exit()

if DEFAULT_VOCAB_SIZE is None:
    print("Error: Vocabulary size could not be determined.")
    exit()

# --- Tokenizer ---
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_int.get(c, 0) for c in s]
decode = lambda l: ''.join([int_to_string.get(i, '?') for i in l])


############################### Model Classes ####################################

class Head(nn.Module):
    """one head of self_attention"""

    tril: Tensor

    def __init__(self, head_size, block_size, n_embed, dropout_rate) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        # register buffer correctly assigns the tensor at runtime 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        batch, time, channel = x.shape 
        k = self.key(x)   # b, t, hs 
        q = self.query(x) # b, t, hs
        # compute attention scores ("affinities")
        # := (b, t, hs) @ (b, hs, t) -> (b, t, t)
        weight = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        weight = weight.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1) # b, t, t 
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        v = self.value(x) # (b, t, hs)
        out = weight @ v # (b,t,t) @ (b,t,hs) -> (b,t,hs)
        return out

################################################################################

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embed, dropout_rate) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), 
            nn.Linear(4 * n_embed, n_embed), nn.Dropout(dropout_rate),
        )
        
    def forward(self, x): return self.net(x)

################################################################################

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout_rate) -> None:
        super().__init__()
        # four heads running in parallel 
        self.heads = nn.ModuleList([
            Head(
                head_size, block_size, n_embed, dropout_rate
            ) for _ in range(num_heads)
        ])
        self.project = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # concatenate each head together along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (channel dim)
        out = self.dropout(self.project(out))
        return out 

################################################################################

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_head, block_size, dropout_rate):
        # n_embed: embedding dimension; n_head: the number of heads we'd like
        super().__init__()
        # number of features each of the heads are capturing
        head_size = n_embed // n_head 
        self.self_attention = MultiHeadAttention(
            n_embed, n_head, head_size, block_size, dropout_rate
        )
        self.feed_forward = FeedForward(n_embed, dropout_rate)
        # helps smooth out features
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x)) 
        x = x + self.feed_forward(self.layer_norm2(x))   
        return x

################################################################################

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout_rate):
        super().__init__()
        
        # store hyperparams 
        self.config = {
            'vocab_size':vocab_size, 'block_size':block_size, 'n_embed':n_embed,
            'n_head':n_head, 'n_layer':n_layer, 'dropout_rate':dropout_rate
        }

        # pred lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # positional encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # create four decoder layers Sequentially
        self.blocks = nn.Sequential(*[
            Block(
                n_embed, n_head, block_size, dropout_rate
            ) for _ in range(n_layer)
        ])

        self.final_layer_norm = nn.LayerNorm(n_embed) # help converge
        self.lang_model_head = nn.Linear(n_embed, vocab_size)

    # Initialize weights
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # logits are floating points nums that are normalized
        # [x,y,z] -> [x/(x+y+z), y/(...), ...]
        # essentially a probability distribution of what we want to predict
        # think of the time dim as the sequence of integers
        # channels is the vocab_size
        batch, time = idx.shape

        # use block size from model's config 
        # add fallback if config missing
        block_size = self.config.get('block_size', DEFAULT_BLOCK_SIZE) 

        # Use self.config['block_size'] or ensure 'time' doesn't exceed it
        if time > block_size:
           idx = idx[:, -block_size:]
           time = block_size

        # idx and targets are both (b, t) tensor of integers
        tok_embedding = self.token_embedding_table(idx) # (b, t, c)
        pos_embedding = self.position_embedding_table(torch.arange(time, device=idx.device)) # (t, c) 
        x = tok_embedding + pos_embedding # (batch, time, channel)
        x = self.blocks(x) # (batch, time, channel)
        x = self.final_layer_norm(x) # (batch, time, channel)
        logits = self.lang_model_head(x) # (batch, time, vocab_size)
        
        loss = None
        if targets is not None:
            batch_l, time_l, channel_l = logits.shape
            # reshape to (N, C) for cross_entropy
            logits_reshaped = logits.view(batch_l * time_l, channel_l)
            targets_reshaped = targets.view(batch_l * time_l)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
        
        return logits, loss 
    
    # generate tokens
    def generate(self, idx, max_new_tokens):
        block_size = self.config.get('block_size', DEFAULT_BLOCK_SIZE)
        # index is (batch, time) array of indices in the current context 
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens 
            idx_cond = idx[:, -block_size:]
            # get predictions using the cropped content 
            logits, _ = self.forward(idx_cond)
            # focusing only on the last time step (single prev char)
            logits = logits[:, -1, :] # becomes (batch, channel)
            # apply softmax to get probs dist
            probs = F.softmax(logits, dim=-1) # (batch, channel)
            # sample from the distribution
            idx_nxt = torch.multinomial(probs, num_samples=1) # (batch, 1)
            # append sample idx to the running sequence
            idx = torch.cat((idx, idx_nxt), dim=1) # (B, T+1)
        return idx

###################### Load Model ###########################################

print("Loading model configuration and state dictionary...")
model = None
model_config = None

# load config
try:
    with open(CONFIG_FILE, "rb") as f:
        model_config = pickle.load(f)
    print(f"Loaded model config from {CONFIG_FILE}:", model_config)

except FileNotFoundError:
    print(f"Error: Model file not found at {CONFIG_FILE}")
    exit()

except pickle.UnpicklingError:
     print(f"Error: Could not unpickle model file. It might be corrupted or saved with incompatible code.")
     exit()

except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()

if model_config is None:
     print("Model configuration could not be loaded. Exiting.")
     exit()

# instantiate model using config 
try:
    model = GPTLanguageModel(
        vocab_size=model_config['vocab_size'],
        block_size=model_config['block_size'],
        n_embed=model_config['n_embed'],
        n_head=model_config['n_head'],
        n_layer=model_config['n_layer'],
        dropout_rate=model_config['dropout_rate']
    )
    print("Model instantiated successfully.")

except KeyError as e:
    print(f"Error: Missing key {e} in loaded model configuration.")
    exit()

except Exception as e:
    print(f"Error instantiating model from config: {e}")
    exit()

# Load state dictionary 
try:
    state_dict = torch.load(FINAL_MODEL_STATE_DICT_FILE, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model state_dict from {FINAL_MODEL_STATE_DICT_FILE}")

except FileNotFoundError:
    print(f"Error: Model state dict file not found at {FINAL_MODEL_STATE_DICT_FILE}")
    exit()

except Exception as e:
    print(f"Error loading model state_dict: {e}")
    exit()

# prepare model for inference
try:
    m = model.to(device)
    m.eval()
    print("Model moved to device and set to eval mode.")

except Exception as e:
    print(f"Error moving model to device or setting eval mode: {e}")
    exit()

######################## Chat bot ###########################################

print("\n--- GPT Chatbot Ready ---")
print("Enter your prompt, or type 'quit' or 'exit' to end.")

while True:
    prompt = input("\nPrompt: ")
    if prompt.lower() in ['quit', 'exit']:
        break 
    if not prompt: # empty input
        continue

    try:
        # encode the prompt 
        cxt_tokens = encode(prompt)
        context = torch.tensor(cxt_tokens, dtype=torch.long, device=device)

        # reshape cxt for the model: (T) -> (1, T)
        cxt_batch = context.unsqueeze(0)

        # generate completion 
        print("Generating...")
        with torch.no_grad(): # Ensure no gradients are calculated
            generated_indices = m.generate(cxt_batch, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
            # decode the generated indices 
            generated_text = decode(generated_indices)

            print(f"\nGenerated Sequence:\n{generated_text}")

    except Exception as e:
        print(f"\nAn error occurred during generation: {e}")
        # Continue the loop

print("\nChatbot session ended.")
