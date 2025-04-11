########################### GPT Transformer Network #############################

import torch
import torch.nn as nn 
from torch.nn import functional as F

from datasets import load_dataset
print("Attempting to load OpenWebText ...")
# Load the dataset (will download it the first time and cache it)
try:
    # dataset usually only has a 'train' split 
    owt_dataset = load_dataset("openwebtext", split='train')
    print("Dataset loaded successfully")

    # print dataset info
    print(owt_dataset)
    print("\nFirst example:")
    print(owt_dataset[0])
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    print("Please ensure you have enough disk space and a stable internet connection.")
    print("You might also need to log in to Hugging Face Hub if access restrictions apply (though usually not for OpenWebText). Use: `huggingface-cli login`")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size, batch_size, max_iters, lr, eval_iters = 64, 128, 10000, 3e-4, 250
n_embed, n_head, dropout, n_layer = 384, 8, 0.2, 8

############################## Data Setup #########################################
chars = ""
with open('dataset/pg22566.txt', "r", encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

# Create a tokenizer which consists of a encoder and decoder
# an encoder will encode each element of the char array to an int starting from 0 for the 
# first element. 

vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# encode the story in a tensor
data = torch.tensor(encode(text), dtype=torch.long) # torch.long ensure long sequence of int
#print(data[:100])

####################################################################################

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # rand indices for the dataset 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch('train')

################################################################################

class Head(nn.Module):
    """one head of self_attention"""
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(), 
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

################################################################################

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        # four heads running in parallel 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.project = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenate each head together along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (channel dim)
        out = self.dropout(self.project(out))
        return out 

################################################################################

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension; n_head: the number of heads we'd like
        super().__init__()
        # number of features each of the heads are capturing
        head_size = n_embed // n_head 
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        # helps smooth out features
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.self_attention(x)
        x = self.layer_norm1(x + y)
        y = self.feed_forward(x)
        x = self.layer_norm2(x + y)
        return x

################################################################################

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # pred lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # positional encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # create four decoder layers Sequentially
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

        self.final_layer_norm = nn.LayerNorm(n_embed) # help converge
        self.lang_model_head = nn.Linear(n_embed, vocab_size)
        # apply weights 
        self.apply(self.init_weights)

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

        # idx and targets are both (b, t) tensor of integers
        tok_embedding = self.token_embedding_table(idx) # (b, t, c)
        pos_embedding = self.position_embedding_table(torch.arange(time, device=device)) # (t, c) 
        x = tok_embedding + pos_embedding # (batch, time, channel)
        x = self.blocks(x) # (batch, time, channel)
        x = self.final_layer_norm(x) # (batch, time, channel)
        logits = self.lang_model_head(x) # (batch, time, vocab_size)

        if targets is None:
            loss = None 
        else:
            batch, time, channel = logits.shape
            # reshape to (N, C) for cross_entropy
            logits = logits.view(batch * time, channel)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss 
    
    # generate tokens
    def generate(self, idx, max_new_tokens):
        # index is (batch, time) array of indices in the current context 
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens 
            idx_cond = idx[:, -block_size:]
            # get predictions using the cropped content 
            logits, loss = self.forward(idx_cond)
            # focusing only on the last time step (single prev char)
            logits = logits[:, -1, :] # becomes (batch, channel)
            # apply softmax to get probs dist
            probs = F.softmax(logits, dim=-1) # (batch, channel)
            # sample from the distribution
            idx_nxt = torch.multinomial(probs, num_samples=1) # (batch, 1)
            # append sample idx to the running sequence
            idx = torch.cat((idx, idx_nxt), dim=1) # (B, T+1)
        return idx

# -----------------------Initialize model------------------------------------
model = GPTLanguageModel(vocab_size)
m = model.to(device)
# -----------------------Optimizer-------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad() # makes sure torch doesn't use gradient for this fn.
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    # every 250 iterations, print the loss val
    if iter % eval_iters == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
    # sample a batch of data 
    x_batch, y_batch = get_batch('train')
    # eval the loss 
    logits, loss = m.forward(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"The loss is: {loss.item()}")

#############################################################################



#############################################################################

cxt = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(cxt, max_new_tokens=500)[0].tolist())
print("Generated characters:\n")
print(generated_chars)


