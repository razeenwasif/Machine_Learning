########################### GPT Transformer Network ############################

import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.nn import functional as F
import random
import mmap
import time 
import functools
import os
import pickle
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

#parser = argparse.ArgumentParser(description="GPT Transformer Network")
#parser.add_argument("-batch_size", type=str, required=True, help="provide a batch size")
#args = parser.parse_args()
#print(f"batch size: {args.batch_size}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Define HYPERPARAMETER RANGES/CHOICES for Optuna ---
DEFAULT_BLOCK_SIZE = 128
DEFAULT_BATCH_SIZE = 64 # args.batch_size # batch sizes can be 64, 128, 256, ...
DEFAULT_VOCAB_SIZE = None # Will be set after loading vocab
DEFAULT_MAX_ITERS = 15000 # for the final training run 
DEFAULT_LR = 3e-4 # default it not tuning 

# --- File paths ---
TRAIN_FILE = os.path.join(".", "dataset_code_contests", "output_train.txt")
VAL_FILE = os.path.join(".", "dataset_code_contests", "output_val.txt")
VOCAB_FILE = os.path.join(".", "dataset_code_contests", "vocab.txt")
STUDY_DB_FILE = "sqlite:///gpt_tuning.db"
BEST_PARAMS_FILE = "best_params.pkl"
FINAL_MODEL_STATE_DICT_FILE = "final_model_state.pth"

# --- Optuna Tuning  settings ---
ENABLE_OPTUNA = True
N_TRIALS = 50 # Number of Optuna trials to run
TUNING_MAX_ITERS = 1000 # Reduced iterations PER TRIAL for faster tuning
TUNING_EVAL_INTERVAL = 250 # How often to evaluate within a tuning trial
TUNING_EVAL_ITERS = 100 # Batches to average for eval loss during Tuning

############################## Data Setup ########################################
chars = ""
try:
    with open(VOCAB_FILE, "r", encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
except FileNotFoundError:
    print(f"Error: Vocab file not found at {VOCAB_FILE}")
    exit()

DEFAULT_VOCAB_SIZE = len(chars)
print(f"Vocabulary size: {DEFAULT_VOCAB_SIZE}")

# Create a tokenizer which consists of a encoder and decoder
# an encoder will encode each element of the char array to an int starting from 0 for the first element. 
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_int.get(c, 0) for c in s]
decode = lambda l: ''.join([int_to_string.get(i, '?') for i in l])

############################## Timer #############################################

def timer_decorator(func):
    """Decorator that prints the execution time of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(f"\n--- Starting execution of '{func.__name__}' ---")
        start_time = time.perf_counter()
        cpu_start_time = time.process_time()

        value = func(*args, **kwargs)

        cpu_end_time = time.process_time()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time 
        cpu_elapsed_time = cpu_end_time - cpu_start_time 

        print(f"\nFunction '{func.__name__}' used {cpu_elapsed_time:.4f} seconds of CPU time.")
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds (wall time).")
        print(f"--- Finished execution of '{func.__name__}' ---") 

        return value
    return wrapper_timer

############################# Data Loading Functions #############################

# memory mapping allows accessing memory addressses (kind of like but not really only reading a part of a file without opening the entire thing) and treating a file as if it were memory. The OS lazily loads only the necessary parts (pages) of the file from disk into physical RAM on demand when you actually access them.

# memory map for using small snippets of text from a single file 
def get_rand_chunk(split, block_size, batch_size):

    filename = TRAIN_FILE if split == "train" else VAL_FILE
    
    try:
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Determing file size and a rnd pos to start reading 
                file_size = len(mm)
                # Need at least block_size + 1 tokens for one sequence (x and y)
                min_required_size_per_item = block_size + 1 
                # aim to read enough for maybe 2*batch_size items reliably
                read_size = min_required_size_per_item * batch_size * 2 
                
                if file_size < min_required_size_per_item:
                     print(f"Warning: File '{filename}' ({file_size} bytes) too small for block_size {block_size}.")
                     return torch.empty(0, dtype=torch.long)

                if read_size > file_size:
                    read_size = file_size

                max_start_pos = file_size - read_size
                if max_start_pos < 0: max_start_pos = 0

                # random: (start, end) -> end threshold = cutoff near the end 
                # to prevent out of bounds error
                start_pos = random.randint(0, max_start_pos)
                # seek the random pos and read the block of text 
                mm.seek(start_pos)
                # read a snippet of text from the start_pos
                block = mm.read(read_size)

                # Decode the bin block to a string, ignoring any invalid byte sequences 
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
                # train and test splits
                # encode it into tokenized forms
                data = torch.tensor(encode(decoded_block), dtype=torch.long)
                return data

    except FileNotFoundError:
        print(f"Error: Data file not found at {filename}")
        return torch.empty(0, dtype=torch.long) # Return empty tensor on errors

    except ValueError as e:
        return torch.empty(0, dtype=torch.long)

def get_batch(split, block_size, batch_size):
    data = get_rand_chunk(split, block_size, batch_size)

    if len(data) < block_size + 1:
        # Not enough data in the loaded chunk even for one sequence
        return None, None

    max_idx = len(data) - block_size - 1 # Max valid start index for sequence i
    if max_idx < 0:
       return None, None # Should be caught by len check above, but safe

    # Ensure we don't request more batches than possible start indices
    actual_batch_size = min(batch_size, max_idx + 1)
    if actual_batch_size <= 0: return None, None # Should not happen if max_idx >= 0

    # rand indices for the dataset 
    ix = torch.randint(0, max_idx + 1, (actual_batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(), 
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x):
        return self.net(x)

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

        # Use self.config['block_size'] or ensure 'time' doesn't exceed it
        if time > self.config['block_size']:
           idx = idx[:, -self.config['block_size']:]
           time = self.config['block_size']

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
        self.eval()
        with torch.no_grad():
            # index is (batch, time) array of indices in the current context 
            for _ in range(max_new_tokens):
                # crop index to the last block_size tokens 
                idx_cond = idx[:, -self.config['block_size']:]
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

###################### Loss estimation ##########################################

@torch.no_grad() # makes sure torch doesn't use gradient for this fn.
def estimate_loss(model, block_size, batch_size, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.full((eval_iters,), float('nan'), device=device)
        valid_count = 0 
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size)

            if X is None or Y is None: continue

            logits, loss = model(X, Y)

            if loss is not None:
                losses[k] = loss
                valid_count += 1

        if valid_count > 0:
            #  Calculate mean only on valid (non-NaN) losses
            valid_losses = losses[~torch.isnan(losses)]
            out[split] = valid_losses.mean().item() # scalar value 
        else:
            out[split] = float('inf') # high val if all batches failed
    model.train()
    return out

# ----------------- Main training loop function -----------------------------
@timer_decorator 
def run_training_loop(model, optimizer, max_iters, eval_interval, eval_iters,
                      block_size, batch_size, gradient_accumulation_steps=1,
                      trial=None, scheduler=None):
    """
    Runs the main training loop.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        max_iters: Total number of iterations to train for.
        eval_interval: How often (in iterations) to evaluate the model.
        eval_iters: Number of batches to use for estimating loss.
        block_size: The context length for sequences.
        batch_size: The number of sequences per batch.
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        trial: An Optuna trial object. If provided, interacts with Optuna.
        scheduler: A PyTorch learning rate scheduler object.
    """
    m = model 
    m.train()
    accumulated_loss = 0.0 
    steps_since_eval = 0 
    final_loss_value = None 

    print(f"Starting training loop for {max_iters} iterations...")
    for iter_num in range(max_iters):
        loss = None # reset for this step 
        # --- training step ----
        for micro_step in range(gradient_accumulation_steps):
            x_batch, y_batch = get_batch('train', block_size, batch_size)
            if x_batch is None or y_batch is None:
                 print(f"Iter {iter_num}, MicroStep {micro_step}: Skipping due to batch error.")
                 continue # Skip this step

            try:
                with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                    logits, step_loss = m.forward(x_batch, y_batch)

                if step_loss is not None:
                    # scale loss for gradient accumulation 
                    step_loss = step_loss / gradient_accumulation_steps
                    step_loss.backward()
                    accumulated_loss += step_loss.item() * gradient_accumulation_steps
                    loss = accumulated_loss 

            except RuntimeError as e:
                print(f"Iter {iter_num}, MicroStep {micro_step}: RuntimeError during forward/backward: {e}")
                if "out of memory" in str(e) and trial is not None:
                      print("OOM detected, pruning trial.")
                      raise optuna.exceptions.TrialPruned() # Prune on OOM
                else:
                      # Handle other runtime errors - maybe skip step or raise?
                      # For now, let's try to continue if not OOM
                      pass
            except Exception as e: # Catch other errors
                 print(f"Iter {iter_num}, MicroStep {micro_step}: Unexpected error: {e}")
                 # Maybe prune or break? For now, continue.
                 pass
        
        # --- Optimizer step --- 
        if iter_num % gradient_accumulation_steps == gradient_accumulation_steps -1:
            try:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # --- learning rate scheduler step --- 
                if scheduler is not None:
                    scheduler.step()
                
                # average loss over accumulation steps 
                final_loss_value = accumulated_loss / gradient_accumulation_steps
                accumulated_loss = 0.0 # Reset accumulated loss
            
            except Exception as e:
                  print(f"Iter {iter_num}: Error during optimizer step: {e}")

        # --- Evaluation & optuna reporting / pruning ---
        is_last_iter = (iter_num == max_iters - 1)
        if (iter_num > 0 and iter_num % eval_interval == 0) or is_last_iter:
            # get current learning rate 
            current_lr = optimizer.param_groups[0]['lr'] # lr from 1st param group
            estimated_losses = estimate_loss(m, block_size, batch_size, eval_iters)
            val_loss = estimated_losses.get('val', float('inf'))
            train_loss = estimated_losses.get('train', float('inf')) # Get train loss too

            print(f"  Iter {iter_num}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Optuna Interaction (if trial object is provided)
            if trial is not None:
                # Ensure val_loss is finite before reporting
                if not isinstance(val_loss, (float, int)) or not torch.isfinite(torch.tensor(val_loss)):
                    print(f"  Trial {trial.number}, Iter {iter_num}: Invalid validation loss ({val_loss}). Pruning.")
                    raise optuna.exceptions.TrialPruned()

                trial.report(val_loss, iter_num) # Report metric to Optuna

                # Check if the trial should be pruned
                if trial.should_prune():
                    print(f"  Trial {trial.number}, Iter {iter_num}: Pruning trial based on intermediate value.")
                    raise optuna.exceptions.TrialPruned()

        # Check for NaN loss, might indicate instability
        if loss is not None and torch.isnan(torch.tensor(loss)):
             print(f"Iter {iter_num}: Loss is NaN. Stopping training.")
             if trial is not None: raise optuna.exceptions.TrialPruned()
             break # Stop regular training


    print(f"Finished training loop. Last computed loss: {final_loss_value}")
    return final_loss_value # Return last computed average loss

######################## Optuna Objective Function #############################

def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function""" 

    # categorical for discrete choices often works well 
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # make n_embed divisible by n_head. Suggest n_head first.
    n_head = trial.suggest_categorical("n_head", [4, 8])
    n_embed_options = [128, 256, 384, 512]
    valid_n_embed_options = [n for n in n_embed_options if n % n_head == 0]

    if not valid_n_embed_options:
        # shouldn't happen but still
        print(f"Pruning trial: n_embed {n_embed} not divisible by n_head {n_head}")
        raise optuna.exceptions.TrialPruned()

    n_embed = trial.suggest_categorical("n_embed", valid_n_embed_options)
    n_layer = trial.suggest_int("n_layer", 2, 8)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.3)
    block_size = trial.suggest_categorical("block_size", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: lr={lr:.6f}, n_embed={n_embed}, n_head={n_head}, n_layer={n_layer}, dropout={dropout_rate:.3f}")

    # --- build the model and optimizer 

    try:
        model = GPTLanguageModel(
            vocab_size=DEFAULT_VOCAB_SIZE, block_size=block_size,
            n_embed=n_embed, n_head=n_head, n_layer=n_layer, 
            dropout_rate=dropout_rate
        )
        m = model.to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

    except Exception as e:
        print(f"Error during model/optimizer creation: {e}")
        return float('inf') # Return high loss if setup fails

    # --- training loop ---
    try:
        final_val_loss = run_training_loop(
            model=m, optimizer=optimizer, max_iters=TUNING_MAX_ITERS,
            eval_interval=TUNING_EVAL_INTERVAL, eval_iters=TUNING_EVAL_ITERS,
            block_size=block_size, batch_size=batch_size,
            trial=trial 
        )
    except optuna.exceptions.TrialPruned as e:
         print(f"Trial {trial.number} pruned.")
         raise e 
    except Exception as e: 
         print(f"Trial {trial.number}: Unexpected error during training: {e}")
         return float('inf') # High loss on unexpected error

    print(f"--- Finished Trial {trial.number} ---")
    # Return final validation loss metric (already checked for finite inside loop)
    if final_val_loss is None or not torch.isfinite(torch.tensor(final_val_loss)):
       return float('inf') # Return high value if training failed to produce valid loss
    # Optuna minimizes the return value
    return final_val_loss
    
################################# Run optuna study ###############################

best_params = None
if ENABLE_OPTUNA:
    print("\n--- Starting Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=TUNING_EVAL_INTERVAL * 2, n_min_trials=5), # Prune after a few evals/trials
        study_name='gpt_transformer_tuning',
        storage=STUDY_DB_FILE,
        load_if_exists=True
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None) # Add gc_after_trial=True if memory is tight
    except KeyboardInterrupt:
        print("Optimization stopped manually.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")

    print("\n--- Optuna Study Summary ---")
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        # Check if any trials completed successfully
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            best_trial = study.best_trial
            print(f"Best trial value (min validation loss): {best_trial.value:.4f}")
            print("Best hyperparameters:")
            best_params = best_trial.params
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            # Save best params
            try:
                with open(BEST_PARAMS_FILE, "wb") as f:
                    pickle.dump(best_params, f)
                print(f"Best parameters saved to {BEST_PARAMS_FILE}")
            except Exception as e:
                 print(f"Error saving best parameters: {e}")
        else:
            print("No trials completed successfully. Cannot determine best parameters.")
            best_params = None # Ensure it's None

    except ValueError:
         print("No trials recorded in the study (or study failed). Cannot determine best parameters.")
         best_params = None

    print("\n--- Optimization Finished ---")
else:
    print("\n--- Optuna Tuning Skipped ---")
    # Try loading previously saved best parameters if Optuna is disabled
    if os.path.exists(BEST_PARAMS_FILE):
        print(f"Loading best parameters from {BEST_PARAMS_FILE}")
        try:
            with open(BEST_PARAMS_FILE, "rb") as f:
                best_params = pickle.load(f)
            print("Loaded parameters:", best_params)
        except Exception as e:
            print(f"Error loading parameters file: {e}")
            best_params = None
    else:
        print("No best parameters file found.")
        best_params = None

################################################################################

# ----------------------- Train model with best params ---------------------------

if best_params is not None:
    print("\n--- Training Final Model with Best Parameters ---")

    # Use best params found or loaded
    final_lr = best_params['lr']
    final_n_embed = best_params['n_embed']
    final_n_head = best_params['n_head']
    final_n_layer = best_params['n_layer']
    final_dropout = best_params['dropout']
    # Use best block/batch size OR defaults? Decide here. Let's use best found.
    final_block_size = best_params.get('block_size', DEFAULT_BLOCK_SIZE)
    final_batch_size = best_params.get('batch_size', DEFAULT_BATCH_SIZE)

    print("Using parameters for final training:")
    print(f"  lr={final_lr:.6f}, n_embed={final_n_embed}, n_head={final_n_head}, n_layer={final_n_layer}, dropout={final_dropout:.3f}, block={final_block_size}, batch={final_batch_size}")

    try:
        final_model = GPTLanguageModel(
            vocab_size=DEFAULT_VOCAB_SIZE,
            block_size=final_block_size,
            n_embed=final_n_embed,
            n_head=final_n_head,
            n_layer=final_n_layer,
            dropout_rate=final_dropout
        )
        final_m = final_model.to(device)
        final_optimizer = torch.optim.AdamW(final_m.parameters(), lr=final_lr)

        # --- create the learning rate scheduler --- 
        # Cosine Annealing decays the learning rate following a cosine curve 
        # T_max: the number of iterations for half a cycle (or full cycle if eta_min=final_lr)
        # setting it to max_iters makes it decay over the whole training run 
        # eta_min: the minimum learning rate it will reach at the end of the cycle.
        scheduler = CosineAnnealingLR(
            optimizer=final_optimizer,
            T_max=DEFAULT_MAX_ITERS, # total iter for the decay cycle
            eta_min=final_lr * 0.1 # decays to 10% of the initial lr 
        )
        print(f"Using CosineAnnealingLR scheduler: T_max={DEFAULT_MAX_ITERS}, eta_min={final_lr * 0.1:.7f}")

        # --- Run Full Training ---
        # Apply timer decorator here if desired
        final_training_loss = run_training_loop(
            model=final_m,
            optimizer=final_optimizer,
            max_iters=DEFAULT_MAX_ITERS, # Use full iterations
            eval_interval=TUNING_EVAL_INTERVAL, # Can use tuning interval or a different one
            eval_iters=TUNING_EVAL_ITERS,    # Can use tuning iters or different ones
            block_size=final_block_size,
            batch_size=final_batch_size,
            gradient_accumulation_steps=1,
            trial=None, # No Optuna trial for final run
            scheduler=scheduler
        )

        print(f"\nFinal model training completed. Last loss: {final_training_loss}")

        # --- Save the Final Model State Dict---
        try:
            torch.save(final_m.state_dict(), FINAL_MODEL_STATE_DICT_FILE)
            print(f"Final model state_dict saved to {FINAL_MODEL_STATE_DICT_FILE}")
            # Also save the config used to build this model
            with open("final_model_config.pkl", "wb") as f:
                 pickle.dump(final_m.config, f)
            print("Final model configuration saved.")
        except Exception as e:
            print(f"Error saving final model state_dict: {e}")

    except Exception as e:
        print(f"Error during final model training: {e}")
        final_m = None # Ensure model is None if training failed

################################################################################

################################ Text Generation ###############################

    if final_m is not None:
            print("\n--- Generating Text with Final Trained Model ---")
            final_m.eval() # Ensure eval mode
            cxt = torch.zeros((1,1), dtype=torch.long, device=device)
            generated_output = None
            try:
                 # generate method already uses torch.no_grad() internally if called directly
                 generated_indices = final_m.generate(cxt, max_new_tokens=5000)[0].tolist()
                 generated_output = decode(generated_indices)
            except Exception as e:
                 print(f"Error during generation: {e}")

            if generated_output:
                print("Generated characters:\n")
                print(generated_output)
            else:
                print("Text generation failed.")
    else:
            print("Skipping generation as final model training failed.")

else:
    print("\n--- Skipping Final Model Training & Generation (No Best Parameters Found/Loaded) ---")

