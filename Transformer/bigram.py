import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

with open('dataset/pg22566.txt', "r", encoding='utf-8') as f:
    text = f.read()

# print(text[:200])

# list of all characters used in the text
chars = sorted(set(text))
# print(chars)

# Create a tokenizer which consists of a encoder and decoder
# an encoder will encode each element of the char array to an int starting from 0 for the 
# first element. 

vocabulary_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# encode the story in a tensor
data = torch.tensor(encode(text), dtype=torch.long) # torch.long ensure long sequence of int
# print(data[:100])

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
block_size = 8 
batch_size = 4

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('when input is', context, 'target is', target)
