import torch
import torch.nn as nn
import numpy as np
import string
import json
import os
import tqdm


def get_vocab():
    # Construct the character 'vocabulary'
    # we allow lowercase, uppercase and digits only, along with special characters:
    # "" - Empty string used to denote elements for the RNN to ignore
    # "<bos>" - Beginning of sequence token for the input the the RNN
    # "." - End of sequence token
    vocab = ["", "<bos>", "."] + list(string.ascii_lowercase + string.ascii_uppercase + string.digits + " ")
    id_to_char = {i: v for i, v in enumerate(vocab)} # maps from ids to characters
    char_to_id = {v: i for i, v in enumerate(vocab)} # maps from characters to ids
    return vocab, id_to_char, char_to_id


def load_data(filename):
    # read in the list of names
    data = json.load(open(filename, "r"))
    # append the end of sequence token to each name
    data = [v+'.' for v in data]
    return data


def seqs_to_ids(seqs, char_to_id, max_len=20):
    """Takes a list of names and turns them into a list of tokens ids.
    Responsible for padding sequences shorter than max_len with 0 so that all sequences are max_len.
    Also truncates names that are longer than max_len.
    Should also skip empty sequences if there are any.

    Args:
        seqs (list(str)): A list of names as strings.
        char_to_id (dict(str : int)): The mapping for characters to token ids
        max_len (int, optional): The maximum length of the ouput sequence. Defaults to 20.

    Returns:
        np.array: the names represented using token ids as 2d numpy array, 
            where each row corresponds to a name. The size of the array should be N * max_len
            where N is the number of non-empty sequences input. Padded with zeros if needed.
    """
    all_seqs = []
    
    for name in seqs:
        # handle empty names
        if not name:
            continue
        id_list = [char_to_id[char] for char in name]
        if len(id_list) < max_len: id_list.extend([0]*(max_len - len(id_list)))
        elif len(id_list) > max_len: id_list = id_list[:max_len]
        all_seqs.append(id_list)
    
    return np.array(all_seqs)


class RNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size = 32, gru_size=32):
        super(RNNLM, self).__init__()

        # store layer sizes
        self.emb_size = emb_size
        self.gru_size = gru_size

        # for embedding characters (ignores those with value 0: the padded values)
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # GRU layer
        self.gru = nn.GRU(emb_size, gru_size, batch_first=True)
        # linear layer for output
        self.linear = nn.Linear(gru_size, vocab_size)
    
    def forward(self, x, h_last=None):
        """Takes a batch of names/sequences expressed as token ids and passes them through the GRU.
            The output is the predicted (un-normalized) probabilities of 
            the next token for all prefixes of the input sequences.

        Args:
            x (torch.tensor): A 2d tensor of longs giving the token ids for each batch. Shape B * S
                where B is the batch size (any batch size >= 1 is permitted), S is the length of the sequence.
            h_last (torch.tensor, optional): A 2d float tensor of size B * G where B is the batch size and G
                is the dimensionality of the GRU hidden state. The hidden state from the previous step, provide only if 
                generating sequences iteratively. Defaults to None.

        Returns:
            tuple(torch.tensor, torch.tensor): first element of the tuple is the B * S * V where V is the vocabulary size.
                This is the logit output of the RNNLM. The second element is the hidden state of the final step
                of the GRU it should be B * G dimensional.
        """

        # TODO: implement this function which does the forward pass of the RNNLM net 
        
        # Get the embeddings
        embedded_x = self.emb(x)

        # Pass embeddings through GRU 
        out, h = self.gru(embedded_x, h_last)

        # Pass GRU output through linear layer to get logits 
        logits = self.linear(out)

        return logits, h


def train_model(model, Xtrain, Ytrain, Xval, Yval, id_to_char, max_epoch):
    """Train the RNNLM model using the Xtrain and Ytrain examples.
    Uses batch stochastic gradient descent with the Adam optimizer on 
    the mean cross entropy loss. Prints out the validation loss
    after each epoch using calc_val_loss.

    Args:
        model (RNNLM): the RNNLM model.
        Xtrain (torch.tensor): The training data input sequence of size B * S. 
            B is the batch size, S is the sequence length. The sequences always start with the <bos> token id.
            The rest of the sequence is just Ytrain shifted to the right one position.
            The sequence is zero padded.
        Ytrain (torch.tensor): The expected output sequence of size B * S. Does not start with the <bos> token.
        Xval (torch.tensor): The validation data input sequence of size B * S. 
            B is the batch size, S is the sequence length. The sequences always start with the <bos> token id.
            The rest of sequence is just Yval shifted to the right one position.
            The sequence is zero padded.
        Yval (torch.tensor): The expected output sequence for the validation data of size B * S. 
            Does not start with the <bos> token. Is zero padded.
        id_to_char (dict(int : str)): A mapping from ids to tokens.
        max_epoch (int): the maximum number of epochs to train for.
    """
    # construct the adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # construct the cross-entropy loss function
    # we want to ignore padding cells with value == 0
    lossfn = nn.CrossEntropyLoss(ignore_index=0)

    # calculate number of batches
    batch_size = 32
    num_batches = int(Xtrain.shape[0] / batch_size)

    # run the main training loop over many epochs
    for epoch in range(max_epoch):
        # create a batch loop to go through each batch 
        for n in range(num_batches):
            start_idx = n * batch_size 
            end_idx = (n+1) * batch_size
            # get the slice of data for curr batch (mini-batch)
            X_batch = Xtrain[start_idx:end_idx]
            Y_batch = Ytrain[start_idx:end_idx]

            # zero the gradients
            optim.zero_grad()

            # forward pass 
            outputs, _ = model(X_batch)

            # calculate loss 
            loss = lossfn(outputs.permute(0, 2, 1), Y_batch)

            # backward pass 
            loss.backward()

            # update weights
            optim.step()
        
        validation = calc_val_loss(model, Xval, Yval)
        print(f"Validation for epoch: {epoch} is: {validation}")


def gen_string(model, id_to_char, max_len=20, sample=True):
    """Generate a name using the model. The generation process should finish once
    the end token is seen. We either sample from the model, where the next token is
    chosen randomly according to the categorical probability distribution produced by softmax,
    or we use argmax decoding where the most likely token is chosen at every generation step.

    Args:
        model (RNNLM): The trained RNNLM model.
        id_to_char (dict(int, str)): A mapping from token ids to token strings.
        max_len (int, optional): The maximum length of the output senquence. Defaults to 20.
        sample (bool, optional): If True then generate samples. If False then use argmax decoding. 
            Defaults to True.

    Returns:
        str: The generated name as a string.
    """
    # put the model into eval mode because we don't need gradients
    model.eval()

    # setup the initial input to the network
    # we will use a batch size of one for generation
    h = torch.zeros((1,1,model.gru_size), dtype=torch.float) # h0 is all zeros
    x = torch.ones((1, 1), dtype=torch.long) # x is the <bos> token id which = 1
    out_str = ""

    # generate the sequence step by step
    for i in range(max_len):
        # get scores for next char and update hidden state for nxt iter 
        logits, h = model(x, h) 
        # sampling options 
        if not sample: next_char_id = torch.argmax(logits, dim=-1).item()
        else: 
            probability_dist = torch.softmax(logits, dim=-1)
            # prob_dist (3D -> 2D) using squeeze
            next_char_id = torch.multinomial(probability_dist.squeeze(), 
                                             num_samples=1).item()
        # check if the char is '.', break if it is 
        if next_char_id == 2: break

        # if not end token, convert char id back to string and add to out 
        else:
            next_char = id_to_char[next_char_id]
            out_str = out_str + next_char 

        # repeat
        x = torch.tensor([[next_char_id]], dtype=torch.long)

    # set the model back to training mode in case we need gradients later
    model.train()

    return out_str


def calc_val_loss(model, Xval, Yval):
    """Calculates the validation loss in average nats per character.

    Args:
        model (RNNLM): the RNNLM model.
        Xval (torch.tensor): The validation data input sequence of size B * S. 
            B is the batch size, S is the sequence length. The sequences always start with the <bos> token id.
            The rest of sequence is just Yval shifted to the right one position.
            The sequence is zero padded.
        Yval (torch.tensor): The expected output sequence for the validation data of size B * S. 
            Does not start with the <bos> token. Is zero padded.

    Returns:
        float: validation loss in average nats per character.
    """

    # use cross entropy loss
    lossfn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # put the model into eval mode because we don't need gradients
    model.eval()

    # calculate number of batches, we need to be precise this time
    batch_size = 32
    num_batches = int(Xval.shape[0] / batch_size)
    if Xval.shape[0] % batch_size != 0:
        num_batches += 1

    # sum up the total loss
    total_loss = 0
    total_chars = 0
    for n in range(num_batches):

        # calculate batch start end idxs 
        s = n * batch_size
        e = (n+1)*batch_size
        if e > Xval.shape[0]:
            e = Xval.shape[0]

        # compute output of model        
        out,_ = model(Xval[s:e])

        # compute loss and store
        loss = lossfn(out.permute(0, 2, 1), Yval[s:e]).detach().cpu().numpy()
        total_loss += loss

        char_count = torch.count_nonzero(Yval[s:e].flatten())
        total_chars += char_count.detach().cpu().numpy()

    # compute average loss per character
    total_loss /= total_chars
    
    # set the model back to training mode in case we need gradients later
    model.train()

    return total_loss


def main():

    # load the data from disk
    data = load_data(os.path.join("data", "names_small.json"))

    # get the letter 'vocabulary'
    vocab, id_to_char, char_to_id = get_vocab()
    vocab_size = len(vocab)

    # convert the data into a sequence of ids which will be the target for our RNN
    Y = seqs_to_ids(data, char_to_id)
    # the input needs to be shifted by 1 and have the <bos> tokenid prepended to it
    # this also means we have to remove the last element of the sequence to keep the length constant
    X = np.concatenate([np.ones((Y.shape[0], 1)), Y[:, :-1]], axis=1)

    # split the data int training and validation
    # convert the data into torch tensors
    train_frac = 0.9
    num_train = int(X.shape[0]*train_frac)
    Xtrain = torch.tensor(X[:num_train], dtype=torch.long)
    Ytrain = torch.tensor(Y[:num_train], dtype=torch.long)
    Xval = torch.tensor(X[num_train:], dtype=torch.long)
    Yval = torch.tensor(Y[num_train:], dtype=torch.long)

    # train the model
    model = RNNLM(vocab_size)
    train_model(model, Xtrain, Ytrain, Xval, Yval, id_to_char, max_epoch=10)

    # use the model to generate and print some names
    print("Argmax: ", gen_string(model, id_to_char, sample=False))
    print("Random:")
    for i in range(10):
        gstr = gen_string(model, id_to_char)
        print(gstr)

if __name__ == "__main__":
    vocab, id_to_char, char_to_id = get_vocab()
    main()

