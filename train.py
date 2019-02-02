from pathlib import Path
import requests
import zipfile
from tqdm import trange, tqdm
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Initialize W&B project
wandb.init(project="char-RNN")

# Define hyper-parameters
config = wandb.config           # for shortening
config.rnn_module = "GRU"       # "RNN", "LSTM" or "GRU"
config.hidden_size = 256        # hidden size of RNN module
config.num_layers = 2           # number of layers of RNN module
config.dropout = 0.1            # dropout between RNN layers (0 means no dropout)
config.epochs = 100             # number of epochs for training
config.batches_per_epoch = 300  # number of batches of data processed per epoch
config.sequence_per_batch = 8   # number of sequence of characters per batch
config.char_per_sequence = 150  # number of characters per sequence

# Download dataset
PATH_DATA = Path("data")
FILENAME_DATA = Path("monte_cristo.txt")
URL_DATA = "https://www.gutenberg.org/files/1184/1184-0.txt"
PATH_DATA.mkdir(exist_ok = True)
PATH_DATAFILE = PATH_DATA / FILENAME_DATA
if not (PATH_DATAFILE).exists():
    r = requests.get(URL_DATA)
    PATH_DATAFILE.open("wb").write(r.content)

# Read the text and filter out content, bibliography…
with open(PATH_DATAFILE, 'r', encoding="utf8") as f:
    lines = f.readlines()
    # Remove start and end of file (not interesting data)
    lines = lines[319:60662]
    chars = ''.join(lines)

# Map between chars and int
# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/data.py
class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)

# Convert data from char to tokens
data_dictionary = Dictionary()
tensor_data = torch.LongTensor(len(chars))

for i, c in enumerate(chars):
    tensor_data[i] = data_dictionary.add_char(c)
    
n_elements = len(data_dictionary)

# Split the data between test and validation sets
# We actually don't go through the entire test set at each epoch while we do for validation set
split = round(0.98 * len(tensor_data))      # to be adjusted based on file size (2% validation of 2.6MB is enough)
train_data, train_label = tensor_data[:split], tensor_data[1:split+1]
valid_data, valid_label = tensor_data[split:-2], tensor_data[split+1:]

# Create a class to handle data in batch
class TrainingData():    
    def __init__(self, train_data, train_label, device, sequence_per_batch = 64, char_per_sequence = 128):
        
        self.train_data = train_data
        self.train_label = train_label
        self.sequence_per_batch = sequence_per_batch
        self.char_per_sequence = char_per_sequence
        self.device = device
        self.length = len(train_data)
        
        # We start reading the text at even sections based on number of sequence per batch
        self.batch_idx = range(0, self.length, self.length // sequence_per_batch)
        self.batch_idx = self.batch_idx[:sequence_per_batch]
        assert len(self.batch_idx) == sequence_per_batch, '{} batches expected vs {} actual'.format(sequence_per_batch,
                                                                                                    len(self.batch_idx))
        
    def next_batch(self):
        
        # loop to the start if we reached the end of text
        self.batch_idx = list(idx if idx + self.char_per_sequence < self.length else 0 for idx in self.batch_idx)
        
        # Extract sequences
        sequences_input = tuple(self.train_data[idx:idx+self.char_per_sequence] for idx in self.batch_idx)
        sequences_label = tuple(self.train_label[idx:idx+self.char_per_sequence] for idx in self.batch_idx)

        # Transform input into one-hot (source: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/29)
        sequences_input = tuple(torch.zeros(len(data), n_elements, device = self.device).scatter_(1, data.unsqueeze(-1), 1) for data in sequences_input)
        
        # Move next idx
        self.batch_idx = (idx + self.char_per_sequence for idx in self.batch_idx)
        
        # Concatenate tensors
        return torch.stack(sequences_input, dim=1), torch.stack(sequences_label, dim=1)

# Define NN
class Model(nn.Module):
    def __init__(self, input_size, batch_size, rnn_module = "RNN", hidden_size = 64, num_layers = 1, dropout = 0):
        super(Model, self).__init__()
        self.input_size = input_size
        self.rnn_module = rnn_module
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if rnn_module == "RNN":
            self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)
        elif rnn_module == "LSTM":
            self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)
        elif rnn_module == "GRU":
            self.rnn = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout)
            
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        output = input.view(1, -1, self.input_size)
        output, hidden = self.rnn(output, hidden)
        output = self.output(output[0])
        return output, hidden

    def initHidden(self, batch_size):
        # initialize hidden state to zeros
        if self.rnn_module == "LSTM":
            return torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(
                self.num_layers, batch_size, self.hidden_size)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Define loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer_function = optim.Adam

# Build the NN
model = Model(len(data_dictionary), config.sequence_per_batch, config.rnn_module, config.hidden_size, config.num_layers, config.dropout)
hidden = model.initHidden(config.sequence_per_batch)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = train_data.to(device)
train_label = train_label.to(device)
valid_data = valid_data.to(device)
valid_label = valid_label.to(device)
model.to(device)
if config.rnn_module == "LSTM":
    for h in hidden:
        h = h.to(device)
else:
    hidden = hidden.to(device)

# Store network topology
wandb.watch(model)

# Define optimizer
optimizer = optimizer_function(model.parameters())

# Load data
training_data = TrainingData(train_data, train_label, device, config.sequence_per_batch, config.char_per_sequence)
valid_length = len(valid_data)

# Start training
for epoch in trange(config.epochs):
    train_loss = 0   # training loss
    valid_loss = 0   # validation loss
    
    # Training of one epoch
    model.train()
    for i in trange(config.batches_per_epoch):
        
        # Get a batch of sequences
        input_vals, label_vals = training_data.next_batch()

        # Detach hidden layer and reset gradients
        if config.rnn_module == "LSTM":
            tuple(h.detach_() for h in hidden)
        else:
            hidden.detach_()
        optimizer.zero_grad()
        
        # Forward pass and calculate loss
        loss_sequence = torch.zeros(1, device=device)
        for (input_val, label_val) in zip(input_vals, label_vals):
            output, hidden = model(input_val, hidden)
            loss = loss_function(output, label_val.view(-1))
            loss_sequence += loss
            
        # Backward propagation and weight update
        loss_sequence.backward()
        optimizer.step()
        
        train_loss += loss_sequence.item() / config.batches_per_epoch / config.char_per_sequence
        
    # Calculate validation loss
    with torch.no_grad():
        model.eval()

        # Detach hidden layers
        hidden_valid = model.initHidden(1)
        if config.rnn_module == "LSTM":
            for h in hidden_valid:
                h = h.to(device)
        else:
            hidden_valid = hidden_valid.to(device)
            
        # Process validation data one character at a time
        for i in range(valid_length-1):
            input_val = valid_data[i].view(1)
            label_val = valid_label[i]

            # One-hot input
            input_val = torch.zeros(len(input_val), n_elements, device = device).scatter_(1, input_val.unsqueeze(-1), 1)

            # Forward pass and calculate loss
            output, hidden_valid = model(input_val, hidden_valid)
            loss = loss_function(output, label_val.view(-1))
            valid_loss += loss.item() / (valid_length - 1)

    # Log results
    wandb.log({"Training loss": train_loss,
               "Validation loss": valid_loss})
        
    tqdm.write("\nEpoch {} - Training loss {} - Validation loss {}\n".format(epoch+1, train_loss, valid_loss))
    
# Save model to W&B
torch.save(model, Path(wandb.run.dir) / 'model.pt')