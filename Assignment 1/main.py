import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import json

def generate_dataset(train_size, test_size, seed):
    """
    Generates a dataset of binary integers A, B, and C = A * B.
    Each integer has at most 8 digits in their binary representation.
    The dataset is generated using the given random seed.
    Returns two lists of tuples: one for the training set and one for the test set.
    Each tuple contains three binary integers: A, B, and C.
    """
    random.seed(seed)
    train_set = []
    test_set = []
    for i in range(train_size + test_size):
        a = random.randint(0, 2**8 - 1)
        b = random.randint(0, 2**8 - 1)
        c = a * b
        train_or_test = "train" if i < train_size else "test"
        if train_or_test == "train":
            train_set.append((bin(a)[2:].zfill(8), bin(b)[2:].zfill(8), bin(c)[2:].zfill(16)))
        else:
            test_set.append((bin(a)[2:].zfill(8), bin(b)[2:].zfill(8), bin(c)[2:].zfill(16)))
    return train_set, test_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains an RNN to perform multiplication of binary integers A * B = C")
    parser.add_argument("--train-size", type=int, default=10000, help="size of the generated training set")
    parser.add_argument("--test-size", type=int, default=1000, help="size of the generated test set")
    parser.add_argument("--seed", type=int, default=1234, help="random seed used for creating the datasets")
    args = parser.parse_args()

    train_set, test_set = generate_dataset(args.train_size, args.test_size, args.seed)

    print("Training set:")
    for a, b, c in train_set:
        print(f"{a} * {b} = {c}")
    print("Test set:")
    for a, b, c in test_set:
        print(f"{a} * {b} = {c}")

 
# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        output = self.i2o(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Define the training function
def train(rnn, input_tensor, target_tensor, criterion, optimizer):
    hidden = rnn.initHidden(input_tensor.size(0))
    optimizer.zero_grad()
    loss = 0

    for i in range(input_tensor.size(1)):
        output, hidden = rnn(input_tensor[:, i, :], hidden)
        loss += criterion(output.squeeze(), target_tensor[i])

    loss.backward()
    optimizer.step(momentum=momentum)

    return output, loss.item() / input_tensor.size(1)

# Generate the dataset
train_set, test_set = generate_dataset(train_size=10000, test_size=1000, seed=1234)

# Load Hyperparameters from json file
with open("param.json", "r") as f:
    hyperparams = json.load(f)

# Access the hyperparameters for the optimization algorithm
learning_rate = hyperparams["optim"]["lr"]
momentum = hyperparams["optim"]["momentum"]

# Access the hyperparameters for the model
input_size = hyperparams["model"]["input_size"]
hidden_size = hyperparams["model"]["hidden_size"]
output_size = hyperparams["model"]["output_size"]
num_layers = hyperparams["model"]["num_layers"]
dropout = hyperparams["model"]["dropout"]

'''
input_size = 18
hidden_size = 128
output_size = 17
learning_rate = 0.01
num_epochs = 1000
'''

# Initialize the RNN model, loss function, and optimizer
rnn = RNN(input_size, hidden_size, output_size, num_layers, dropout)
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum)

# Train the RNN model
for epoch in range(num_epochs):
    for a, b, c in train_set:
        input_tensor = torch.tensor([int(digit) for digit in a+b+"0"]).view(-1, 1, input_size)
        target_tensor = torch.tensor([int(digit) for digit in "0"+c]).view(-1)

        output, loss = train(rnn, input_tensor, target_tensor, criterion, optimizer)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

# Evaluate the RNN model on the test set
total_loss = 0
with torch.no_grad():
    for a, b, c in test_set:
        input_tensor = torch.tensor([int(digit) for digit in a+b+"0"]).view(-1, 1, input_size)
        target_tensor = torch.tensor([int(digit) for digit in "0"+c]).view(-1)

        hidden = rnn.initHidden()
        loss = 0

        for i in range(input_tensor.size()[0]):
            output, hidden = rnn(input_tensor[i], hidden)
            loss += criterion(output, target_tensor[i])

        total_loss += loss.item() / input_tensor.size()[0]

print(f"Test loss: {total_loss / len(test_set):.4f}")