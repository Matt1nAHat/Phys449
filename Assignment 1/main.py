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
    parser = argparse.ArgumentParser(description="Assignment 1: Trains an RNN to perform multiplication of binary integers A * B = C")
    parser.add_argument("--train-size", type=int, default=10000, help="size of the generated training set")
    parser.add_argument("--test-size", type=int, default=1000, help="size of the generated test set")
    parser.add_argument("--seed", type=int, default=1234, help="random seed used for creating the datasets")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--save", default="binaryMult.pth", help="path/file name to save the model")
    args = parser.parse_args()

    train_set, test_set = generate_dataset(args.train_size, args.test_size, args.seed)

    '''print("Training set:")
    for a, b, c in train_set:
        print(f"{a} * {b} = {c}")
    print("Test set:")
    for a, b, c in test_set:
        print(f"{a} * {b} = {c}")'''

 
# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input, hidden):
        # Reshape the input to (batch_size, sequence_length, input_size)
        input = input.view(1, 1, -1)
        # Forward propagate the RNN
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        # Pass the output through the relu activation function
        output = self.relu(output)
        output = self.fc2(output)
        # Pass the output through the sigmoid activation function
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# Define the training function
def train(rnn, input_tensor, target_tensor, criterion, optimizer):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    loss = 0

    '''
    "Working" model that only predicts 0 

    for i in range(input_tensor.size(1)):
        output, hidden = rnn(input_tensor[:, i, :], hidden)
        #print("output.shape:", output.shape)  # Should be [1, num_classes]
        #print("target_tensor[i].unsqueeze(0).shape:", target_tensor[i].unsqueeze(0).shape)  # Should be [1]
        loss += criterion(output, target_tensor[i].unsqueeze(0))'''
    
    '''
    Model with batch size mismatch error, cannot get working

    output, hidden = rnn(input_tensor, hidden)
    print(f"Input tensor batch size: {input_tensor.size(0)}")
    print(f"Output tensor batch size: {output.size(0)}")
    print(f"Target tensor batch size: {target_tensor.size(0)}")
    
    output = output.view(output_size, 1)  # Reshape output to (N, C)
    target_tensor = target_tensor.view(-1)  # Reshape target_tensor to (N)
    print(f"Output tensor batch size: {output.size(0)}")
    print(f"Target tensor batch size: {target_tensor.size(0)}")
    loss = criterion(output, target_tensor)'''

    outputs = [] 

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden)
        #print(output[0])
        #print(target_tensor)
        loss += criterion(output[0], target_tensor)
        #output = output.repeat(input_size, 1, 1)
        #target = target_tensor[i].unsqueeze(1).repeat(1, 1, output.size(2))
        #target = target.permute(1, 0, 2)
        #print(output)
        #print(target_tensor)
        #print(target)
        
    #exit()
    loss.backward()

    optimizer.step()
    #scheduler.step()

    return output, loss.item() / input_tensor.size(1)

# Generate the dataset
train_set, test_set = generate_dataset(train_size=1000, test_size=100, seed=1234)

# Load Hyperparameters from json file
with open(args.param, "r") as f:
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
num_epochs = hyperparams["model"]["num_epochs"]


# Initialize the RNN model, loss function, and optimizer
rnn = RNN(input_size, hidden_size, output_size, num_layers, dropout)
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.NLLLoss()
criterion = nn.BCELoss()
#optimizer = optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0.01)
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
rnn.train() # Switch to training mode
# Train the RNN model
for epoch in range(num_epochs):
    total_loss = 0
    for a, b, c in train_set:

        input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32)
        target_tensor = torch.tensor([[int(digit) for digit in c]], dtype=torch.float32)
        
        '''
        Model that has issue with batch size mismatch

        input_tensor = torch.tensor([int(digit) for digit in (a+b).ljust(input_size, "0")]).float().view(1, -1, input_size)
        target_data = [int(digit) for digit in c.ljust(output_size, "0")]
        target_tensor = torch.tensor(target_data, dtype=torch.long).view(1, -1)'''

        output, loss = train(rnn, input_tensor, target_tensor, criterion, optimizer)
        total_loss += loss

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(train_set):.4e}")

# Evaluate the RNN model on the test set
total_loss = 0
rnn.eval() # Switch to evaluation mode
with torch.no_grad():
    for a, b, c in test_set:
        # Convert the binary strings to tensors
        input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32)
        target_tensor = torch.tensor([[int(digit) for digit in c]], dtype=torch.float32)

        '''
        Model that has issue with batch size mismatch
        
        input_data = [int(digit) for digit in (a+b).ljust(input_size, "0")]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, -1, input_size)
        target_data = [int(digit) for digit in c.ljust(output_size, "0")]
        target_tensor = torch.tensor(target_data, dtype=torch.long).view(-1, 1)'''
        
        # Initialize the hidden state
        hidden = rnn.initHidden()

        '''loss = 0
        output_sequence = []

        #print(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
        for i in range(input_tensor.size(1)):
            output, hidden = rnn(input_tensor[:, i, :], hidden)
            print(output)
            predicted_digit = output.argmax(dim=1).item()
            print(predicted_digit)
            binary_representation = bin(predicted_digit)[2:]  # Convert to binary and remove '0b'
            output_sequence.append(binary_representation)
            loss += criterion(output, target_tensor[i].unsqueeze(0))
        '''
        # Forward pass through the RNN
        output, hidden = rnn(input_tensor, hidden)

        # Apply threshold to output probabilities
        output_sequence = (output >= 0.5).int().squeeze().tolist()
        
        '''print(output)
        print(output_sequence)
        exit()'''
        # Compute the loss
        loss = criterion(output, target_tensor.view(1, 1, -1))
        total_loss += loss.item()

        '''output_sequence = output.argmax(dim=2).squeeze().tolist()
        if isinstance(output_sequence, int):  # Add this line
            output_sequence = [output_sequence]  # Add this line
        output_sequence = [bin(bit)[2:] for bit in output_sequence]  # Convert to binary'''
        output_sequence = ''.join(str(bit) for bit in output_sequence)


        # Print out the test data and the expected and resulting output
        print(f"Test data: {a}, {b}, {c}")
        print(f"Expected output: {c}")
        #print(output)
        print(f"Resulting output: {output_sequence}")

        '''output, hidden = rnn(input_tensor, hidden)
        output_sequence = output.argmax(dim=2).squeeze().tolist()
        
        loss = criterion(output.view(-1, output_size), target_tensor)

        # Print out the test data and the expected and resulting output
        print(f"Test data: {a}, {b}, {c}")
        print(f"Expected output: {target_tensor.tolist()}")
        print(f"Resulting output: {output_sequence}")
        total_loss += loss.item() / input_tensor.size()[0]'''


print(f"Test loss: {total_loss / len(test_set):.4f}")


torch.save(rnn.state_dict(), args.save)