import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datasetGenerator import generate_dataset
from plot import plot_loss

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 1: Trains an RNN to perform multiplication of binary integers A * B = C")
    parser.add_argument("--train-size", type=int, default=10000, help="size of the generated training set")
    parser.add_argument("--test-size", type=int, default=1000, help="size of the generated test set")
    parser.add_argument("--seed", type=int, default=1234, help="random seed used for creating the datasets")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--save", default="binaryMult.pth", help="path/file name to save the model")
    args = parser.parse_args()

    train_set, test_set = generate_dataset(args.train_size, args.test_size, args.seed)

 
# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # New fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # New fully connected layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # New fully connected layer
        #self.sigmoid = nn.Sigmoid()
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
        output = self.relu(output)
        output = self.fc3(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# Define the training function
def train(rnn, input_tensor, target_tensor, criterion, optimizer):
    hidden = rnn.initHidden().to(device)
    optimizer.zero_grad()
    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden)
        loss += criterion(output[0], target_tensor)
        
    loss.backward()

    optimizer.step()
    #scheduler.step()

    return output, loss.item()

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
rnn = rnn.to(device)  # Move the model to the GPU if available
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.NLLLoss()
#criterion = nn.BCELoss()
criterion = nn.MSELoss()
#optimizer = optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=0.01)
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
rnn.train() # Switch to training mode

# Train the RNN model
losses = []
for epoch in range(num_epochs):
    
    total_loss = 0
    for a, b, c in train_set:

        input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32).to(device)
        target_tensor = torch.tensor([[int(digit) for digit in c]], dtype=torch.float32).to(device)

        output, loss = train(rnn, input_tensor, target_tensor, criterion, optimizer)
        total_loss += loss

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(train_set):.4e}")
        losses.append(total_loss / len(train_set))

plot_loss(num_epochs, losses)

# Evaluate the RNN model on the test set
total_loss = 0
rnn.eval() # Switch to evaluation mode
with torch.no_grad():
    for a, b, c in test_set:
        # Convert the binary strings to tensors
        input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32).to(device)
        target_tensor = torch.tensor([[int(digit) for digit in c]], dtype=torch.float32).to(device)

        
        # Initialize the hidden state
        hidden = rnn.initHidden().to(device)

        # Forward pass through the RNN
        output, hidden = rnn(input_tensor, hidden)

        # Apply threshold to output probabilities
        output_sequence = (output >= 0.5).int().squeeze().tolist()
        
        # Compute the loss
        loss = criterion(output, target_tensor.view(1, 1, -1))
        total_loss += loss.item()

        
        output_sequence = ''.join(str(bit) for bit in output_sequence)


        # Print out the test data and the expected and resulting output
        print(f"Test data: {a}, {b}, {c}")
        print(f"Expected output: {c}")
        #print(output)
        print(f"Resulting output: {output_sequence}")



print(f"Test loss: {total_loss / len(test_set):.4f}")


torch.save(rnn.state_dict(), args.save)