#Import the necessary libraries
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
    #Add arguments to the parser
    args = parser.parse_args()

 
# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNN, self).__init__()
        # Define the model parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        #Define the activation functions
        self.relu = nn.ReLU()
        # Adding fully connected layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input, hidden):
        # Reshape the input to (batch_size, sequence_length, input_size)
        input = input.view(1, 1, -1)
        # Forward propagate the RNN
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        # Pass the output through activation functions and fully connected layers
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        return output, hidden

    def initHidden(self):
        # Initialize the hidden state to zeros
        return torch.zeros(self.num_layers, 1, self.hidden_size)

#Function to train the RNN Model
def train(rnn, input_tensor, target_tensor, criterion, optimizer):
    # Initialize the hidden state
    hidden = rnn.initHidden().to(device)
    # Clear the gradients
    optimizer.zero_grad()
    # Initialize the loss to zero
    loss = 0

    # Forward pass through the RNN for each character in the input string
    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden)
        loss += criterion(output[0], target_tensor)

    # Backpropagate and update the weights    
    loss.backward()
    optimizer.step()

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
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum)



# Initialize lists to store the training and evaluation loss at each epoch
tLosses = [] 
eLosses = []

for epoch in range(num_epochs+1):

    # Train the RNN model
    rnn.train() # Switch to training mode (enables dropout)

    #Initialize the total loss for the epoch
    total_loss = 0
    total_swapped_loss = 0

    #loop over the training set
    for a, b, c in train_set:
        # Convert the binary strings to tensors
        input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32).to(device)
        swapped_input_tensor = torch.tensor([[int(digit) for digit in b+a]], dtype=torch.float32).to(device)
        target_tensor = torch.tensor([[int(digit) for digit in c]], dtype=torch.float32).to(device)

        # Train the model on the current batch and compute the loss
        output, loss = train(rnn, input_tensor, target_tensor, criterion, optimizer)
        total_loss += loss

        # Compute the output and loss for the swapped inputs
        swapped_output, swapped_loss = train(rnn, swapped_input_tensor, target_tensor, criterion, optimizer)
        total_swapped_loss += swapped_loss
    
    # Store the average training loss for the epoch
    tLosses.append([total_loss / len(train_set),total_swapped_loss / len(train_set)])

    #Print the average loss every 10 epochs 
    if epoch % 10 == 0:
        print(f"\033[1mEpoch {epoch}\033[0m:\n\tTrain loss = {total_loss / len(train_set):.4f}\n\tSwapped train loss = {total_swapped_loss / len(train_set):.4f}")


    # Evaluate the RNN model on the test set
    rnn.eval() # Switch to evaluation mode

    #Reset the total loss for the evaluation during current epoch
    total_loss = 0
    total_swapped_loss = 0

    with torch.no_grad():
        for a, b, c in test_set:
            # Convert the binary strings to tensors
            input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32).to(device)
            swapped_input_tensor = torch.tensor([[int(digit) for digit in b+a]], dtype=torch.float32).to(device)
            target_tensor = torch.tensor([[int(digit) for digit in c]], dtype=torch.float32).to(device)

            # Initialize the hidden states
            hidden = rnn.initHidden().to(device)
            swapped_hidden = rnn.initHidden().to(device)

            # Forward pass through the RNN
            output, hidden = rnn(input_tensor, hidden)
            swapped_output, swapped_hidden = rnn(swapped_input_tensor, swapped_hidden)
            
            # Convert probabilities to binary digits
            output_sequence = (output >= 0.5).int().squeeze().tolist()
            swapped_output_sequence = (swapped_output >= 0.5).int().squeeze().tolist()
            
            # Compute the loss
            loss = criterion(output, target_tensor.view(1, 1, -1))
            total_loss += loss.item()
            swapped_loss = criterion(swapped_output, target_tensor.view(1, 1, -1))
            total_swapped_loss += swapped_loss.item()

            #Convert the output sequence to a string
            output_sequence = ''.join(str(bit) for bit in output_sequence)
            swapped_output_sequence = ''.join(str(bit) for bit in swapped_output_sequence)

    # Store the average training loss for the epoch
    eLosses.append([total_loss / len(test_set),total_swapped_loss / len(test_set)])

    #Print the average loss every 10 epochs 
    if epoch % 10 == 0:
        print(f"\tTest loss = {total_loss / len(test_set):.4f}\n\tSwapped test loss = {total_swapped_loss / len(test_set):.4f}")


# Plot the loss over time after training & evaluation
plot_loss(num_epochs+1, tLosses, eLosses)


#Save the trained model
torch.save(rnn.state_dict(), args.save)