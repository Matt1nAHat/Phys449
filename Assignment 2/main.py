import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from plot import plot_loss

#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2: Trains a Boltzmann machine on data from a 1-D classical Ising chain")
    #parser.add_argument("--train-size", type=int, default=10000, help="size of the generated training set")
    #parser.add_argument("--test-size", type=int, default=1000, help="size of the generated test set")
    #parser.add_argument("--seed", type=int, default=1234, help="random seed used for creating the datasets")
    #parser.add_argument("--save", default="binaryMult.pth", help="path/file name to save the model")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--data", default=".\data\.\in.txt", help="path to the json file containing the hyperparameters")
    #Add arguments to the parser
    args = parser.parse_args()


# Boltzmann Machine class
class BoltzmannMachine(nn.Module):
    def __init__(self, num_visible, batch_size=100, gibbs_steps=10):
        super(BoltzmannMachine, self).__init__()
        self.num_visible = num_visible
        self.batch_size = batch_size
        self.gibbs_steps = gibbs_steps
        
        # Initialize weights and biases
        self.weights = nn.Parameter(torch.randn(num_visible, num_visible))
        self.bias = nn.Parameter(torch.zeros(num_visible))
        
    def forward(self, visible_nodes):
        for _ in range(self.gibbs_steps):
            prob_visible, visible = self.visToVis(visible_nodes)
        return prob_visible 

    def free_energy(self, visible_nodes):
        # Calculate the free energy of the system
        visible_bias_term = visible_nodes.mv(self.bias)
        wx_b = F.linear(visible_nodes, self.weights, self.bias)
        visible_term = wx_b.exp().add(1).log().sum(1)
        return (-visible_term - visible_bias_term).mean()

    def sample(self, prob):
        # Sample from a Bernoulli distribution
        #return F.relu(torch.sign(prob - torch.rand_like(prob.size())))
        return torch.distributions.Bernoulli(prob).sample()

    def visToVis(self, visible_nodes):
        prob_vis = F.sigmoid(F.linear(visible_nodes, self.weights, self.bias))
        sample_vis = self.sample(prob_vis)
        return prob_vis, sample_vis


    def train(self, training_data, learning_rate, epochs):
        losses = []
        #Train using contrastive divergence to approximate the log-likelihood gradient and update weights and biases accordingly

        # Initialize the optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            # Shuffle the training data
            training_data = training_data[torch.randperm(training_data.size()[0])]

            # Assuming training_data is your data tensor
            min_val = torch.min(training_data)
            max_val = torch.max(training_data)

            # Min-Max normalization
            normalized_training_data = (training_data - min_val) / (max_val - min_val)

            # Batch training
            for i in range(0, normalized_training_data.size()[0], self.batch_size):
                # Get the mini-batch
                mini_batch = normalized_training_data[i:i+self.batch_size]

                # Perform one step of CD
                initial_visible_units = mini_batch
                
                visible_units_after_k_steps = self.forward(initial_visible_units)
                loss = torch.mean(self.free_energy(initial_visible_units)) - torch.mean(self.free_energy(visible_units_after_k_steps))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
                        
            # Compute the KL divergence
            # Compute the KL divergence
            model_samples = self.forward(normalized_training_data).detach()
            model_samples_distribution = torch.histc(model_samples, bins=10, min=0, max=1)
            training_data_distribution = torch.histc(normalized_training_data, bins=10, min=0, max=1)

            kl_divergence = torch.sum(training_data_distribution * torch.log((training_data_distribution / (model_samples_distribution + 1e-5)) + 1e-5))

            # Print the loss for this epoch
            print('Epoch: {}, Loss: {}, KL Divergence: {}'.format(epoch+1, loss, kl_divergence))

            losses.append([loss.item(), kl_divergence.item()])

        plot_loss(num_epochs, losses)
        pass

    def predict(self):
        # Get the weights of the Boltzmann Machine
        weights = self.weights.detach().numpy()
        print(weights)
        # Initialize an empty dictionary to store the coupler values
        couplers = {}

        # Loop over the weights and add them to the dictionary
        for i in range(weights.shape[1] - 1):  # Subtract 1 to avoid index out of bounds
            j = (i + 1) # The index of the next spin in the loop
            couplers[(i, j)] = weights[i, j]

        # Add the last coupler for the closed loop
        couplers[(weights.shape[1] - 1, 0)] = weights[weights.shape[1] - 1, 0]

        return couplers



#Read in data and convert + and - to integer values
with open(args.data, 'r') as file:
    lines = file.readlines()
spins = torch.tensor([[1 if spin == '+' else -1 for spin in line.strip()] for line in lines], dtype=torch.float32)

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

# Define the number of visible and hidden units
num_visible = len(spins[0])
num_hidden = 10  # You can adjust this value

# Create a BoltzmannMachine instance
bm = BoltzmannMachine(num_visible, num_hidden)

# Train the BoltzmannMachine
bm.train(spins, learning_rate, num_epochs) 

# Test the BoltzmannMachine
test_data = spins[:10]  # Use the first 10 samples for testing
predictions = bm.predict()
print(predictions)
