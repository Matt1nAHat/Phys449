import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from plot import plot_loss

#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2: Trains a Boltzmann machine on data from a 1-D classical Ising chain")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--data", default=".\data\.\in.txt", help="path to the json file containing the hyperparameters")
    #Add arguments to the parser
    args = parser.parse_args()


# Boltzmann Machine class
class BoltzmannMachine(nn.Module):
    def __init__(self, num_visible, batch_size=50, gibbs_steps=10):
        super(BoltzmannMachine, self).__init__()
        self.num_visible = num_visible
        self.batch_size = batch_size
        self.gibbs_steps = gibbs_steps
        
        # Initialize weights and biases
        #self.weights = nn.Parameter(torch.FloatTensor(num_visible, num_visible).uniform_(-0.05, 0.05))
        #self.weights.data.fill_diagonal_(0)
        self.bias = nn.Parameter(torch.zeros(num_visible))
        self.weights = nn.Parameter(torch.FloatTensor(num_visible, num_visible))
        nn.init.xavier_uniform_(self.weights)  # Xavier initialization
        
    def forward(self, visible_nodes):
        # Perform Gibbs sampling for k steps
        for _ in range(self.gibbs_steps):
            prob_visible, visible_nodes = self.visToVis(visible_nodes)
        return prob_visible 

    def free_energy(self, visible_nodes):
        # Calculate the free energy of the system
        visible_bias_term = visible_nodes.mv(self.bias)
        wx_b = F.linear(visible_nodes, self.weights, self.bias)
        visible_term = wx_b.exp().add(1).log().sum(1)
        return (-visible_term - visible_bias_term).mean()

    def sample(self, prob):
        # Sample from a Bernoulli distribution
        return torch.distributions.Bernoulli(prob).sample()

    def visToVis(self, visible_nodes):
        prob_vis = torch.sigmoid(F.linear(visible_nodes, self.weights, self.bias))
        sample_vis = self.sample(prob_vis)
        return prob_vis, sample_vis


    def train(self, training_data, learning_rate, epochs):
        losses = []
        #Train using contrastive divergence to approximate the log-likelihood gradient and update weights and biases accordingly

        # Initialize the optimizer
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Initialize persistent visible units
        persistent_visible = torch.randn(self.batch_size, self.num_visible)


        # Shuffle the training data
        training_data = training_data[torch.randperm(training_data.size()[0])]

        # Assuming training_data is your data tensor
        min_val = torch.min(training_data)
        max_val = torch.max(training_data)

        # Min-Max normalization
        normalized_training_data = (training_data - min_val) / (max_val - min_val)


        # Training loop
        for epoch in range(epochs):
            
        
            # Batch training
            for i in range(0, normalized_training_data.size()[0], self.batch_size):
                # Get the mini-batch
                mini_batch = normalized_training_data[i:i+self.batch_size]

                # Perform one step of CD
                initial_visible_units = mini_batch
                
                #visible_units_after_k_steps = self.forward(initial_visible_units)
                # Forward pass with persistent visible units
                output = self.forward(persistent_visible)

                loss = torch.mean(self.free_energy(initial_visible_units)) - torch.mean(self.free_energy(output))

                # Add L1 regularization
                l1_lambda = 0.0001  # Set the L1 regularization rate
                l1_norm = sum(p.abs().sum() for p in bm.parameters())
                loss += l1_lambda * l1_norm
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                optimizer.step()

                # Update persistent visible units
                persistent_visible = self.sample(self.forward(persistent_visible))

                # Enforce symmetry
                self.weights.data = (self.weights.data + self.weights.data.t()) / 2
                self.weights.data.fill_diagonal_(0)


                # Zero the gradients
                self.weights.grad.data.zero_()

    
                        
            # Compute the KL divergence
            model_samples = self.forward(normalized_training_data).detach()
            model_samples_distribution = torch.histc(model_samples, bins=50, min=0, max=1)
            training_data_distribution = torch.histc(normalized_training_data, bins=50, min=0, max=1)

            # Normalize the histograms
            model_samples_distribution /= torch.sum(model_samples_distribution)
            training_data_distribution /= torch.sum(training_data_distribution)

            kl_divergence = torch.sum(training_data_distribution * torch.log((training_data_distribution / (model_samples_distribution + 1e-5)) + 1e-5))

            # Print the loss for this epoch
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}, KL Divergence: {kl_divergence:.4f}')

            losses.append([loss.item(), kl_divergence.item()])

        plot_loss(num_epochs, losses)
        pass

    def predict(self):
        # Set the diagonal elements to zero
        #self.weights.data = torch.tanh(self.weights.data)
        weights = np.sign(self.weights.detach().numpy())
        
        print(weights)
        # Initialize an empty dictionary to store the coupler values
        couplers = {}

        # Loop over the weights and add them to the dictionary
        for i in range(weights.shape[0] - 1):  # Subtract 1 to avoid index out of bounds
            # The key is a tuple of the indices
            key = (i, i + 1)

            # The value is the weight between the i-th and (i+1)-th visible units
            value = weights[i, i + 1]

            # Add the coupler value to the dictionary
            couplers[key] = value

        # Add the last coupler for the closed loop
        couplers[(weights.shape[0] - 1, 0)] = weights[weights.shape[0] - 1, 0]
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

# Access the hyperparameters for the model
num_epochs = hyperparams["model"]["num_epochs"]
gibbs_steps = hyperparams["model"]["gibbs_steps"]

# Define the number of visible units
num_visible = len(spins[0])

# Create a BoltzmannMachine instance
bm = BoltzmannMachine(num_visible, gibbs_steps)

# Train the BoltzmannMachine
bm.train(spins, learning_rate, num_epochs) 

# Test the BoltzmannMachine
predictions = bm.predict()
print(predictions)
