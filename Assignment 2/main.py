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
    def __init__(self, num_visible, num_hidden, batch_size=10, k=1):
        super(BoltzmannMachine, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.k = k
        
        # Initialize weights and biases
        self.weights = nn.Parameter(torch.randn(num_hidden, num_visible))
        self.bias_v = nn.Parameter(torch.zeros(num_visible))
        self.bias_h = nn.Parameter(torch.zeros(num_hidden))
        
    def forward(self, visible_nodes):
        prob_hidden, hidden = self.visToHid(visible_nodes)
        
        for i in range(self.k):
            prob_visible, visible = self.hidToVis(hidden)
            prob_hidden, hidden = self.visToHid(visible)    
        
        return prob_visible 

    def free_energy(self, visible_nodes):
        # Calculate the free energy of the system
        visible_bias_term = visible_nodes.mv(self.bias_v)
        wx_b = F.linear(visible_nodes, self.weights, self.bias_h)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        #hidden_term = torch.log(1 + torch.exp(wx_b)).sum(1)
        return (-hidden_term - visible_bias_term).mean()

    def sample(self, prob):
        # Sample from a Bernoulli distribution
        #return F.relu(torch.sign(prob - torch.rand_like(prob.size())))
        return torch.distributions.Bernoulli(prob).sample()

    def visToHid(self, visible_nodes):
        prob_hid = F.sigmoid(F.linear(visible_nodes, self.weights, self.bias_h))
        sample_hid = self.sample(prob_hid)
        return prob_hid, sample_hid
    
    def hidToVis(self, hidden_nodes):
        prob_vis = F.sigmoid(F.linear(hidden_nodes, self.weights.t(), self.bias_v))
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

            # Batch training
            for i in range(0, training_data.size()[0], self.batch_size):
                # Get the mini-batch
                mini_batch = training_data[i:i+self.batch_size]

                # Perform one step of CD
                initial_visible_units = mini_batch
                
                visible_units_after_k_steps = self.forward(initial_visible_units)
                loss = torch.mean(self.free_energy(initial_visible_units)) - torch.mean(self.free_energy(visible_units_after_k_steps))
                
                #initial_hidden_probabilities, _ = self.visToHid(initial_visible_units)
                #hidden_probabilities_after_k_steps, _ = self.visToHid(visible_units_after_k_steps)

                # Compute the loss
                #loss = torch.mean(self.free_energy(initial_visible_units)) - torch.mean(self.free_energy(visible_units_after_k_steps))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
            
            
            

            normalized_training_data = (training_data + 1) / 2
            #print(self.forward(normalized_training_data))
            #print(normalized_training_data / self.forward(normalized_training_data))
            #print(torch.log(normalized_training_data / self.forward(normalized_training_data)))
            
            # Compute the KL divergence
            kl_divergence = torch.sum(normalized_training_data * torch.log(normalized_training_data / self.forward(normalized_training_data) + 1e-5))

            # Print the loss for this epoch
            print('Epoch: {}, Loss: {}, KL Divergence: {}'.format(epoch+1, loss, kl_divergence))

            losses.append([loss.item(), kl_divergence.item()])

        plot_loss(num_epochs, losses)
        pass

    def predict(self, test_data):
        v = test_data
        prob_hid, _ = self.visToHid(v)
        return self.sample(prob_hid)



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
predictions = bm.predict(test_data)
print(predictions)
