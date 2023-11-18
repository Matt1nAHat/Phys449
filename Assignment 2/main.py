import argparse
import json
import torch
from boltzmannMachine import FVBM

#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2: Trains a Boltzmann machine on data from a 1-D classical Ising chain to predict coupler values")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--data", default=".\data\.\in.txt", help="path to the json file containing the hyperparameters")
    parser.add_argument("--save", default=".\\results\.\\boltzmannMachine.pth", help="path/file name to save the model")
    parser.add_argument("--v", default=True, help="Enable verbose mode to see more information about the training process")
    #Add arguments to the parser
    args = parser.parse_args()

#Read in data and convert + and - to integer values
with open(args.data, 'r') as file:
    lines = file.readlines()
spins = torch.tensor([[1 if spin == '+' else -1 for spin in line.strip()] for line in lines], dtype=torch.float32)

# Load Hyperparameters from json file
with open(args.param, "r") as f:
    hyperparams = json.load(f)

# Access the hyperparameters for the optimization algorithm
learning_rate = hyperparams["optim"]["lr"]
weight_decay = hyperparams["optim"]["weight_decay"]

# Access the hyperparameters for the model
num_epochs = hyperparams["model"]["num_epochs"]
gibbs_steps = hyperparams["model"]["gibbs_steps"]
batch_size = hyperparams["model"]["batch_size"]
Lreg = hyperparams["model"]["L1_reg"]

# Define the number of visible units
num_visible = len(spins[0])

# Create a BoltzmannMachine instance
bm = FVBM(num_visible, batch_size, gibbs_steps)

# Train the BoltzmannMachine
bm.train(spins, learning_rate, weight_decay, num_epochs, Lreg, args.v) 

# Test the BoltzmannMachine
predictions = bm.predict(args.v)
if args.v:
    print(predictions)

#Save the trained model
torch.save(bm.state_dict(), args.save)
