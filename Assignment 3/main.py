import argparse
import json
import torch
import torch.optim as optim
from VAE import VAEModel
from loadData import get
from plot import plot_loss
from testModel import test
import os

#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3: Trains a VAE on MNIST data and generates even numbered samples from the trained model.")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--data", default=".\data\.\even_mnist.csv", help="path to the json file containing the hyperparameters")
    parser.add_argument("--save", default=".\\results\.\\mnistVAE.pth", help="path/file name to save the model")
    parser.add_argument("--o", default=".\\results_dir\.", help="path/file name to save the model")
    parser.add_argument("--v", default=True, help="Enable verbose mode to see more information about the training process")
    parser.add_argument('--n', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--test', default=None, help='Skip training/evaluations and test a pretrained model')

    #Add arguments to the parser
    args = parser.parse_args()

# Get the output directory from the command line arguments
output_dir = args.o

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Train new model if testing mode is not enabled
if args.test == None:
        
    # Load Hyperparameters from json file
    with open(args.param, "r") as f:
        hyperparams = json.load(f)

    # Access the hyperparameters for the optimization algorithm
    learning_rate = hyperparams["optim"]["lr"]

    # Access the hyperparameters for the model
    batch_size = hyperparams["model"]["batch_size"]
    num_epochs = hyperparams["model"]["num_epochs"]

    # Create an instance of the VAE and an optimizer
    model = VAEModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load the data from csv file
    train_data, test_data = get(args.data, batch_size)
    losses = []

    '''Training loop'''
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_data):
            data = data.view(-1, 1, 14, 14)  # Reshape the data
            optimizer.zero_grad()  # Zero the gradients
            recon_batch, mu, logvar = model(data) # Forward pass
            loss = model.vae_loss(recon_batch, data, mu, logvar) # Compute the loss
            loss.backward()  # Backpropagate the loss
            train_loss += loss.item()
            optimizer.step()  # Update the weights
        
        
        #Save and print the average loss every 5 epochs 
        losses.append(train_loss / len(train_data.dataset))
        
        #Only print if verbose mode is enabled
        if args.v:
            if epoch % 5 == 0:
                print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_data.dataset):.4f}')

    #Plot the loss over time
    plot_loss(num_epochs, losses, output_dir)

    '''Evaluation loop'''

    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    with torch.no_grad():  # Disable gradient tracking
        for i, (data, _) in enumerate(test_data):
            data = data.view(-1, 1, 14, 14)  # Reshape the data
            recon_batch, mu, logvar = model(data) # Forward pass
            test_loss += model.vae_loss(recon_batch, data, mu, logvar).item() # Compute the loss

    test_loss /= len(test_data.dataset)
    print(f'Test set loss: {test_loss}')

    #Save the trained model
    torch.save(model.state_dict(), args.save)

    #Generate samples from the trained model
    test(args.save, args.n, output_dir)

# Test a pretrained model if testing mode is enabled
else:
    test(args.test, args.n, output_dir)