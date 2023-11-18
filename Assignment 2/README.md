# PHYS449 Assignment 2

This project involves training a Boltzmann Machine (BM) to solve the 1D Ising chain problem. 
Use of AI tools, namely copilot, were used to write code for this project. 

The task is to learn the coupler values for a 1D Ising chain model. The model is trained on a dataset of spin configurations and their corresponding energies. Each configuration is a 1D array of spins, and the energy is a scalar value.

The model architecture consists of a BM with visible units. The model is trained using a Contrastive Divergence (CD) loss function with free energies and Adam optimizer.

The loss and KL-divergence at each epoch is recorded and a plot of the training loss over time is generated and saved to a file (unless verbose mode is set to false). 

The trained model parameters are saved.


## Dependencies

- json
- numpy
- argparse
- torch
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py [-h] [--param param.json] [--data in.txt] [--save boltzmannMachine.pth] [--v BOOLEAN]