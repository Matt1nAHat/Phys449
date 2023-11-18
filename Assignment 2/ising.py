import numpy as np

"""
This script generates spin configurations for a 1D Ising chain model.

The script first defines the number of chains and the coupler values for the Ising model. 
It then generates all possible spin configurations for the chain and calculates their probabilities 
based on their energy.

The script then samples configurations according to these probabilities and writes them to a file. 
The configurations are written as strings of '+' and '-' characters, representing spin up and spin down, respectively.
"""

# Number of chains
num_chains = 5000

# Coupler values
J = np.array([1, -1, 1, -1, 1, 1, -1])

def energy(configuration, J):
    """
    Calculate the energy of a given configuration in the 1D Ising model.

    The energy is calculated as the negative sum of the product of the coupler values 
    and the spins in the configuration.

    Run the script by just using ising.py in the terminal

    Args:
        configuration (numpy.ndarray): The spin configuration. It should be a 1D array 
            of 1s and -1s, where 1 represents spin up and -1 represents spin down.
        J (numpy.ndarray): The coupler values for the Ising model. It should be a 1D array 
            of the same length as the configuration.
    Returns:
        float: The energy of the configuration.
    """
    
    return -np.sum(J[:-1] * configuration[:-1] * configuration[1:])

# Open the file for writing
with open('./data/inTEST.txt', 'w') as f:
    # Generate all possible spin configurations for this chain
    configurations = np.array(np.meshgrid(*[[1, -1] for _ in range(J.shape[0])])).T.reshape(-1,J.shape[0])

    # Calculate the probabilities of the configurations
    probabilities = np.exp(-np.array([energy(c, J) for c in configurations]))
    probabilities /= np.sum(probabilities)

    # Loop over the number of chains
    for _ in range(num_chains):
        # Sample a configuration according to these probabilities
        sampled_configuration = configurations[np.random.choice(range(len(configurations)), p=probabilities)]

        # Write the sampled configuration to the file
        f.write(''.join(['+' if spin == 1 else '-' for spin in sampled_configuration]) + '\n')