import numpy as np

# Number of spins in each chain
N = 4

# Number of chains
num_chains = 1000

# Coupler values
J = np.array([-1, 1, 1, 1])

# Calculate the energy of a configuration
def energy(configuration, J):
    return -np.sum(J[:-1] * configuration[:-1] * configuration[1:])

# Open the file for writing
with open('./data/inTEST.txt', 'w') as f:
    # Generate all possible spin configurations for this chain
    configurations = np.array(np.meshgrid(*[[1, -1] for _ in range(N)])).T.reshape(-1,N)

    # Calculate the probabilities of the configurations
    probabilities = np.exp(-np.array([energy(c, J) for c in configurations]))
    probabilities /= np.sum(probabilities)

    # Loop over the number of chains
    for _ in range(num_chains):
        # Sample a configuration according to these probabilities
        sampled_configuration = configurations[np.random.choice(range(len(configurations)), p=probabilities)]

        # Write the sampled configuration to the file
        f.write(''.join(['+' if spin == 1 else '-' for spin in sampled_configuration]) + '\n')