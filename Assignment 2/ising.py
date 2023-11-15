import numpy as np

# Number of spins in each chain
N = 4

# Number of chains
num_chains = 100

# Coupler values
J = np.array([-1, -1, -1, 1])

# Open the file for writing
with open('./data/out.txt', 'w') as f:
    # Loop over the number of chains
    for _ in range(num_chains):
        # Generate all possible spin configurations for this chain
        configurations = np.array(np.meshgrid(*[[1, -1] for _ in range(N)])).T.reshape(-1,N)

        # Write the configurations to the file
        for configuration in configurations:
            f.write(''.join(['+' if spin == 1 else '-' for spin in configuration]) + '\n')