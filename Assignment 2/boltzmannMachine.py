import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plot import plot_loss

# Boltzmann Machine class
class FVBM(nn.Module):
    def __init__(self, num_visible, batch_size=50, gibbs_steps=10):
        """
        Initialize the Boltzmann Machine.

        Args:
            num_visible (int): The number of visible nodes.
            batch_size (int, optional): The size of the batches used for training. Defaults to 50.
            gibbs_steps (int, optional): The number of Gibbs sampling steps to perform in the forward pass. Defaults to 10.
        """
        super(FVBM, self).__init__()
        # Initialize the hyperparameters
        self.num_visible = num_visible
        self.batch_size = batch_size
        self.gibbs_steps = gibbs_steps
        
        # Initialize weights and biases
        self.bias = nn.Parameter(torch.zeros(num_visible))
        self.weights = nn.Parameter(torch.FloatTensor(num_visible, num_visible))
        nn.init.xavier_uniform_(self.weights)  # Xavier initialization to improve convergence
        self.weights.data.fill_diagonal_(0) # Set the diagonal to 0
        
    def forward(self, visible_nodes):
        """
        Perform a forward pass through the Boltzmann Machine by performing Gibbs sampling.

        Args:
            visible_nodes (Tensor): The visible nodes of the Boltzmann Machine.
        Returns:
            Tensor: The probability of the visible nodes after Gibbs sampling.
        """

        # Gibbs sampling for k steps
        for _ in range(self.gibbs_steps):
            # Determine the probabilities of the visible nodes
            prob_visible, visible_nodes = self.visToVis(visible_nodes)
        return prob_visible 

    def free_energy(self, visible_nodes):
        """
        Calculate the free energy of the system.

        Args:
            visible_nodes (Tensor): The visible nodes of the Boltzmann Machine.
        Returns:
            float: The mean free energy of the system.
        """

        # Compute the negative sum of the visible term and the visible bias term
        visible_bias_term = visible_nodes.mv(self.bias)
        wx_b = F.linear(visible_nodes, self.weights, self.bias)
        visible_term = wx_b.exp().add(1).log().sum(1)
        return torch.mean(-visible_term - visible_bias_term)

    def sample(self, prob):
        """
        Sample new visible nodes from a Bernoulli distribution.

        Args:
            prob (Tensor): The probabilities of the Bernoulli distribution.
        Returns:
            Tensor: A sample from the Bernoulli distribution.
        """
        # Sample from a Bernoulli distribution
        return torch.distributions.Bernoulli(prob).sample()

    def visToVis(self, visible_nodes):
        """
        Perform one step of Gibbs sampling.
        This method computes the probabilities of the visible nodes given the current state of the system,
        and then samples new visible nodes from a Bernoulli distribution with these probabilities.

        Args:
            visible_nodes (Tensor): The current state of the visible nodes.
        Returns:
            prob_vis (Tensor): The computed probabilities of the visible nodes.
            sample_vis (Tensor): The new state of the visible nodes after sampling.
        """
        # Compute the probabilities of the visible nodes after transformation
        prob_vis = torch.sigmoid(F.linear(visible_nodes, self.weights, self.bias)).clamp(min=0, max=1)
        sample_vis = self.sample(prob_vis)
        return prob_vis, sample_vis


    def train(self, training_data, learning_rate, decay, epochs, Lreg, verbose):
        """
        Train the model using contrastive divergence.

        This method approximates the log-likelihood gradient and updates the model's weights and biases accordingly. 
        It uses the Adam optimizer with weight decay for optimization.

        The training data is shuffled at the beginning of training. 
        Persistent visible units are initialized with random values.

        Args:
            training_data (Tensor): The data to train the model on. 
                It should be a 2D tensor where each row is a training example.
            learning_rate (float): The learning rate for the Adam optimizer.
            epochs (int): The number of epochs to train for.
        Returns:
            None
        """
        
        # Initialize an empty list to store the losses
        losses = []

        #Train using contrastive divergence to approximate the log-likelihood gradient and update weights and biases accordingly
        # Initialize the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=decay)

        # Initialize persistent visible units 
        persistent_visible = torch.randn(self.batch_size, self.num_visible)

        # Shuffle the training data
        training_data = training_data[torch.randperm(training_data.size()[0])]

        # Min-Max normalization of training data
        min_val = torch.min(training_data)
        max_val = torch.max(training_data)
        normalized_training_data = (training_data - min_val) / (max_val - min_val)


        # Begin Training Loop 
        for epoch in range(epochs):
            
            # Batch training  
            for i in range(0, normalized_training_data.size()[0], self.batch_size):
                
                # Get the mini-batch 
                mini_batch = normalized_training_data[i:i+self.batch_size]

                # Forward pass with persistent visible units
                persistent_visible = self.sample(self.forward(persistent_visible))

                # Perform one step of CD
                initial_visible_units = mini_batch
                output = self.forward(persistent_visible)
                
                # Compute the loss
                loss = (self.free_energy(initial_visible_units)) - (self.free_energy(output))

                # Add L1 regularization
                l1_lambda = Lreg  # Set the L1 regularization rate
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss += l1_lambda * l1_norm
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

                # Update the weights and biases
                optimizer.step()

                # Update persistent visible units
                persistent_visible = self.sample(self.forward(persistent_visible))

                # Enforce symmetry and 0 diagonal
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
            # Perform final KL divergence calculation
            kl_divergence = torch.sum(training_data_distribution * torch.log((training_data_distribution / (model_samples_distribution + 1e-5)) + 1e-5))

            if verbose:
                # Print the loss for this epoch
                if epoch % 10 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}, KL Divergence: {kl_divergence:.4f}')

                # Add the loss and KL-divergence to the list for plotting
                losses.append([loss.item(), kl_divergence.item()])

        if verbose: 
            # Plot the loss and KL-divergence
            plot_loss(epochs, losses)
        pass

    def predict(self, verbose):
        """
        Predict the coupler values for a 1D Ising chain model.

        This method converts the weights of the Boltzmann machine to a dictionary of coupler values. 
        The keys of the dictionary are tuples representing the indices of the spins in the Ising model, 
        and the values are the weights between these spins.

        Returns:
            couplers (dict): A dictionary where the keys are tuples of spin indices, 
            and the values are the weights between these spins.
        """
    
        # Convert the weights to a numpy array and take the sign of the weights
        weights = np.sign(self.weights.detach().numpy())
        
        # Print the weights
        if verbose:   
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

