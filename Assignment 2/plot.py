# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, losses):

    # Convert the list to numpy array
    losses = np.array(losses)

    # Generate the plot
    plt.plot(range(epochs), losses[:,0])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    # Save the plot to a file
    plt.savefig('results\\loss_plot.png')
    plt.close()
    # Generate the plot
    plt.plot(range(epochs), losses[:,1])
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('Training KL Divergence Over Time')
    # Save the plot to a file
    plt.savefig('results\\KL_plot.png')