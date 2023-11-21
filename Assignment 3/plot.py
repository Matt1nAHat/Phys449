# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, losses, output_dir):

    # Generate the plot
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    # Save the plot to a file
    plt.savefig(f'{output_dir}\\loss_plot.pdf')
    plt.close()
     