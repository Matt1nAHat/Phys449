# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, tLosses, eLosses):

    # Convert the lists to numpy arrays
    tLosses = np.array(tLosses)
    eLosses = np.array(eLosses)
    # Generate the plot
    plt.plot(range(epochs), tLosses[:,0], label='Training Loss (a*b)')
    plt.plot(range(epochs), tLosses[:,1], label='Training Loss (b*a)')
    plt.plot(range(epochs), eLosses[:,0], label='Evaluation Loss (a*b)')
    plt.plot(range(epochs), eLosses[:,1], label='Evaluation Loss (b*a)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Save the plot to a file
    plt.savefig('loss_plot.png')