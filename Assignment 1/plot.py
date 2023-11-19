# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, tLosses, eLosses):
    """
    Plots the training and evaluation loss over time for both normal and swapped inputs.

    Args:
        epochs (int): The number of epochs for training the model.
        tLosses (list): A list of tuples where each tuple contains the training loss for (a*b) and (b*a) at each epoch.
        eLosses (list): A list of tuples where each tuple contains the evaluation loss for (a*b) and (b*a) at each epoch.
    """

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
    plt.savefig('results\\loss_plot.png')