# Import the necessary libraries
import matplotlib.pyplot as plt

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, losses, output_dir):
    """
    This function generates a line plot of the training loss over a specified number of epochs. 
    The plot is saved as a PDF in the specified output directory.

    Args:
    - epochs (int): The total number of epochs for which the model was trained.
    - losses (list or np.array): A list or array containing the loss value at each epoch.
    - output_dir (str): The directory where the plot PDF should be saved.

    Returns:
    Plot of the loss saved as PDFs in the output directory.
    """
    # Generate the plot
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    # Save the plot to a file
    plt.savefig(f'{output_dir}\\loss_plot.pdf')
    plt.close()
     