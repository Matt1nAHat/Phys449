# Import the necessary libraries
import matplotlib.pyplot as plt

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, tLosses, tSwapped_losses, eLosses, eSwapped_losses):

    # Generate the plot
    plt.plot(range(epochs), tLosses, label='Training Loss (a*b)')
    plt.plot(range(epochs), tSwapped_losses, label='Training Loss (b*a)')
    #plt.plot(range(epochs), eLosses, label='Evaluation Loss (a*b)')
    #plt.plot(range(epochs), eSwapped_losses, label='Evaluation Loss (b*a)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Save the plot to a file
    plt.savefig('loss_plot.png')