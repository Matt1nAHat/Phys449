# Import the necessary libraries
import matplotlib.pyplot as plt

# This function assumes that 'train' returns the loss at each epoch
def plot_loss(epochs, losses):


    print(losses)
    print(range(0, epochs, 10))
    # Generate the plot
    plt.plot(range(0, epochs, 10), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')

    # Save the plot to a file
    plt.savefig('loss_plot.png')