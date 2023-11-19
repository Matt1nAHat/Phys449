#Import the necessary libraries
import argparse
import json
import torch
from datasetGenerator import generate_dataset
from RNN import model

#Parser for command-line options, arguments and sub-commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 1: Trains an RNN to perform multiplication of binary integers A * B = C")
    parser.add_argument("--param", default=".\param\.\param.json", help="path to the json file containing the hyperparameters")
    parser.add_argument("--test-size", type=int, default=1000, help="size of the generated test set")
    parser.add_argument("--seed", type=int, default=4321, help="random seed used for creating the datasets")
    parser.add_argument("--v", default=False, help="Enable verbose mode to see more information about the testing")
    #Add arguments to the parser
    args = parser.parse_args()

print(args.test_size)

# Define a function to test the model on a dataset
def accuracy(model, test_set):
    """
    Calculates the accuracy of a model on a test set.
    This function loops through the test set, makes predictions for each example, and compares the predictions to the target values. \

    Args:
        model (torch.nn.Module): The model to test.
        test_set (list): The test set, a list of tuples where each tuple contains two lists of binary digits and a string of binary digits.
    Returns:
        accuracy (float): The accuracy of the model on the test set, as a percentage.
    """
    # Initialize the correct count
    correct = 0

    with torch.no_grad():  # No need to track the gradients
        for a, b, c in test_set:
            # Convert the test case to a tensor and move it to the same device as the model
            input_tensor = torch.tensor([[int(digit) for digit in a+b]], dtype=torch.float32)

            # Initialize the hidden states
            hidden = model.initHidden()

            # Make a prediction
            output, _ = model(input_tensor, hidden)
            # Convert probabilities to binary digits
            output_sequence = (output >= 0.5).int().squeeze().tolist()

            #Convert the output sequence to a string
            output_sequence = ''.join(str(bit) for bit in output_sequence)
            # If the output sequence matches the target sequence, increment the correct count
            if output_sequence == c:
                correct += 1
                if args.v:
                    #Print out successful predictions
                    print(output_sequence, c)

    # Calculate and return the accuracy
    accuracy = correct / len(test_set) * 100
    return accuracy

_, test_set = generate_dataset(train_size=1, test_size=args.test_size, seed=args.seed)

# Load Hyperparameters from json file
with open(args.param, "r") as f:
    hyperparams = json.load(f)

# Access the hyperparameters for the model
input_size = hyperparams["model"]["input_size"]
hidden_size = hyperparams["model"]["hidden_size"]
output_size = hyperparams["model"]["output_size"]
num_layers = hyperparams["model"]["num_layers"]
dropout = hyperparams["model"]["dropout"]

# Define the model architecture
testModel = model(input_size, hidden_size, output_size, num_layers, dropout)
# Load the state_dict into the model
testModel.load_state_dict(torch.load('.\\results\.\\binaryMult.pth'))
# Set the model to evaluation mode
testModel.eval()
# Calculate the accuracy of the model on the test set
print(f"Accuracy: {accuracy(testModel, test_set):.2f}%")