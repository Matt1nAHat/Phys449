import torch
import torch.nn as nn


# Define the RNN model
class model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(model, self).__init__()
        """
        A simple RNN model.

        This model uses an RNN layer followed by three fully connected layers and ReLU activations.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output.
            num_layers (int): The number of layers in the RNN.
            dropout (float): The dropout probability.
        """
        # Define the model parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        #Define the activation functions
        self.relu = nn.ReLU()
        # Adding fully connected layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input, hidden):
        """
        Defines the forward pass of the RNN.
        This method reshapes the input to (batch_size, sequence_length, input_size), then forward propagates it through the RNN. 
        The output is then passed through a dropout layer, three fully connected layers, and ReLU activations.
        
        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
            hidden (torch.Tensor): The initial hidden state for the RNN.
        Returns:
            output (torch.Tensor): The output of the RNN.
            hidden (torch.Tensor): The final hidden state of the RNN.
        """
        # Reshape the input to (batch_size, sequence_length, input_size)
        input = input.view(1, 1, -1)
        # Forward propagate the RNN
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        # Pass the output through activation functions and fully connected layers
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        return output, hidden

    def initHidden(self):
        """
        Initializes the hidden state of the RNN to zeros before the start of forward propagation.

        Returns:
            hidden (torch.Tensor): A tensor filled with zeros of shape (num_layers, 1, hidden_size).    
        """
        return torch.zeros(self.num_layers, 1, self.hidden_size)


