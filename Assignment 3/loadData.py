from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import torch

def get(path, batch_size):
    """
    Load and preprocess image data from the specified path, split it into training and test sets,
    and create PyTorch DataLoaders for efficient batch processing.

    Args:
    - path (str): The file path to the data file. The data should be in a whitespace-delimited format.
    - batch_size (int): The number of samples in each batch for training and testing DataLoader.

    Returns:
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    """
    # Load the data
    data = pd.read_csv(path, delim_whitespace=True)

    # Split the data and labels
    X = data.iloc[:, :-1] / 255.0  # normalize pixel values to 0-1
    y = data.iloc[:, -1]

    # Determine the split index
    split_index = len(data) - 3000  # 3000 for testing, adjust as needed

    # Split the data into training and test sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Convert to numpy arrays and reshape
    X_train = X_train.values.reshape(-1, 1, 14, 14)
    X_test = X_test.values.reshape(-1, 1, 14, 14)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train.values).long()
    y_test = torch.from_numpy(y_test.values).long()
    
    # Create TensorDatasets
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    # Create and return DataLoaders
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size)
    return train_loader, test_loader
