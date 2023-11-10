# PHYS449 Assignment 1

This project involves training a Recurrent Neural Network (RNN) to solve the binary multiplication task. 
Use of AI tools, namely copilot, were used to write code for this project. 

The task is to learn to multiply two 8-bit binary numbers. The model is trained on a dataset of binary integers A, B, and C where C = A * B. Each integer has at most 8 digits in their binary representation.

The model architecture consists of an RNN layer followed by a dropout layer for regularization and then three fully connected layers. The model is trained using an MSE loss function and stochastic gradient descent optimizer.

The loss at each epoch is recorded and a plot of the training loss over time is generated and saved to a file. 

After training, the model can be used to make predictions on new data. The trained model parameters are saved and can be loaded to make predictions on new binary multiplication tasks.


## Dependencies

- json
- numpy
- argparse
- torch
- random
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py [-h] [--param param.json] [--train-size INT] [--test-size INT] [--seed INT] [--save binaryMult.pth]
```
