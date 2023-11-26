# PHYS449 Assignment 3

This assignment focuses on Variational Autoencoders (VAEs), a type of generative model that uses deep learning methods to both model and generate data.
Use of AI tools, namely copilot, were used to write code for this project. 

The task is to learn to write even numbers from 0 to 8. The model is trained on the MNIST dataset containing only even numbers which have been downsized to 14 x 14 pixels. 3000 of the 29492 samples have been set aside for testing the model.

The model architecture consists of a VAE with convolutional units in the encoder. The model is trained using a loss function that combines a reconstruction loss and a KL-divergence loss, and an Adam optimizer.

The loss at each epoch is recorded and a plot of the training loss over time is generated and saved to a file (unless verbose mode is set to false). 

The trained model parameters are saved.


## Dependencies

- json
- pandas
- argparse
- torch
- matplotlib
- os

## Running `main.py`

To run `main.py` using the suggested parameters, after pulling the repository simply use

```sh
python main.py 
```

To skip training a new model and generate samples using the pre-trained model in the results folder, run the following command:

```sh
python main.py --test .\\results\.\\mnistVAE.pth
```

In the case that you would like to fully customize the inputs, the parameters can be adjusted by changing any of the following:

```sh
python main.py -h --param .\param\.\param.json --data .\data\.\even_mnist.csv --save .\\results\.\\mnistVAE.pth --o .\\results_dir\. --v True --n 100 --test None
```

To further adjust the hyperparameters, changes can be made in the param.json file found in the param folder