import torch
import matplotlib.pyplot as plt
from VAE import VAEModel

def test(modelPath, n, output_dir='.\\results\.'):
    """
    Generate and save reconstructed samples using a trained VAE model.

    Args:
    - modelPath (str): Path to the saved state dictionary of the trained VAE model.
    - n (int): Number of samples to generate and save.
    - output_dir (str, optional): Directory to save the generated samples as PDF files. Default is '.\\results\\'.

    Returns:
    N sample images saved as PDFs in the output directory.
    """
    # Load the trained model
    testModel = VAEModel()
    testModel.load_state_dict(torch.load(modelPath))
    testModel.eval() # Set the model to evaluation mode

    # Generate n samples
    z = torch.randn((n, 20))  
    samples = testModel.decode(z)

    # Convert each sample to an image and save it as a PDF file
    for i in range(n):
        sample = samples[i].detach().cpu().numpy()
        sample = sample.squeeze() # Remove the channel dimension to get a 2D image
        plt.imshow(sample, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{output_dir}\\{i+1}.pdf', format='pdf')