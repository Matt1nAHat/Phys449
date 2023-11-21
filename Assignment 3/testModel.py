import torch
import matplotlib.pyplot as plt
import argparse
from torchvision.utils import make_grid
from VAE import VAEModel

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate digit samples')
parser.add_argument('-n', type=int, default=100, help='Number of samples to generate')
parser.add_argument("--path", default=".\\results\.\\mnistVAE.pth", help="path/file name to load the model from")
args = parser.parse_args()

def test(modelPath, n, output_dir='.\\results\.'):
    # Load the trained model
    testModel = VAEModel()
    testModel.load_state_dict(torch.load(modelPath))
    testModel.eval()

    # Generate n samples
    z = torch.randn((n, 20))  # Replace model.latent_dim with the size of your model's latent space
    samples = testModel.decode(z)

    # Convert each sample to an image and save it as a PDF file
    for i in range(args.n):
        sample = samples[i].detach().cpu().numpy()
        sample = sample.squeeze()
        plt.imshow(sample, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{output_dir}\\{i+1}.pdf', format='pdf')

        
test(args.path, args.n)
