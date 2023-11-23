import torch
from torch import nn

class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        """
        Variational Autoencoder (VAE) neural network model.

        The VAEModel class defines a simple VAE architecture with an encoder and decoder.
        The model is designed for grayscale images with a size of 14x14.

        Methods:
        - encode(x): Encodes the input image 'x' into the latent space.
        - reparameterize(mu, logvar): Reparameterizes the latent space for sampling.
        - decode(z): Decodes a point in the latent space to reconstruct an image.
        - forward(x): Performs a forward pass through the VAE (encoding, reparameterization, and decoding).
        - vae_loss(recon_x, x, mu, logvar): Computes the VAE loss, a combination of reconstruction and KL divergence losses.
        """
        # Encoder functions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 4 * 4, 20)  # Latent space size is 20
        self.fc_logvar = nn.Linear(64 * 4 * 4, 20)

        # Decoder functions
        self.fc_decode = nn.Linear(20, 64 * 4 * 4)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv_transpose2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        """
        Encodes the input image 'x' into the latent space.

        Args:
        - x (torch.Tensor): Input image tensor.

        Returns:
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.
        """
        # Run the input through the encoder layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent space for sampling.

        Args:
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.

        Returns:
        - z (torch.Tensor): Reparameterized latent space tensor.
        """
        std = torch.exp(0.5 * logvar) # Compute the standard deviation
        eps = torch.randn_like(std) # Sample from the standard normal distribution
        return mu + eps * std

    def decode(self, z):
        """
        Decodes a point in the latent space to reconstruct an image.

        Args:
        - z (torch.Tensor): Latent space tensor.

        Returns:
        - reconstructed_image (torch.Tensor): Reconstructed image tensor.
        """
        # Run the input through the decoder layers
        z = torch.relu(self.fc_decode(z))
        z = z.view(z.size(0), 64, 4, 4)  # Un-flatten the tensor
        z = torch.relu(self.conv_transpose1(z))
        z = torch.sigmoid(self.conv_transpose2(z))  # Use sigmoid to output probabilities
        return z

    def forward(self, x):
        """
        Performs a forward pass through the VAE (encoding, reparameterization, and decoding).

        Args:
        - x (torch.Tensor): Input image tensor.

        Returns:
        - reconstructed_image (torch.Tensor): Reconstructed image tensor.
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def vae_loss(self, recon_x, x, mu, logvar):
        """
        Computes the VAE loss, a combination of reconstruction and KL divergence losses.

        Args:
        - recon_x (torch.Tensor): Reconstructed image tensor.
        - x (torch.Tensor): Input image tensor.
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.

        Returns:
        - loss (torch.Tensor): VAE loss.
        """
        BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 196), x.view(-1, 196), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD