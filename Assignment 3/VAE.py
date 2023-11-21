import torch
from torch import nn

class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 4 * 4, 20)  # Latent space size is 20
        self.fc_logvar = nn.Linear(64 * 4 * 4, 20)

        # Decoder
        self.fc_decode = nn.Linear(20, 64 * 4 * 4)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv_transpose2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = torch.relu(self.fc_decode(z))
        z = z.view(z.size(0), 64, 4, 4)  # Un-flatten the tensor
        z = torch.relu(self.conv_transpose1(z))
        z = torch.sigmoid(self.conv_transpose2(z))  # Use sigmoid to output probabilities
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def vae_loss(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 196), x.view(-1, 196), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD