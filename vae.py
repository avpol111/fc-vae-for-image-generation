import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
dataset = ImageFolder(root="images-g",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
                                         
class VAE(nn.Module):

    def __init__(self, input_dim=12288, hidden_dim=400, latent_dim=200):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )

        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
    
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return reproduction_loss + KLD


def train(model, optimizer, epochs):
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            x = data[0]
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()

            x_hat, mean, logvar = model(x)
            loss = loss_function(x, x_hat, mean, logvar)

            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", running_loss/len(dataloader))

train(model, optimizer, epochs=50)
# let's generate an image using our VAE:
def generate_image(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float)
    x_decoded = model.decode(z_sample)
    image = x_decoded.detach().reshape(64, 64, 3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

generate_image(0.0, 1.0)
# the values in the brackets determine how different the generated
# image will be from the images the network was trained on          