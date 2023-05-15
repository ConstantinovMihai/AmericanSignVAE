# Setup
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import try_gpu

plt.rcParams['figure.dpi'] = 200
s_img=28

class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dims, hdim[1])
        self.linear2 = nn.Linear(hdim[1], hdim[0])
        self.linear3 = nn.Linear(hdim[0], s_img * s_img)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        ########################################################################
        #    TODO: Apply full forward function                                 #
        #    NOTE: Please have a close look at the forward function of the     #
        #    encoder                                                           #
        ########################################################################

        z = self.relu(self.linear1(z))
        z = self.relu(self.linear2(z))
        z = self.sigmoid(self.linear3(z))
        z = z.reshape((-1, 1, s_img, s_img))

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

        return z


# encoder
class VarEncoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim):
        super(VarEncoder, self).__init__()

        # layers for g1
        self.linear1_1 = nn.Linear(s_img * s_img, hdim[0])
        self.linear2_1 = nn.Linear(hdim[0], hdim[1])
        self.linear3_1 = nn.Linear(hdim[1], latent_dims)

        # layers for g2
        self.linear1_2 = nn.Linear(s_img * s_img, hdim[0])
        self.linear2_2 = nn.Linear(hdim[0], hdim[1])
        self.linear3_2 = nn.Linear(hdim[1], latent_dims)

        self.relu = nn.ReLU()

        # distribution setup
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(try_gpu())  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(try_gpu())
        self.kl = 0

    ########################################################################
    #    TODO: Define function for:                                        #
    #    1. the Kullback-Leibner loss "kull_leib"                          #
    #    2. the Reparameterization trick                                   #
    ########################################################################

    def kull_leib(self, mu, sigma):
        return (sigma ** 2 + mu ** 2 - torch.clip(torch.log(sigma), min=1e-5, max=1e5) - 1 / 2).sum()

    def reparameterize(self, mu, sig):
        return mu + sig * self.N.sample(mu.shape)

    ########################################################################
    #                         END OF YOUR CODE                             #
    ########################################################################

    def forward(self, x):
        ########################################################################
        #    TODO: Create mean and variance                                    #
        ########################################################################

        x = torch.flatten(x, start_dim=1)

        x1 = self.relu(self.linear1_1(x))
        x1 = self.relu(self.linear2_1(x1))

        x2 = self.relu(self.linear1_2(x))
        x2 = self.relu(self.linear2_2(x2))

        sig = torch.exp(self.linear3_1(x1))
        mu = self.linear3_2(x2)

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

        # reparameterize to find z
        z = self.reparameterize(mu, sig)

        # loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)

        return z


# decoder: same as before

# autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim=[100, 50]):
        super(VarAutoencoder, self).__init__()

        self.encoder = VarEncoder(latent_dims, s_img, hdim)
        self.decoder = Decoder(latent_dims, s_img, hdim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y
