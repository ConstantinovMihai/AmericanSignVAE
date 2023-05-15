# Setup
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import try_gpu

plt.rcParams['figure.dpi'] = 200
s_img=28

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Sampling vector
        self.fc3 = nn.Linear(2048, 2048)
        self.fc_bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 7 * 7 * 16)
        self.fc_bn4 = nn.BatchNorm1d(7 * 7 * 16)
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, z):


        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 7, 7)
        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        z = self.conv8(conv7).view(-1, 1, 28, 28)

        return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(7 * 7 * 16, 2048)
        self.fc_bn1 = nn.BatchNorm1d(2048)
        self.fc21 = nn.Linear(2048, 2048)
        self.fc22 = nn.Linear(2048, 2048)
        # Sampling vector
        self.fc3 = nn.Linear(2048, 2048)
        self.fc_bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 7 * 7 * 16)
        self.fc_bn4 = nn.BatchNorm1d(7 * 7 * 16)

        self.relu = nn.ReLU()

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(try_gpu())  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(try_gpu())
        self.kl = 0


    def kull_leib(self, mu, sigma):
        res = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return res

    def reparameterize(self, mu, sig):
        return mu + sig * self.N.sample(mu.shape)
    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 7 * 7 * 16)
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mu = self.fc21(fc1)
        sig = self.fc22(fc1).exp()
        # reparameterize to find z
        z = self.reparameterize(mu, sig)
        # loss between N(0,I) and learned distribution
        self.kl = self.kull_leib(mu, sig)

        return z
# autoencoder
class ConvVarAutoencoder(nn.Module):
    def __init__(self):
        super(ConvVarAutoencoder, self).__init__()

        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
