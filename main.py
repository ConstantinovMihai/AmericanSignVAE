# Setup
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from tqdm import tqdm

from loader import CustomImageDataset
from model import VarAutoencoder

plt.rcParams['figure.dpi'] = 200


# Additional Setup to use Tensorboard
# !pip install -q tensorflow
# %load_ext tensorboard
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_vae(train_loader, net, optimizer, device=device):
    """
    Trains variational autoencoder network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        device: whether the network runs on cpu or gpu
    """

    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # convert the inputs to run on GPU if set
        inputs = inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)

        loss = ((inputs - outputs) ** 2).sum() + net.encoder.kl
        # print(((inputs - outputs)**2).sum(), net.encoder.kl)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)



mnist_data = CustomImageDataset('sign_mnist_train.csv', transform=ToTensor())
# mnist_data = datasets.MNIST('./data',
#              transform=transforms.ToTensor(),
#             download=True)

# Put it into a dataloader for easier handling in pytorch
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)

# Set the number of dimensions of the latent space
latent_dim = 2
s_img = np.size(mnist_data[1][0], axis=2)
hdims = [100, 50]
# Create a writer to write to Tensorboard
writer = SummaryWriter()

# Create instance of Autoencoder

CVAE = VarAutoencoder(latent_dim, s_img, hdims).to(device)

# Create loss function and optimizer
criterion = F.mse_loss

optimizer = optim.Adam(CVAE.parameters(), lr=5e-5, weight_decay=0.5)

# Set the number of epochs to for training
epochs = 5
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    # Train on data
    train_loss = train_vae(mnist_loader, CVAE, optimizer, device)
    if (train_loss.isnan()):
        print("poopies")
        break
    else:
        print(train_loss.item())
    # Write metrics to Tensorboard
    writer.add_scalars("Loss", {'Train': train_loss}, epoch)
