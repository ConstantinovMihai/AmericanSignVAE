import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from Conv_VAE import ConvVarAutoencoder
from loader import CustomImageDataset

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

        loss = ((inputs - outputs) ** 2).sum() + net.kl
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)


# mnist_data = CustomImageDataset('sign_mnist_train.csv', transform=ToTensor())
mnist_data = datasets.MNIST('./data',
             transform=transforms.ToTensor(),
            download=True)

# Put it into a dataloader for easier handling in pytorch
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)

# Create a writer to write to Tensorboard
writer = SummaryWriter()

# Create instance of Autoencoder

# model = VarAutoencoder(latent_dim, s_img, hdims).to(device)
model = ConvVarAutoencoder().cuda()
# Create loss function and optimizer
criterion = F.mse_loss

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.1)

# Set the number of epochs to for training
epochs = 50
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    # Train on data
    train_loss = train_vae(mnist_loader, model, optimizer, device)
    print(train_loss.item(), model.kl.item(), end='\n')
    # Write metrics to Tensorboard
    writer.add_scalars("Loss", {'Train': train_loss}, epoch)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "model.pt")

