# Setup
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import ToTensor
from Conv_VAE import ConvVarAutoencoder
from loader import CustomImageDataset
from test import test
from utils import plot_latent, plot_reconstructed

plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Additional Setup to use Tensorboard
# !pip install -q tensorflow
# %load_ext tensorboard
mnist_data = CustomImageDataset('sign_mnist_train.csv', transform=ToTensor())
# mnist_data = datasets.MNIST('./data',
#              transform=transforms.ToTensor(),
#             download=True)

# Put it into a dataloader for easier handling in pytorch
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)

model = ConvVarAutoencoder().to(device)
model.load_state_dict(torch.load("model_good_50.pt", map_location=device))
#plot_latent(model, mnist_loader)
#plot_reconstructed(model)