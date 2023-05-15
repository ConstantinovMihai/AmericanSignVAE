# Setup
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


# plot latent space

def plot_latent(model, data_loader, num_batches=100, device=device):
    '''
    Plots position of data in latent space (which is either 2D or 3D)

    Args:
      autoencoder: pytorch network that contains an encoder subnetwork
      data_loader: the data we want to plot in latent space
      num_batches: number of batches to use in for the plot

    '''
    # Iterate over all data
    plt.rcParams['figure.figsize'] = (5, 3)
    plt.rcParams['figure.dpi'] = 144

    for idx, data in enumerate(data_loader):

        x, y = data
        model(x.to(device))
        z = model.z.to(device)  # Encode image data
        z = z.to('cpu').detach().numpy()  # Get numpy version of data in latent space

        # 2D latent space (single image)
        if np.size(z, axis=1) == 2:
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')  # Add data to plot

        # ------------------------------------------------------------------------
        # 3D latent space (4 images: 3D image, and 3 x 2D projections onto xy, xz
        # and yz)
        if np.size(z, axis=1) == 3:
            if idx == 0:  # initialize at first iteration
                plt.rcParams['figure.figsize'] = (5, 5)
                fig1 = plt.figure()
                plt.rcParams['figure.figsize'] = (15, 5)
                fig2 = plt.figure()
                ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
                ax2 = fig2.add_subplot(1, 3, 1)
                ax3 = fig2.add_subplot(1, 3, 2)
                ax4 = fig2.add_subplot(1, 3, 3)
                ax1.grid(False)
                # Hide axes ticks
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_zticks([])

                # set labels
                ax1.set_xlabel('dimension 1')
                ax1.set_ylabel('dimension 2')
                ax1.set_zlabel('dimension 3')
                ax2.set_xlabel('dimension 1')
                ax2.set_ylabel('dimension 2')
                ax3.set_xlabel('dimension 1')
                ax3.set_ylabel('dimension 3')
                ax4.set_xlabel('dimension 2')
                ax4.set_ylabel('dimension 3')

            ax1.scatter3D(z[:, 0], z[:, 1], z[:, 2], c=y, cmap='tab10');
            ax2.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10');
            ax3.scatter(z[:, 0], z[:, 2], c=y, cmap='tab10');
            ax4.scatter(z[:, 1], z[:, 2], c=y, cmap='tab10');

            if idx > num_batches:
                fig1.tight_layout()
                fig2.tight_layout()

        # Stop if we've reach the maximum number of batches
        if idx > num_batches:
            if np.size(z, axis=1) == 2:
                plt.colorbar()
            break
    plt.show()


def plot_reconstructed(autoencoder, r0=(-3, 3), r1=(-3, 3), n=12):
    '''
    Plots reconstruction from a decoder for variables between r0 bounds and r1 bounds with a stepsize of n

    Args:
      autoencoder: pytorch network that contains a decoder subnetwork with input size of two
      r0: bounds of first latent variable
      r1: bounds of second latent variable
      n: sqrt of amount of numbers to decode

    '''

    w = 28  # side length of MNIST images

    img = np.zeros((n * w, n * w))  # prepare large image to hold all decoded images

    # Iterate over bounds with stepsize n
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)  # Create a tensor from inputs

            x_hat = autoencoder.decoder(z)  # Run decoder

            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()  # Reshape to mnist sized image

            # Place decoded image in larger image:
            #
            #  _____________________
            # |r00 |r01 | ...  | r0n|
            # |____|____|      |____|
            # |r10 |                |
            # |____|             :  |
            # |  :               :  |
            # |  :               :  |
            # |____             ____|
            # |rn0 |    ....   |rnn |
            # |____|___________|____|
            #
            # in which row is the reconstructed image produced by the decoder
            # for the latent vector [r0[i], r1[j]]

            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat

    plt.imshow(img, extent=[*r0, *r1])  # Show complete image
