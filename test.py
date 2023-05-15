import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from Conv_VAE import ConvVarAutoencoder
from loader import CustomImageDataset

device = 'cuda'


def test(epoch, model, loader):
    model.eval()
    test_loss = 0
    test_losses = []
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(device)
            recon_batch = model(data)
            test_loss += ((data - recon_batch) ** 2).sum() + model.encoder.kl
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(128, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction/' + str(epoch) + '.png', nrow=n)

    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    test_losses.append(test_loss)



normalize = transforms.Normalize(160, 50)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
mnist_data = CustomImageDataset('sign_mnist_test.csv', transform=transform)
# mnist_data = datasets.MNIST('./data',
#                             transform=transforms.ToTensor(),
#                             download=True)

# Put it into a dataloader for easier handling in pytorch
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)

mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)

model = ConvVarAutoencoder().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
writer = SummaryWriter()
epochs = 5
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    # Test
    test_loss = test(epoch, model, mnist_loader)
    # Write metrics to Tensorboard
    #writer.add_scalars("Loss", {'Train': test_loss}, epoch)