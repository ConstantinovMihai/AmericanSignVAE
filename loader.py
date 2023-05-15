import matplotlib.pyplot as plt
import pandas as pd
# Setup
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

plt.rcParams['figure.dpi'] = 200


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.data = pd.read_csv(img_dir)
        self.img_labels = self.data.loc[:, "label"]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        bools = [True for x in range(len(self.data.columns))]
        bools[0] = False
        # print(self.data.loc[idx, bools].to_numpy().reshape(28,28))
        image = self.data.loc[idx, bools].to_numpy().reshape(28, 28).astype('float32')

        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# mnist_data = CustomImageDataset('sign_mnist_train.csv', transform=ToTensor())
# # mnist_data = datasets.MNIST('./data',
# #              transform=transforms.ToTensor(),
# #             download=True)
#
# # Put it into a dataloader for easier handling in pytorch
# mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)
#
# # Show some example images
# nr_dig = 15
# fig, axs = plt.subplots(nr_dig, nr_dig, figsize=(10, 10))
# for i in range(int(nr_dig * nr_dig)):
#     x, _ = mnist_data[i]
#     ax = axs[i // nr_dig][i % nr_dig]
#     ax.imshow(x.view(28, 28), cmap='gray')
#     ax.axis('off')
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
