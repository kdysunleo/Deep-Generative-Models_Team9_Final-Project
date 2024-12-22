import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, root, norm=True, transform=None, subsample_size=None, use_class='all', **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        if subsample_size is not None:
            assert isinstance(subsample_size, int)

        self.root = root
        self.norm = norm
        self.transform = transform
        self.dataset = CIFAR10(
            f"{self.root}/cifar-10-batches-py", train=True, download=True, transform=transform, **kwargs
        )
        self.subsample_size = subsample_size

        # remain 2 classes
        if use_class == "all":
            remain_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # [3, 5, 7] # automobile, bird, cat, dog, horse

        else:
            remain_class = [1, 2] # [3, 5, 7] # automobile, bird


        self.images = []
        for idx in range(len(self.dataset)):
            img, label = self.dataset[idx]

            if label in remain_class:
                self.images.append(img)


    def __getitem__(self, idx):
        # img, _ = self.dataset[idx]
        img = self.images[idx]
        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0
        return torch.tensor(img).permute(2, 0, 1).float()

    def __len__(self):
        # return len(self.dataset) if self.subsample_size is None else self.subsample_size
        return len(self.images) if self.subsample_size is None else self.subsample_size


if __name__ == "__main__":
    root = "/data/kushagrap20/datasets/"
    dataset = CIFAR10Dataset(root)
    print(dataset[0].shape)
