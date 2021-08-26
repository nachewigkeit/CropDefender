import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import h5py
from time import time


class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100):
        self.data_path = data_path
        self.secret_size = secret_size

        self.dataset = None
        with h5py.File(self.data_path, 'r') as file:
            self.dataset_len = len(file["image"])

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.data_path, 'r')["image"]
        img_cover = self.dataset[idx]
        img_cover = img_cover.transpose((2, 0, 1))
        img_cover = torch.from_numpy(img_cover).float()

        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        return img_cover, secret

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    # dataset = StegaData(data_path='F:\\VOCdevkit\\VOC2012\\JPEGImages')
    # print(len(dataset))
    # img_cover, secret = dataset[10]
    # print(type(img_cover), type(secret))
    # print(img_cover.shape, secret.shape)

    dataset = StegaData(data_path="./data/images.hdf5", secret_size=100)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(image_input.shape, secret_input.shape)
    print(image_input[0].max())
    print(image_input.dtype)
