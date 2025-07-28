"""
Data generator for training 2-head/decoders U-Net models
This script is used to load data from disk to memory (to GPU or CPU) and apply augmentations to the data.
"""

import os.path
import numpy as np
import rasterio
import torch
from kornia.geometry import vflip, hflip
from torch import rot90
from torch.utils.data import Dataset
from tqdm import tqdm


def load_dataset(image_paths, label1_paths, label2_paths, weight_paths=None, channel_count=3, data_device='cpu',
                 patch_size=512, output_folder="test"):
    # check if data exists in memory
    try:
        images = torch.load(os.path.join(output_folder, "images.pt")).to(data_device)
        labels = torch.load(os.path.join(output_folder, "labels.pt")).to(data_device)
        if weight_paths is not None:
            weights = torch.load(os.path.join(output_folder, "weights.pt")).to(data_device)
            print(f"Data loaded from {output_folder} folder.")
            return images, labels, weights
        else:
            print(f"Data loaded from {output_folder} folder. No weights found.")
            return images, labels, None

    # load images and labels to memory
    except FileNotFoundError:
        num_images = len(image_paths)
        images = torch.empty((num_images, channel_count, patch_size, patch_size), dtype=torch.uint8)
        labels = torch.empty((num_images, 2, patch_size, patch_size), dtype=torch.float32)
        weights = torch.empty((num_images, 1, patch_size, patch_size), dtype=torch.uint8)
        for i, (image_path, label1_path, label2_path) in tqdm(enumerate(zip(image_paths, label1_paths, label2_paths)),
                                                              total=num_images, desc="Loading data to memory: "):
            # load image and label
            with rasterio.open(image_path) as src:
                _image = src.read()
            _image = _image[:channel_count]
            with rasterio.open(label1_path) as src:
                label1 = src.read(1)
            with rasterio.open(label2_path) as src:
                label2 = src.read(1)  # multi class with 10pixels distance errosion
            # label2 = np.where(label2 == 1, label1, 0).astype(np.uint8)
            # label1 = np.where(label1 == 0, 0, 1).astype(np.uint8)  # binary for head one

            _label = np.stack([label2, label1], axis=0)
            images[i] = torch.tensor(_image, dtype=torch.uint8)
            labels[i] = torch.tensor(_label, dtype=torch.float32)

            if weight_paths is not None:
                with rasterio.open(weight_paths[i]) as src:
                    _weight = src.read(1)
                # _weight = np.where(_weight == 0, 0, 1)
                _weight = np.expand_dims(_weight, axis=0).astype(np.uint8)
                weights[i] = torch.tensor(_weight, dtype=torch.uint8)

        print(f"The memory size of the images is: {images.element_size() * images.nelement() / (1024 ** 3)} GB")
        print(f"The memory size of the labels is: {labels.element_size() * labels.nelement() / (1024 ** 3)} GB")

        # save tensors to memory
        if output_folder is not None:
            torch.save(images, os.path.join(output_folder, "images.pt"))
            torch.save(labels, os.path.join(output_folder, "labels.pt"))
            if weight_paths is not None:
                torch.save(weights, os.path.join(output_folder, "weights.pt"))
        if weight_paths is not None:
            print(
                f"The memory size of the weights is: {weights.element_size() * weights.nelement() / (1024 ** 3)} GB")
            weights = weights.to(data_device)
        else:
            weights = None
        images, labels = images.to(data_device), labels.to(data_device)

        return images, labels, weights


class TensorDataGenerator(Dataset):
    def __init__(self, image_paths, label1_paths, label2_paths, weight_paths=None, patch_size=128,
                 device='cpu', data_device='cpu', output_folder="test", pixel_dist=None):
        self.patch_size = patch_size
        self.device = device
        self.data_device = data_device
        self.label1_paths = label1_paths
        self.label2_paths = label2_paths
        self.image_paths = image_paths
        self.weight_paths = weight_paths
        self.pixel_dist = pixel_dist  # distance for second head label

        self.channel_count = 3
        # augmentation chances
        self.vflip_chance = 0.5
        self.hflip_chance = 0.5
        self.rot90_chance = 0.5
        self.brightness_chance = 0.5
        self.channel_drop_chance = 1 / (self.channel_count * 2) if self.channel_count > 1 else 0.
        self.pixel_drop_chance = 0.1
        self.images_written = 0  # Max 100
        self.pixel_noise_chance = 0.25
        self.channel_noise_chance = 0.125
        self.pixel_drop_p = 0.01
        self.vflip_chance = 0.5
        self.hflip_chance = 0.5
        self.images, self.labels, self.weights = load_dataset(image_paths=self.image_paths,
                                                              label1_paths=self.label1_paths,
                                                              label2_paths=self.label2_paths,
                                                              weight_paths=self.weight_paths,
                                                              data_device=self.data_device,
                                                              output_folder=output_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        X = self.images[index].clone()/255
        y = self.labels[index].clone()
        y[1] = torch.where(y[1] == 0, 0, 1) # complete building mask for first head
        if self.pixel_dist is not None:
            y[0] = torch.where(y[0] > self.pixel_dist, 1, 0)  # second head label is distance mask
        if self.weights is not None:
            w = self.weights[index].clone()

        # Optionally apply your augmentations here
        file_dict = {'X': X.float().to(self.device),
                     'y': y.long().to(self.device)}
        if self.weights is not None:
            file_dict['w'] = w.long().to(self.device)

        return file_dict

    def augment(self, X, y, device, w=None, sigma_n_hyp=0.03, sigma_u_hyp=0.3):
        y = y.float()
        b, c, h, w_ = X.shape
        # flipping horizontal
        hflip_coin = torch.floor(torch.rand((b, 1, 1, 1), device=device) + self.hflip_chance)
        X = X * (1 - hflip_coin) + hflip(X) * hflip_coin
        y = y * (1 - hflip_coin) + hflip(y) * hflip_coin
        if w is not None:
            w = w * (1 - hflip_coin) + hflip(w) * hflip_coin

        # flipping vertical
        vflip_coin = torch.floor(torch.rand((b, 1, 1, 1), device=device) + self.vflip_chance)
        X = X * (1 - vflip_coin) + vflip(X) * vflip_coin
        y = y * (1 - vflip_coin) + vflip(y) * vflip_coin
        if w is not None:
            w = w * (1 - vflip_coin) + vflip(w) * vflip_coin

        # Rotation 90, 180 and 270
        rot90_coin = torch.floor(torch.rand((b, 1, 1, 1), device=device) + self.rot90_chance)
        k = np.random.randint(1, 3)
        X = X * (1 - rot90_coin) + rot90(X, k=k, dims=(2, 3)) * rot90_coin
        y = y * (1 - rot90_coin) + rot90(y, k=k, dims=(2, 3)) * rot90_coin
        if w is not None:
            w = w * (1 - rot90_coin) + rot90(w, k=k, dims=(2, 3)) * rot90_coin

        # Brightness -> per images: Changes brightness between 0.8 and 1.2
        brightness_coin = torch.floor(torch.rand((b, 1, 1, 1), device=device) + self.brightness_chance)
        X = X * (1 - brightness_coin) + torch.clip(
            (X + (torch.rand(size=(b, 1, 1, 1), device=device) * 0.4 - 0.2)), 0,
            1) * brightness_coin

        # augmentation from delfors (not using mask)
        # pixelwise noise (each image has a chance of having random noise per pixel)
        # additive noise (uniform and normal/gaussian distribution noise)
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .015 + sigma_n_hyp
        X += ((torch.randn_like(X).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .05 + sigma_u_hyp
        X += ((torch.rand_like(X) * sigma) * noise_coin_u)

        # multiplicative noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noisenum_classes_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005 + sigma_n_hyp
        X += ((X * torch.randn_like(X).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .015 + sigma_u_hyp
        X += ((X * torch.rand_like(X) * sigma) * noise_coin_u)

        # channelwise noise (each image has a chance of having random noise per channel)
        # additive noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .015 + sigma_n_hyp
        X += ((torch.randn((b, c, 1, 1), device=device).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .05 + sigma_u_hyp
        X += ((torch.rand((b, c, 1, 1), device=device) * sigma) * noise_coin_u)

        # multiplicative noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005 + sigma_n_hyp
        X += ((X * torch.randn((b, c, 1, 1), device=device).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .015 + sigma_u_hyp
        X += ((X * torch.rand((b, c, 1, 1), device=device) * sigma) * noise_coin_u)

        # channel dropout
        channel_droput_coin = torch.floor(
            torch.rand((b, c, 1, 1), device=device) + (1 - self.channel_drop_chance))
        # iterate through noise chances of every image in the batch and make all channels' dropout value 1 if all channels are zero
        # location of zero in the channel_droput_coin is dropout channel location
        X *= torch.stack(
            [_image + 1 if (_image.sum(dim=0) == 0).all() else _image for _image in channel_droput_coin])

        # pixel dropout
        X *= torch.clip(
            torch.floor(torch.rand((b, c, h, w_), device=device) + (1 - self.pixel_drop_p)) +
            torch.floor(torch.rand((b, 1, 1, 1), device=device) + (1 - self.pixel_drop_chance)),
            max=1
        )

        return X, y.long(), w
