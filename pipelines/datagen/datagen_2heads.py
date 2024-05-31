"""
Data generator for training 2-head/decoders U-Net models
"""

import numpy as np
import torch
from kornia.geometry import vflip, hflip
from torch import rot90
from torch.utils.data import Dataset
import rasterio


class DataGenerator(Dataset):
    def __init__(self, image_paths, label1_paths, label2_paths, weight_paths=None, json_paths=None,
                 batch_size=32, patch_size=128, device='cpu'):
        'Initialization'
        self.batch_size = batch_size
        self.label1_paths = label1_paths
        self.label2_paths = label2_paths
        self.image_paths = image_paths
        self.weight_paths = weight_paths
        self.json_paths = json_paths
        self.patch_size = patch_size
        self.device = device
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.image_paths)

    def __getitem__(self, index):
        # load image and label
        with rasterio.open(self.image_paths[index]) as src:
            _image = src.read() / 255
        _image = _image[:self.channel_count]
        with rasterio.open(self.label1_paths[index]) as src:
            label1 = src.read(1)
        with rasterio.open(self.label2_paths[index]) as src:
            label2 = src.read(1)
        _label = np.stack([label1, label2], axis=0)
        _label = np.where(_label == 0, 0, 1).astype(np.uint8)
        X, y = torch.tensor(_image).float().to(self.device), torch.tensor(_label).long().to(self.device)
        file_dict = {'X': X,
                     'y': y}

        if self.weight_paths is not None:
            with rasterio.open(self.weight_paths[index]) as src:
                _weight = src.read(1)
            _weight = np.where(_weight == 0, 0, 1)
            _weight = np.expand_dims(_weight, axis=0).astype(np.uint8)
            w = torch.tensor(_weight).long().to(self.device)
            file_dict['w'] = w

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
        channel_droput_coin = torch.floor(torch.rand((b, c, 1, 1), device=device) + (1 - self.channel_drop_chance))
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
